"""The only places the LLM is used - each one a judgement call that rules can't make.

  show profile  (1 call/show)      reads the loglines -> characters + show-specific reveal/threat words
  moment miner  (1 call/episode)   tags hook and cliffhanger beats BY BEAT ID (never timestamps)
  cold viewer   (1 call/5 clips)   judges ONLY the clip text, with no show context
  rerank        (N calls)          orders the final top 10 head-to-head (relative, not absolute, scores)

Every ID the model returns is validated by code; unknown IDs are dropped and counted.
Calls run on a thread pool; progress is reported from the calling thread, so the UI can show it.
"""
from __future__ import annotations

import random
import string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .candidates import Promo
from .llm import LLM, CacheMiss, Cancelled, LLMError, prompt_hash
from .parse import Episode

# ----------------------------------------------------------------------------- v4 prompts (byte-identical
# to the submitted version, so v4 runs keep hitting the cache - do not edit)
MINER_SYSTEM = """You are a senior promo editor for vertical AI microdramas (ReelShort / DramaBox style ads on TikTok, Reels, Shorts).
You read timestamped screenplay beats and pick the beats a promo could OPEN on (hooks) and CUT on (cliffhangers).
Reply with a single JSON object and nothing else."""

MINER_TEMPLATE = """Show: {show}
Episode {num} logline: {logline}

Beats (id, time, type, text). CAMERA/ACTION/SFX describe what is seen and heard; other lines are dialogue:
{beats}

Pick up to 6 HOOK beats: something that stops a scrolling stranger within 3 seconds with ZERO context -
a slap or strike, violence, a public humiliation, a shocking or cruel line, a reveal, a threat, a proposal.
Pick up to 6 CLIFFHANGER beats: the clip should END right after this beat, with a big question unanswered
(an interruption, a threat or ultimatum, a reveal about to land, someone bursting in, a shocked face before the answer).
A calm, resolved or tender beat is NOT a cliffhanger.

Rules: use beat ids exactly as listed above; never invent ids or timestamps. strength is 1-10, be harsh (most 4-7).
JSON shape:
{{"hooks": [{{"beat": "<id>", "type": "slap|violence|humiliation|shocking_line|reveal|threat|romance|other", "strength": 7, "why": "<6-12 words>"}}],
 "cliffhangers": [{{"beat": "<id>", "strength": 7, "open_question": "<what the viewer needs to know>"}}]}}"""

JUDGE_SYSTEM = """You are a 24-year-old in the US scrolling TikTok. You have NEVER seen or heard of this show.
You are shown short clips as screenplay text (what you would see and hear, with times from the start of the clip).
You judge them honestly and harshly, like a real distracted viewer. Reply with a single JSON object and nothing else."""

JUDGE_TEMPLATE = """Judge each clip below as a social-media promo.

{clips}

For EACH clip score 0-10 (most clips deserve 3-7; give 9-10 only to exceptional ones):
- hook: would what happens between +0.0s and +3.0s stop your scroll?
- cliffhanger: when the clip ends, is something important unresolved so you NEED to know what happens next? (a resolved or calm ending scores low)
- standalone: from this clip alone, can you tell who is who and what is at stake?
Also write: hook_moment (what grabs you in the first 3s, <=15 words), open_question (the question left hanging, <=15 words),
confusing (anything you could not follow, or ""), why (one line: why this promo works or fails).

JSON shape: {{"clips": [{{"clip": 1, "hook": 6, "cliffhanger": 7, "standalone": 5, "hook_moment": "...", "open_question": "...", "confusing": "", "why": "..."}}]}}"""

# ----------------------------------------------------------------------------- v5 prompts
MINER_TEMPLATE_V5 = MINER_TEMPLATE.replace(
    "a slap or strike, violence, a public humiliation, a shocking or cruel line, a reveal, a threat, a proposal.",
    "a slap or strike, violence, a public humiliation, a shocking or cruel line, a reveal, a threat, a proposal,\n"
    "or an EMOTIONAL REVEAL (someone believed dead reappears, a hidden relationship or identity comes out, a long-lost\n"
    "reunion) - tender in tone, but a huge twist for the viewer.",
).replace("reveal|threat|romance|other", "reveal|emotional_reveal|threat|romance|other")

JUDGE_TEMPLATE_V5 = JUDGE_TEMPLATE.replace(
    "confusing (anything you could not follow, or \"\"), why (one line: why this promo works or fails).",
    "confusing (anything you could not follow, or \"\"), why (one line: why this promo works or fails).\n"
    "Then answer three yes/no checks about the ENDING (true/false):\n"
    "- unanswered: does the final line or moment leave a SPECIFIC question unanswered?\n"
    "- stakes: is someone's safety, relationship or secret at stake in the last 4 seconds?\n"
    "- resolved: is anything resolved (answered, settled, made safe) in the last 4 seconds?",
).replace('"confusing": "", "why": "..."}}]}}',       # doubled braces: this is a .format() template
          '"confusing": "", "why": "...", "unanswered": true, "stakes": true, "resolved": false}}]}}')

PROFILE_SYSTEM = """You are a story editor. You read episode summaries of a vertical microdrama and describe the show.
Reply with a single JSON object and nothing else."""

PROFILE_TEMPLATE = """Show: {show}
{material}

Describe this show for a promo editor. Return JSON:
{{"genre": "<2-5 words>",
 "characters": [{{"name": "<as written in the script>", "role": "<protagonist|love interest|villain|family|other>", "relationships": "<one short line>"}}],
 "reveal_words": ["<words or short phrases that, when they appear in a line, signal a big reveal or status flip IN THIS SHOW>"],
 "threat_words": ["<words or short phrases that signal danger or a threat IN THIS SHOW>"]}}
Use at most 8 characters and at most 15 words in each word list. Use words that actually appear in the episodes."""

RERANK_SYSTEM = """You are a senior promo editor choosing which promos to post first on TikTok, Reels and Shorts.
You compare clips against each other. Reply with a single JSON object and nothing else."""

RERANK_TEMPLATE = """Below are {n} candidate promos as screenplay text (times are from the start of each clip).
A great promo stops the scroll in the first 3 seconds, makes sense to someone who has never seen the show,
and cuts at a moment where the viewer NEEDS to know what happens next.

{clips}

Order ALL {n} clips from best promo to worst. Use each label exactly once.
JSON shape: {{"order": [{example}]}}"""

PROMPT_VERSIONS = {
    "miner:v4": prompt_hash(MINER_SYSTEM, MINER_TEMPLATE),
    "miner:v5": prompt_hash(MINER_SYSTEM, MINER_TEMPLATE_V5),
    "judge:v4": prompt_hash(JUDGE_SYSTEM, JUDGE_TEMPLATE),
    "judge:v5": prompt_hash(JUDGE_SYSTEM, JUDGE_TEMPLATE_V5),
    "profile:v1": prompt_hash(PROFILE_SYSTEM, PROFILE_TEMPLATE),
    "rerank:v1": prompt_hash(RERANK_SYSTEM, RERANK_TEMPLATE),
}

_SOFT_FAIL = (LLMError, CacheMiss)      # a failed call degrades to rules for that item; Cancelled stops the run


def run_parallel(items: list, fn, workers: int, tick=None, cancel: threading.Event | None = None) -> list:
    """Run fn(item) on a thread pool; return results in INPUT order (so runs are reproducible).
    `tick(done, total)` is called from this (the caller's) thread after each item finishes - safe for UIs.
    If tick raises (e.g. the UI was cancelled) or fn raises Cancelled, remaining work is cancelled."""
    results: list = [None] * len(items)
    if not items:
        return results
    ex = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    try:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        done = 0
        for f in as_completed(futs):
            results[futs[f]] = f.result()
            done += 1
            if tick:
                tick(done, len(items))
    except BaseException:
        if cancel is not None:
            cancel.set()
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    return results


# ----------------------------------------------------------------------------- show profile
def show_profile(llm: LLM, show: str, episodes: list[Episode], log=print) -> dict:
    """One call: characters + show-specific reveal/threat words. Words are kept only if they really
    occur in the scripts (a code check against invented vocabulary)."""
    if any(e.logline for e in episodes):
        material = "Episode loglines:\n" + "\n".join(f"- Ep{e.number}: {e.logline}" for e in episodes if e.logline)
    else:
        material = "Opening beats of each episode:\n" + "\n".join(
            f"- Ep{e.number}: " + " / ".join(b.text[:90] for b in e.beats[:12] if b.kind != "CAMERA")
            for e in episodes)
    try:
        data = llm.json(PROFILE_SYSTEM, PROFILE_TEMPLATE.format(show=show, material=material[:60000]), tag="profile")
    except _SOFT_FAIL as e:
        log(f"  [profile] failed ({e}) - continuing with generic words only")
        return {"error": str(e)}
    if not isinstance(data, dict):
        return {"error": "profile reply was not an object"}
    corpus = " ".join(b.text.lower() for e in episodes for b in e.beats)
    prof = {"genre": str(data.get("genre", ""))[:60], "characters": [], "reveal_words": [], "threat_words": [],
            "dropped_words": []}
    for c in (data.get("characters") or [])[:8]:
        if isinstance(c, dict) and c.get("name"):
            prof["characters"].append({k: str(c.get(k, ""))[:120] for k in ("name", "role", "relationships")})
    for kind in ("reveal_words", "threat_words"):
        for w in (data.get(kind) or [])[:15]:
            if not isinstance(w, str) or not 2 < len(w.strip()) < 40:
                continue
            (prof[kind] if w.strip().lower() in corpus else prof["dropped_words"]).append(w.strip())
    log(f"  [profile] {prof['genre'] or '?'} · {len(prof['characters'])} characters · "
        f"{len(prof['reveal_words'])} reveal + {len(prof['threat_words'])} threat words "
        f"({len(prof['dropped_words'])} dropped: not in scripts)")
    return prof


# ----------------------------------------------------------------------------- moment miner
def mine_moments(llm: LLM, show: str, episodes: list[Episode], log=print, version: str = "v4",
                 workers: int = 1, tick=None) -> tuple[dict, dict, dict]:
    """Returns (hook_strength, cliff_strength, stats) keyed by beat id, strengths in [0,1]."""
    template = MINER_TEMPLATE_V5 if version == "v5" else MINER_TEMPLATE

    def one(ep: Episode):
        prompt = template.format(show=show, num=ep.number, logline=ep.logline or "(none)",
                                 beats="\n".join(b.render() for b in ep.beats))
        try:
            return ep, llm.json(MINER_SYSTEM, prompt, tag=f"miner:ep{ep.number:02d}"), None
        except _SOFT_FAIL as e:
            return ep, None, e

    results = run_parallel(episodes, one, workers, tick, llm.cancel)
    hooks, cliffs, notes = {}, {}, {}
    stats = {"episodes": 0, "failed_episodes": [], "tags": 0, "invalid_ids": 0, "emotional_reveals": 0}
    for ep, data, err in results:                       # merged in episode order -> deterministic
        if err is not None:
            if not isinstance(err, CacheMiss):
                log(f"  [miner] episode {ep.number}: {err} - falling back to rules for this episode")
            stats["failed_episodes"].append(ep.number)
            continue
        if not isinstance(data, dict):
            stats["failed_episodes"].append(ep.number)
            continue
        valid = {b.id for b in ep.beats}
        stats["episodes"] += 1
        for kind, store in (("hooks", hooks), ("cliffhangers", cliffs)):
            for item in data.get(kind, []) or []:
                if not isinstance(item, dict):
                    continue
                bid = str(item.get("beat", "")).strip()
                if bid not in valid:
                    stats["invalid_ids"] += 1
                    continue
                try:
                    s = max(0.0, min(10.0, float(item.get("strength", 0)))) / 10
                except (TypeError, ValueError):
                    continue
                store[bid] = max(store.get(bid, 0.0), s)
                notes.setdefault(bid, {}).update({k: v for k, v in item.items() if k in ("type", "why", "open_question")})
                if kind == "hooks" and str(item.get("type", "")) == "emotional_reveal":
                    notes[bid]["emotional_strength"] = max(notes[bid].get("emotional_strength", 0.0), s)
                    stats["emotional_reveals"] += 1
                stats["tags"] += 1
        log(f"  [miner] ep{ep.number:02d}: {len(data.get('hooks', []) or [])} hooks, "
            f"{len(data.get('cliffhangers', []) or [])} cliffs")
    stats["notes"] = notes
    return hooks, cliffs, stats


def blend_signals(sig: dict, hooks: dict, cliffs: dict, w_llm: float = 0.6, notes: dict | None = None,
                  emotional_exempt: bool = False) -> None:
    """Mix model tags into the rule signals in place. Untagged beats keep only the rule share, so a
    beat the model ignored can still win if the rules love it (and vice versa)."""
    for bid, s in sig.items():
        s["hook_rule"], s["cliff_rule"] = s["hook"], s["cliff"]
        s["hook"] = (1 - w_llm) * s["hook"] + w_llm * hooks.get(bid, 0.0)
        s["cliff"] = (1 - w_llm) * s["cliff"] + w_llm * cliffs.get(bid, 0.0)
        if emotional_exempt and notes and notes.get(bid, {}).get("emotional_strength", 0.0) >= 0.7:
            s["resolve"] = 0.0                       # a tender reunion can still be the biggest twist
            s["emotional_reveal"] = True


# ----------------------------------------------------------------------------- cold-viewer judge
def render_clip(promo: Promo, ep_by_num: dict[int, Episode]) -> str:
    lines, offset = [], 0.0
    for i, seg in enumerate(promo.segments):
        if i:
            lines.append("   [CUT TO A DIFFERENT MOMENT]")
        for b in ep_by_num[seg.episode].beats_between(seg.start, seg.end):
            rel = offset + max(0.0, b.start - seg.start)
            who = f"{b.speaker}" + (f" ({b.emotion})" if b.emotion else "") if b.kind == "DIALOGUE" else b.kind
            lines.append(f"   +{rel:04.1f}s {who}: {b.text}")
        offset += seg.dur
    return "\n".join(lines)


def _as_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("yes", "true", "no", "false"):
        return v.strip().lower() in ("yes", "true")
    return None


def judge(llm: LLM, promos: list[Promo], ep_by_num: dict[int, Episode], batch: int = 5, log=print,
          version: str = "v4", workers: int = 1, tick=None) -> dict:
    template = JUDGE_TEMPLATE_V5 if version == "v5" else JUDGE_TEMPLATE
    chunks = [promos[i: i + batch] for i in range(0, len(promos), batch)]

    def one(args):
        idx, chunk = args
        clips = "\n\n".join(f"CLIP {k + 1} ({p.duration:.0f}s):\n{render_clip(p, ep_by_num)}"
                            for k, p in enumerate(chunk))
        try:
            return idx, chunk, llm.json(JUDGE_SYSTEM, template.format(clips=clips), tag=f"judge:{idx}"), None
        except _SOFT_FAIL as e:
            return idx, chunk, None, e

    stats = {"judged": 0, "failed": 0, "checks_answered": 0}
    for idx, chunk, data, err in run_parallel(list(enumerate(chunks)), one, workers, tick, llm.cancel):
        if err is not None:
            if not isinstance(err, CacheMiss):
                log(f"  [judge] batch {idx}: {err} - these keep rule scores only")
            stats["failed"] += len(chunk)
            continue
        if isinstance(data, list):
            data = {"clips": data}
        by_idx = {}
        for item in (data or {}).get("clips", []) or []:
            try:
                by_idx[int(item.get("clip"))] = item
            except (TypeError, ValueError, AttributeError):
                pass
        for k, p in enumerate(chunk):
            item = by_idx.get(k + 1)
            if not item:
                stats["failed"] += 1
                continue
            try:
                p.llm = {key: max(0.0, min(10.0, float(item.get(key, 0)))) for key in ("hook", "cliffhanger", "standalone")}
            except (TypeError, ValueError):
                stats["failed"] += 1
                continue
            p.llm.update({key: str(item.get(key, "")) for key in ("hook_moment", "open_question", "confusing", "why")})
            checks = {key: _as_bool(item.get(key)) for key in ("unanswered", "stakes", "resolved")}
            if all(v is not None for v in checks.values()):
                p.llm["checks"] = checks
                p.llm["check_score"] = (checks["unanswered"] + checks["stakes"] + (not checks["resolved"])) / 3
                stats["checks_answered"] += 1
            stats["judged"] += 1
    log(f"  [judge] {stats['judged']}/{len(promos)} clips judged")
    return stats


# ----------------------------------------------------------------------------- listwise rerank
def rerank(llm: LLM, promos: list[Promo], ep_by_num: dict[int, Episode], rounds: int = 3, seed: int = 7,
           log=print, workers: int = 1, tick=None) -> dict:
    """Ask for a full ordering of the top clips, `rounds` times with different shuffles and random
    letter labels (to cancel position/label bias). Invalid orderings are dropped and logged.
    Returns {promo_index: average_position} plus stats."""
    n = len(promos)
    if n < 2 or rounds < 1:
        return {"positions": {}, "valid_rounds": 0, "invalid_rounds": 0}

    def one(r: int):
        rng = random.Random(seed * 1000 + r)
        order = list(range(n))
        rng.shuffle(order)
        labels = rng.sample(string.ascii_uppercase, n)
        lab_to_idx = {labels[k]: order[k] for k in range(n)}
        clips = "\n\n".join(f"CLIP {labels[k]} ({promos[order[k]].duration:.0f}s):\n{render_clip(promos[order[k]], ep_by_num)}"
                            for k in range(n))
        prompt = RERANK_TEMPLATE.format(n=n, clips=clips, example=", ".join(f'"{x}"' for x in labels[:3]) + ", ...")
        try:
            data = llm.json(RERANK_SYSTEM, prompt, tag=f"rerank:{r}")
        except _SOFT_FAIL as e:
            return r, None, str(e)
        got = data.get("order") if isinstance(data, dict) else data
        got = [str(x).strip().upper() for x in got] if isinstance(got, list) else []
        if sorted(got) != sorted(labels):
            return r, None, f"not a complete permutation of the {n} labels"
        return r, [lab_to_idx[x] for x in got], None

    positions: dict[int, list[int]] = {i: [] for i in range(n)}
    valid = invalid = 0
    for r, ranking, err in run_parallel(list(range(rounds)), one, workers, tick, llm.cancel):
        if ranking is None:
            invalid += 1
            log(f"  [rerank] round {r} dropped: {err}")
            continue
        valid += 1
        for pos, idx in enumerate(ranking):
            positions[idx].append(pos)
    avg = {i: sum(p) / len(p) for i, p in positions.items() if p}
    log(f"  [rerank] {valid} valid / {invalid} invalid rounds")
    return {"positions": avg, "valid_rounds": valid, "invalid_rounds": invalid}
