"""The only places the LLM is used - each one a judgement call that rules can't make.

  show profile  (1 call/show)      reads the loglines -> characters + show-specific reveal/threat words
  moment miner  (1 call/episode)   tags hook and cliffhanger beats BY BEAT ID (never timestamps)
  cold viewer   (1 call/5 clips)   judges ONLY the clip text, with no show context
  rerank        (N calls)          orders the final top 10 head-to-head (relative, not absolute, scores)

Every ID the model returns is validated by code; unknown IDs are dropped and counted.
Calls run on a thread pool; progress is reported from the calling thread, so the UI can show it.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import string
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal

from .candidates import TIME_TOL, Promo
from .llm import LLM, CacheMiss, Cancelled, LLMError, prompt_hash
from .parse import Beat, Episode

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

# ----------------------------------------------------------------------------- v7 judge: one clip per call,
# pass/fail/uncertain checks with cited beat refs. The schema version is part of the prompt text, so any change
# to the contract changes every cache key: an old judgment can never be read back as a v7 one.
JUDGE_SCHEMA_VERSION = "judge-v7.2"
VERDICTS = ("pass", "fail", "uncertain")
MOMENT_TYPES = ("suspense", "resolution", "neutral")
SECTION_TEXT = {"hook": ("description", "mechanism"),
                "ending": ("central_question", "final_moment", "open_question", "reason"),
                "standalone": ("conflict", "blocking_confusion", "harmless_curiosity"),
                "transition": ("connection", "reason")}

JUDGE_SYSTEM_V7 = """You are a senior promo editor for short vertical drama ads. You review ONE candidate promo clip,
reading it as a first-time viewer who has never seen or heard of the show: you know only what the clip shows and says.
Judge strictly from the clip's lines. Never invent events, threats, motives or backstory that the lines do not show.
Reply with a single JSON object and nothing else."""

JUDGE_TEMPLATE_V7 = """Candidate promo {clip_id} | {duration:.1f} s | {n_seg} segment(s) | schema {schema}
Each line is one beat in playback order: ref, time in seconds from the start of the promo, type, text.
CAMERA, ACTION and SFX lines describe what is seen and heard; the other lines are dialogue (speaker, delivery, words).
"[partial ...]" marks a beat that was already under way when the promo starts, or is cut off when it ends.

{clip}

Opening window: +0.0 to +{hook_window:.1f} s. Final window: +{final_from:.1f} to +{duration:.1f} s.

Answer using ONLY the lines above.
1. hook: Does what STARTS in the opening window grab a scrolling stranger? Visual action counts as much as dialogue
   (fire, a blow, a fall, a shocking image). evidence = refs of opening-window lines that carry the hook.
2. ending: central_question = the main thing a viewer is waiting to find out while watching this clip.
   Then classify the final moment (final_moment_type) by what HAPPENS in the final window, not by its tone:
   "suspense" = something is still pending at the cut: a threat or decision not yet carried out, an action or
     line interrupted, a reveal about to land, a new question just raised;
   "resolution" = tension is released at the end: a goal achieved, a search succeeds, a danger escaped, a question
     answered, a conflict settled, a scene closed in calm or affection;
   "neutral" = nothing is pending or released.
   A quiet or tender moment is still "suspense" if something is pending.
   open_question = the specific question the FINAL moment leaves open ("what happens next?" is not specific).
   Then read the WHOLE clip again: is open_question already answered anywhere in it? If so set
   answered_within_clip to true. answered_by = the refs where the clip answers its own question or releases its
   tension (required for "resolution" and for answered_within_clip).
   The ending passes only for "suspense" with an open question the clip itself does not answer.
   evidence = refs in the final window that leave the question open. Base the question on those lines and do not
   invent a danger, threat or motive they do not show.
   NOT evidence of an open ending: music or sound cues, "cut to black", an episode-ending marker, or the word
   "cliffhanger" in a stage direction.
3. standalone: Can a first-time viewer follow the immediate conflict and what is at stake? They do not need names or
   backstory. Separate harmless curiosity (makes you want more) from confusion that stops you understanding what is
   happening. evidence = refs that establish the conflict.{transition_rule}

Verdicts are "pass", "fail" or "uncertain" (use uncertain when the lines do not let you decide). Scores are 0-10;
most clips deserve 3-7. Write in a neutral, professional editorial voice and name the concrete mechanism (immediate
danger, injustice, humiliation, identity mystery, interrupted confrontation, threat, betrayal). No slang, and no claims
about platforms, algorithms, virality or how a kind of clip "always" performs.

JSON shape:
{shape}"""

TRANSITION_RULE_V7 = """
4. transition: The promo hard-cuts between segments (the "=== HARD CUT ===" line). Using only these lines, can a
   viewer tell how the second segment relates to the first? Sharing a character or a word is not enough. An
   unexplained jump in place, time or situation that breaks the story fails, and so does a link that would need a
   caption, narration or knowledge from outside the clip. evidence = at least one ref from EACH side of the cut."""

RETRY_NOTE_V7 = ("\n\nYour previous reply was invalid: {errors}. Reply again with the complete JSON object in exactly "
                 "the shape above, using clip_id {clip_id} and only refs that appear in the clip.")


def _judge_shape(stitched: bool) -> str:
    d = {"clip_id": "<the id above>",
         "hook": {"verdict": "pass|fail|uncertain", "evidence": ["B01"],
                  "description": "<what happens in the opening window, max 20 words>", "mechanism": "<2-4 words>"},
         "ending": {"verdict": "pass|fail|uncertain", "evidence": ["B09"],
                    "central_question": "<what the viewer waits to find out during the clip>",
                    "final_moment_type": "suspense|resolution|neutral",
                    "final_moment": "<what is seen or heard in the final window, max 20 words>",
                    "open_question": "<the question left open, or empty>", "answered_within_clip": False,
                    "answered_by": [], "reason": "<one sentence>"},
         "standalone": {"verdict": "pass|fail|uncertain", "evidence": ["B02"],
                        "conflict": "<the immediate conflict and what is at stake, max 25 words>",
                        "blocking_confusion": "<what stops a new viewer understanding, or empty>",
                        "harmless_curiosity": "<questions that only add intrigue, or empty>"}}
    if stitched:
        d["transition"] = {"verdict": "pass|fail|uncertain", "evidence": ["B03", "B07"],
                           "connection": "<how segment 2 follows from segment 1, or empty>", "reason": "<one sentence>"}
    d["scores"] = {"hook": 5, "cliffhanger": 5, "standalone": 5}
    d["why_it_works"] = "<1-2 sentences specific to this clip: the concrete mechanism>"
    d["limitations"] = "<1 sentence, or empty>"
    return json.dumps(d, indent=1)


def judge_schema(stitched: bool) -> dict:
    """JSON Schema of the reply (sent only to providers configured for structured output)."""
    sec = lambda *extra: {"type": "object", "required": ["verdict", "evidence", *extra],
                          "properties": {"verdict": {"enum": list(VERDICTS)},
                                         "evidence": {"type": "array", "items": {"type": "string"}},
                                         **{k: {"type": "string"} for k in extra}}}
    props = {"clip_id": {"type": "string"}, "hook": sec(*SECTION_TEXT["hook"]),
             "ending": sec(*SECTION_TEXT["ending"]), "standalone": sec(*SECTION_TEXT["standalone"]),
             "scores": {"type": "object", "required": ["hook", "cliffhanger", "standalone"],
                        "properties": {k: {"type": "number"} for k in ("hook", "cliffhanger", "standalone")}},
             "why_it_works": {"type": "string"}, "limitations": {"type": "string"}}
    props["ending"]["properties"].update({"answered_within_clip": {"type": "boolean"},
                                          "final_moment_type": {"enum": list(MOMENT_TYPES)},
                                          "answered_by": {"type": "array", "items": {"type": "string"}}})
    props["ending"]["required"] += ["final_moment_type", "answered_within_clip", "answered_by"]
    req = ["clip_id", "hook", "ending", "standalone", "scores", "why_it_works", "limitations"]
    if stitched:
        props["transition"] = sec(*SECTION_TEXT["transition"])
        req.append("transition")
    return {"type": "object", "required": req, "properties": props}


PROMPT_VERSIONS = {
    "miner:v4": prompt_hash(MINER_SYSTEM, MINER_TEMPLATE),
    "miner:v5": prompt_hash(MINER_SYSTEM, MINER_TEMPLATE_V5),
    "judge:v4": prompt_hash(JUDGE_SYSTEM, JUDGE_TEMPLATE),
    "judge:v5": prompt_hash(JUDGE_SYSTEM, JUDGE_TEMPLATE_V5),
    "judge:v7": prompt_hash(JUDGE_SYSTEM_V7, JUDGE_TEMPLATE_V7, TRANSITION_RULE_V7, RETRY_NOTE_V7,
                            _judge_shape(True), JUDGE_SCHEMA_VERSION),
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
        s["hook_llm"], s["cliff_llm"] = hooks.get(bid, 0.0), cliffs.get(bid, 0.0)   # raw miner tags (v7 rule opinions)
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


# ----------------------------------------------------------------------------- v7 cold-viewer judge
@dataclass
class ClipRef:
    """One beat as it plays inside a promo: `rel_*` are seconds from the start of the promo."""
    ref: str
    beat: Beat
    segment: int
    rel_start: float
    rel_end: float
    visible: float              # share of the beat that is inside the promo
    partial_start: bool         # already under way at the cut-in
    partial_end: bool           # cut off at the cut-out


@dataclass
class ClipContext:
    clip_id: str
    key: str
    refs: list[ClipRef]
    duration: float
    seg_offsets: list[float]

    @property
    def stitched(self) -> bool:
        return len(self.seg_offsets) > 1

    def by_ref(self) -> dict[str, ClipRef]:
        return {r.ref: r for r in self.refs}


def clip_context(p: Promo, ep_by_num: dict[int, Episode], tolerance: float = TIME_TOL) -> ClipContext:
    """Exactly the footage of this promo, in playback order (CAMERA, ACTION, dialogue and SFX). Nothing from
    outside the selected segments: no logline, continuity notes or show profile."""
    refs, offsets, off = [], [], Decimal(0)
    for si, seg in enumerate(p.segments):
        offsets.append(float(off))
        for b in ep_by_num[seg.episode].beats:
            if b.end <= seg.start or b.start >= seg.end:
                continue
            vs, ve = max(b.start, seg.start), min(b.end, seg.end)
            if ve - vs <= 0 or (tolerance and ve - vs < 0.05):
                continue
            refs.append(ClipRef(f"B{len(refs) + 1:02d}", b, si, float(off + Decimal(str(vs)) - Decimal(str(seg.start))),
                                float(off + Decimal(str(ve)) - Decimal(str(seg.start))),
                                (ve - vs) / max(b.end - b.start, 0.05),
                                b.start < seg.start - tolerance, b.end > seg.end + tolerance))
        off += Decimal(str(seg.end)) - Decimal(str(seg.start))
    clip_id = "K" + hashlib.sha1(p.key.encode()).hexdigest()[:6].upper()
    return ClipContext(clip_id, p.key, refs, float(off), offsets)


def render_clip_v7(ctx: ClipContext) -> str:
    lines, seg = [], 0
    for r in ctx.refs:
        if r.segment != seg:
            seg = r.segment
            lines.append(f"=== HARD CUT: segment {seg + 1} starts at +{ctx.seg_offsets[seg]:.1f} s ===")
        b = r.beat
        who = (b.speaker + (f" ({b.emotion})" if b.emotion else "")) if b.kind == "DIALOGUE" else b.kind
        note = ""
        if r.partial_start:
            note += f" [partial: under way before the promo starts; {r.visible:.0%} of it is shown]"
        if r.partial_end:
            note += f" [partial: cut off at the end; {r.visible:.0%} of it is shown]"
        lines.append(f"{r.ref} +{r.rel_start:.1f}-{r.rel_end:.1f}s {who}: {b.text}{note}")
    return "\n".join(lines)


def _text(v, n: int = 300) -> str:
    return re.sub(r"\s+", " ", str(v)).strip()[:n] if isinstance(v, (str, int, float)) and str(v).strip() else ""


def _refs(v, ctx: ClipContext) -> tuple[list[str], list[str]]:
    """Normalise a list of refs ('B3', 'b03', 3) -> (known refs, unknown refs)."""
    if v is None:
        return [], []
    items = v if isinstance(v, list) else [v]
    known, unknown, valid = [], [], ctx.by_ref()
    for x in items:
        m = re.fullmatch(r"\s*[Bb]?0*(\d+)\s*", str(x))
        ref = f"B{int(m.group(1)):02d}" if m else str(x)
        (known if ref in valid else unknown).append(ref)
    return list(dict.fromkeys(known)), unknown


def validate_judgment(data, ctx: ClipContext) -> tuple[dict, list[str]]:
    """Normalise one v7 reply and list its STRUCTURAL problems (which trigger a retry). Whether the cited
    evidence actually supports a verdict is decided later, in eligibility.py."""
    if isinstance(data, dict) and isinstance(data.get("clips"), list) and len(data["clips"]) == 1:
        data = data["clips"][0]
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict):
        return {}, ["the reply is not a JSON object"]
    errors = []
    cid = _text(data.get("clip_id"), 40)
    if cid != ctx.clip_id:
        errors.append(f"clip_id must be {ctx.clip_id} (got {cid or 'nothing'})")
    sections = {}
    for name in ("hook", "ending", "standalone") + (("transition",) if ctx.stitched else ()):
        sec = data.get(name)
        if not isinstance(sec, dict):
            errors.append(f"section '{name}' is missing")
            sections[name] = None
            continue
        verdict = str(sec.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            errors.append(f"{name}.verdict must be pass, fail or uncertain")
            verdict = None
        ev, unknown = _refs(sec.get("evidence"), ctx)
        if unknown:
            errors.append(f"{name}.evidence cites refs that are not in the clip: {', '.join(unknown[:4])}")
        out = {"verdict": verdict, "evidence": ev, **{k: _text(sec.get(k)) for k in SECTION_TEXT[name]}}
        if name == "ending":
            out["answered_within_clip"] = _as_bool(sec.get("answered_within_clip"))
            mt = str(sec.get("final_moment_type", "")).strip().lower()
            out["final_moment_type"] = mt if mt in MOMENT_TYPES else None
            if out["final_moment_type"] is None:
                errors.append("ending.final_moment_type must be suspense, resolution or neutral")
            out["answered_by"], unk = _refs(sec.get("answered_by"), ctx)
            if unk:
                errors.append(f"ending.answered_by cites refs that are not in the clip: {', '.join(unk[:4])}")
        sections[name] = out
    raw = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    scores = {}
    for k in ("hook", "cliffhanger", "standalone"):
        try:
            scores[k] = max(0.0, min(10.0, float(raw.get(k))))
        except (TypeError, ValueError):
            scores[k] = None
            errors.append(f"scores.{k} must be a number from 0 to 10")
    return {"clip_id": ctx.clip_id, "for_clip": ctx.key, "sections": sections, "scores": scores,
            "why_it_works": _text(data.get("why_it_works"), 400), "limitations": _text(data.get("limitations"), 300)}, errors


def _new_judge_stats() -> dict:
    return {"prompt": "judge:v7", "schema_version": JUDGE_SCHEMA_VERSION, "clips": 0,
            "api_calls": 0, "cache_hits": 0, "call_errors": 0,
            "replies_parsed": 0, "replies_valid": 0, "replies_invalid": 0, "validation_retries": 0,
            "clips_valid": 0, "clips_partial": 0, "clips_unjudged": 0,
            "answers": {k: {"pass": 0, "fail": 0, "uncertain": 0, "missing": 0}
                        for k in ("hook", "ending", "standalone", "transition")},
            "final_moment_type": {k: 0 for k in (*MOMENT_TYPES, "missing")}}


def merge_judge_stats(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict):
            out[k] = merge_judge_stats(a.get(k, {}), v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(a.get(k, 0), (int, float)):
            out[k] = a.get(k, 0) + v
        else:
            out[k] = v
    return out


def judge_v7(llm: LLM, promos: list[Promo], ep_by_num: dict[int, Episode], hook_window: float = 3.0,
             ending_window: float = 5.0, retries: int = 2, log=print, workers: int = 1, tick=None) -> dict:
    """One call per clip. Each reply is validated (clip id, required sections, verdicts, refs, scores); an
    invalid reply is retried up to `retries` times with the problems listed. What still fails validation
    stays missing, and eligibility.py turns missing into 'uncertain', never into a pass."""
    before = dict(llm.stats)

    def one(p: Promo):
        ctx = clip_context(p, ep_by_num)
        base = JUDGE_TEMPLATE_V7.format(
            clip_id=ctx.clip_id, duration=ctx.duration, n_seg=len(p.segments), schema=JUDGE_SCHEMA_VERSION,
            clip=render_clip_v7(ctx), hook_window=hook_window, final_from=max(0.0, ctx.duration - ending_window),
            transition_rule=TRANSITION_RULE_V7 if ctx.stitched else "", shape=_judge_shape(ctx.stitched))
        j, errors, attempts = {}, [], []
        for a in range(1 + max(0, retries)):
            user = base if a == 0 else base + RETRY_NOTE_V7.format(errors="; ".join(errors[:6]), clip_id=ctx.clip_id)
            try:
                data = llm.json(JUDGE_SYSTEM_V7, user, tag=f"judge:{ctx.clip_id}" + (f":retry{a}" if a else ""),
                                schema=judge_schema(ctx.stitched))
            except _SOFT_FAIL as e:
                attempts.append("error")
                return p, ctx, j, errors or [f"call failed: {e}"], attempts
            j, errors = validate_judgment(data, ctx)
            attempts.append("valid" if not errors else "invalid")
            if not errors:
                break
        return p, ctx, j, errors, attempts

    stats = _new_judge_stats()
    stats["clips"] = len(promos)
    for p, ctx, j, errors, attempts in run_parallel(promos, one, workers, tick, llm.cancel):
        stats["call_errors"] += attempts.count("error")
        stats["replies_valid"] += attempts.count("valid")
        stats["replies_invalid"] += attempts.count("invalid")
        stats["replies_parsed"] += attempts.count("valid") + attempts.count("invalid")
        stats["validation_retries"] += max(0, len(attempts) - 1)
        usable = bool(j) and not any(e.startswith("clip_id") for e in errors)
        if not usable:
            stats["clips_unjudged"] += 1
            p.judgment = {"for_clip": ctx.key, "clip_id": ctx.clip_id, "sections": {}, "scores": {},
                          "unavailable": True, "errors": errors or ["no reply"], "attempts": len(attempts)}
            p.llm = {}
            continue
        stats["clips_valid" if not errors else "clips_partial"] += 1
        j["errors"], j["attempts"] = errors, len(attempts)
        p.judgment = j
        for name, sec in j["sections"].items():
            v = (sec or {}).get("verdict")
            stats["answers"][name]["missing" if v is None else v] += 1
        stats["final_moment_type"][(j["sections"].get("ending") or {}).get("final_moment_type") or "missing"] += 1
        sc = j["scores"]
        p.llm = {}
        if all(sc.get(k) is not None for k in ("hook", "cliffhanger", "standalone")):
            secs = j["sections"]
            p.llm = {**{k: sc[k] for k in ("hook", "cliffhanger", "standalone")},
                     "hook_moment": (secs.get("hook") or {}).get("description", ""),
                     "open_question": (secs.get("ending") or {}).get("open_question", ""),
                     "confusing": (secs.get("standalone") or {}).get("blocking_confusion", ""),
                     "why": j.get("why_it_works", "")}
    stats["api_calls"] = llm.stats["calls"] - before["calls"]
    stats["cache_hits"] = llm.stats["cache_hits"] - before["cache_hits"]
    log(f"  [judge] {stats['clips_valid']} valid + {stats['clips_partial']} partial + {stats['clips_unjudged']} "
        f"unjudged of {len(promos)} clips · {stats['api_calls']} new calls, {stats['cache_hits']} cached, "
        f"{stats['validation_retries']} validation retries")
    return stats
