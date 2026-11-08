"""One authoritative eligibility decision per candidate (v7).

    deterministic checks (code)       duration, source bounds, segment order, dialogue boundaries
    semantic checks (judge + code)    hook, ending, standalone, and transition for two-segment promos
    decision                          any fail -> rejected · any uncertain -> review · all pass -> recommended

Every check is {status: pass|fail|uncertain, reason, evidence: [beats with source and clip times], source}.
A judge verdict only counts when its cited evidence holds up in code: the hook beat must start in the first
3 s, the ending beat must play in the final seconds, a transition needs a beat from each side of the cut.
A missing, malformed or unsupported answer is 'uncertain', never a pass. Scores rank eligible clips; they
never decide eligibility.
"""
from __future__ import annotations

import re

from .candidates import TIME_TOL, Promo, hook_strength
from .config import Settings
from .llm_stages import ClipContext, ClipRef, clip_context
from .parse import Beat, Episode, fmt_ts

PASS, FAIL, UNCERTAIN = "pass", "fail", "uncertain"
CHECK_LABELS = {
    "duration": "20-60 s", "source_bounds": "inside source", "segment_order": "segment order",
    "dialogue_boundaries": "no mid-line cut", "hook": "hook in first 3 s", "ending": "ends unresolved",
    "standalone": "new viewer follows", "transition": "cut makes sense",
}
DETERMINISTIC = ("duration", "source_bounds", "segment_order", "dialogue_boundaries")


def check(status: str, reason: str, evidence: list | None = None, source: str = "code") -> dict:
    return {"status": status, "reason": reason, "evidence": evidence or [], "source": source}


def _short(t: str, n: int = 90) -> str:
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def evidence_row(r: ClipRef) -> dict:
    b = r.beat
    return {"ref": r.ref, "beat": b.id, "episode": b.episode, "time": f"{fmt_ts(b.start)}-{fmt_ts(b.end)}",
            "clip_time": f"+{r.rel_start:.1f}-{r.rel_end:.1f}s", "kind": b.kind,
            "text": (f"{b.speaker}: " if b.kind == "DIALOGUE" else f"{b.kind}: ") + _short(b.text)}


# ----------------------------------------------------------------------------- deterministic checks
def deterministic_checks(p: Promo, ep_by_num: dict[int, Episode], S: Settings) -> dict:
    c, tol = {}, S.len_tolerance
    d = p.duration
    c["duration"] = check(PASS if S.min_len - tol <= d <= S.max_len + tol else FAIL,
                          f"{d:.2f} s (allowed {S.min_len:.0f}-{S.max_len:.0f} s, ±{tol} s rounding)")
    bad = [s.key() for s in p.segments if s.episode not in ep_by_num or s.start < -tol
           or s.end > ep_by_num[s.episode].duration + tol]           # tol: '01:58.9' displayed for 01:58.88
    c["source_bounds"] = check(FAIL if bad else PASS,
                               f"outside the episode: {', '.join(bad)}" if bad else "every segment lies inside its episode")
    order = [f"{s.key()} ends before it starts" for s in p.segments if s.end - s.start <= 0]
    for a, b in zip(p.segments, p.segments[1:]):
        if (b.episode, b.start) < (a.episode, a.end - TIME_TOL):
            order.append(f"{b.key()} plays before or inside {a.key()} in story order")
    c["segment_order"] = check(FAIL if order else PASS, "; ".join(order) if order else
                               "segments run forward in story time and do not overlap")
    cuts = []
    for s in p.segments:
        ep = ep_by_num.get(s.episode)
        for t, side in ((s.start, "in"), (s.end, "out")):
            line = next((b for b in (ep.beats if ep else []) if b.kind == "DIALOGUE"
                         and b.start < t - TIME_TOL and b.end > t + TIME_TOL), None)
            if line:
                cuts.append(f"cut-{side} at Ep{s.episode:02d} {fmt_ts(t)} falls inside the line {line.id} "
                            f"({fmt_ts(line.start)}-{fmt_ts(line.end)})")
    c["dialogue_boundaries"] = check(FAIL if cuts else PASS, "; ".join(cuts) if cuts else
                                     "every cut falls between spoken lines")
    return c


# ----------------------------------------------------------------------------- rule opinions (secondary)
# Rules never pass a semantic check on their own. They can only turn a judge verdict they contradict into a
# human review: a clear visual hook the judge failed, or a visibly settled ending the judge passed.
def _opening_refs(ctx: ClipContext, S: Settings) -> list[ClipRef]:
    return [r for r in ctx.refs if r.segment == 0 and r.rel_start < S.hook_window
            and (not r.partial_start or r.visible >= 0.5)]


def _final_refs(ctx: ClipContext, S: Settings) -> list[ClipRef]:
    return [r for r in ctx.refs if r.rel_end > ctx.duration - S.ending_window + 0.05]


def rule_hook(ctx: ClipContext, sig: dict, S: Settings) -> tuple[str, ClipRef | None]:
    best = max(_opening_refs(ctx, S), key=lambda r: hook_strength(sig, r.beat.id), default=None)
    if best is not None and hook_strength(sig, best.beat.id) >= 0.8:
        return "strong_pass", best
    return "neutral", best


def rule_ending(ctx: ClipContext, sig: dict, S: Settings) -> tuple[str, ClipRef | None]:
    """strong_fail only when the last story beat is a visible resolution (an embrace, relief, 'safe') and
    nothing in the final window carries tension. A calm or tender tone alone is not used."""
    story = [r for r in _final_refs(ctx, S) if not sig[r.beat.id].get("music_cue")]
    if not story:
        return "neutral", None
    last = max(story, key=lambda r: r.rel_end)
    tension = max(sig[r.beat.id].get("cliff_rule", sig[r.beat.id]["cliff"]) for r in story)
    if sig[last.beat.id]["resolve"] >= 0.8 and tension < 0.7:
        return "strong_fail", last
    return "neutral", last


_NAME_STOP = {"the", "with", "and", "his", "her", "young", "old", "man", "woman", "voice", "off", "screen", "narrator",
              # generic honorifics and titles: 'LADY X' and 'X' are the same person
              "lady", "lord", "sir", "mrs", "miss", "doctor", "father", "mother", "king", "queen", "prince",
              "princess", "duke", "duchess", "count", "countess", "officer", "detective", "boss"}


def _speaker_tokens(speaker: str) -> set[str]:
    """'LADY JULIA' -> {'julia'}; a speaker known only by a title ('DUCHESS') keeps the title."""
    words = set(re.findall(r"[a-z]{3,}", speaker.lower()))
    return (words - _NAME_STOP) or words


def _names(ctx: ClipContext, seg: int, vocab: set[str]) -> set[str]:
    out = set()
    for r in ctx.refs:
        if r.segment != seg:
            continue
        b = r.beat
        if b.kind == "DIALOGUE":
            out |= _speaker_tokens(b.speaker)
        out |= {w for w in re.findall(r"[a-z]{3,}", b.text.lower()) if w in vocab}
    return out


def rule_transition(ctx: ClipContext, S: Settings) -> tuple[str, str]:
    """strong_fail when the two sides of the cut share no named character at all. Sharing one is NOT a pass."""
    vocab = {w for r in ctx.refs if r.beat.kind == "DIALOGUE" for w in _speaker_tokens(r.beat.speaker)}
    a, b = _names(ctx, 0, vocab), _names(ctx, 1, vocab)
    return ("strong_fail", "the two segments share no named character") if a and b and not (a & b) else ("neutral", "")


# ----------------------------------------------------------------------------- semantic checks
def _ev(ctx: ClipContext, refs: list[str]) -> list[ClipRef]:
    m = ctx.by_ref()
    return [m[x] for x in refs if x in m]


def _verdict_evidence(name: str, sec: dict, ctx: ClipContext, sig: dict, S: Settings) -> tuple[str, str, list[ClipRef]]:
    """Check the judge's cited evidence against the clip. Returns (verdict to use, reason, evidence)."""
    v = sec["verdict"]
    ev = _ev(ctx, sec.get("evidence", []))
    if name == "hook":
        good = [r for r in ev if r in _opening_refs(ctx, S)]
        if v == PASS and not good:
            cited = ", ".join(f"{r.ref} at +{r.rel_start:.1f} s" for r in ev) or "no beat"
            return UNCERTAIN, f"the judge passed the hook but cited {cited}, not a beat that starts in the first {S.hook_window:.0f} s", ev
        return v, sec.get("description") or sec.get("mechanism") or "", good if v == PASS else ev
    if name == "ending":
        answered = _ev(ctx, sec.get("answered_by", []))
        mtype = sec.get("final_moment_type")
        where = ", ".join(f"{r.ref} at +{r.rel_start:.1f} s" for r in answered)
        if sec.get("answered_within_clip") is True:
            if answered:
                q = f"its own question {sec['open_question']!r}" if sec.get("open_question") else "its own question"
                return FAIL, f"the clip answers {q} itself ({where})", answered
            return UNCERTAIN, "the judge says the clip answers its own question but cites no beat that does", ev
        if mtype == "resolution":
            moment = sec.get("final_moment") or sec.get("reason") or "the final beat"
            if answered:
                return FAIL, f"the final moment releases the clip's tension ({moment}; released at {where})", answered
            if v == PASS:
                return UNCERTAIN, "the judge passed the ending but classed the final moment as a resolution", ev
            return FAIL, f"the final moment releases the clip's tension: {moment}", ev
        if mtype == "neutral":
            if v == PASS:
                return UNCERTAIN, "the judge passed the ending but found nothing pending in the final moment", ev
            return FAIL, f"nothing is pending at the cut: {sec.get('final_moment') or sec.get('reason') or 'final beat'}", ev
        if v == PASS:
            final = _final_refs(ctx, S)
            good = [r for r in ev if r in final and not sig[r.beat.id].get("music_cue")]
            if mtype is None:
                return UNCERTAIN, "the judge passed the ending without classifying the final moment", ev
            if not sec.get("open_question"):
                return UNCERTAIN, "the judge passed the ending without naming the question it leaves open", ev
            if _PRESUPPOSES.search(sec["open_question"]):
                return UNCERTAIN, (f"the open question presupposes something the clip already settled "
                                   f"({sec['open_question']!r})"), ev
            if not good:
                if any(r in final for r in ev):
                    return UNCERTAIN, "the only cited ending evidence is a music or sound cue", ev
                cited = ", ".join(f"{r.ref} at +{r.rel_start:.1f} s" for r in ev) or "no beat"
                return UNCERTAIN, (f"the judge's open question rests on {cited}, not on the final "
                                   f"{S.ending_window:.0f} s of the clip"), ev
            return PASS, f"{sec.get('final_moment') or 'final beat'} leaves open: {sec['open_question']}", good
        if v == FAIL and not (sec.get("reason") or sec.get("final_moment")):
            return UNCERTAIN, "the judge failed the ending without a reason", ev
        return v, sec.get("reason") or sec.get("final_moment") or "", ev
    if name == "standalone":
        if v == PASS and not (sec.get("conflict") and ev):
            return UNCERTAIN, "the judge passed standalone without citing the beats that set up the conflict", ev
        if v == FAIL and not sec.get("blocking_confusion"):
            return UNCERTAIN, "the judge failed standalone without saying what blocks understanding", ev
        return v, (sec.get("conflict") if v == PASS else sec.get("blocking_confusion")) or "", ev
    # transition
    if v == PASS:
        sides = {r.segment for r in ev}
        if not sec.get("connection") or sides != set(range(len(ctx.seg_offsets))):
            return UNCERTAIN, "the judge passed the cut without a stated connection backed by a beat from each side", ev
        return PASS, sec["connection"], ev
    if v == FAIL and not (sec.get("reason") or sec.get("connection")):
        return UNCERTAIN, "the judge failed the cut without a reason", ev
    return v, sec.get("reason") or "", ev


_SCORE_KEY = {"hook": "hook", "ending": "cliffhanger", "standalone": "standalone"}
# "What will she do now that she has found it?" asks about the aftermath of something the clip resolved
_PRESUPPOSES = re.compile(r"\bnow that\b", re.I)
_CODE_REASONS = ("the clip answers", "the final moment", "nothing is pending")
_FAIL_PREFIX = {"hook": "the first 3 s don't grab attention", "ending": "the ending doesn't leave a question open",
                "standalone": "a new viewer can't follow it", "transition": "the cut between segments doesn't hold together"}


def semantic_checks(p: Promo, ctx: ClipContext, sig: dict, S: Settings, unavailable: str = "") -> dict:
    names = ["hook", "ending", "standalone"] + (["transition"] if ctx.stitched else [])
    j = p.judgment if p.judgment.get("for_clip") == p.key else {}
    if p.judgment and not j:
        unavailable = "the stored judgment was made for different footage, so it is not reused"
    if not j or j.get("unavailable"):
        why = unavailable or "; ".join(j.get("errors", [])[:2]) or "no model judgment"
        return {n: check(UNCERTAIN, f"not verified: {why}", source="none") for n in names}
    out = {}
    secs, scores = j.get("sections", {}), j.get("scores", {})
    for n in names:
        sec = secs.get(n)
        if not sec or sec.get("verdict") is None:
            out[n] = check(UNCERTAIN, "the judge's answer was missing or malformed after retries", source="judge")
            continue
        v, reason, ev = _verdict_evidence(n, sec, ctx, sig, S)
        rows = [evidence_row(r) for r in ev]
        if v == UNCERTAIN:
            if sec["verdict"] == UNCERTAIN:          # the judge itself was unsure (not an evidence problem)
                reason = "the judge could not decide from the clip" + (f": {reason}" if reason else "")
            out[n] = check(UNCERTAIN, reason or "the judge could not decide from the clip", rows, "judge")
            continue
        if v == FAIL and n in _FAIL_PREFIX and not reason.startswith(_CODE_REASONS):
            reason = _FAIL_PREFIX[n] + (f": {reason}" if reason else "")
        score = scores.get(_SCORE_KEY.get(n, ""))
        if score is not None and ((v == PASS and score <= 4) or (v == FAIL and score >= 8)):
            out[n] = check(UNCERTAIN, f"the judge's {v} verdict contradicts its own {score:.0f}/10 rating", rows, "judge")
            continue
        source = "judge"
        if n == "hook":
            op, r = rule_hook(ctx, sig, S)
            if v == FAIL and op == "strong_pass":
                out[n] = check(UNCERTAIN, f"the judge failed the hook, but {r.ref} ({r.beat.kind}, +{r.rel_start:.1f} s) "
                                          f"is a strong opening beat: {_short(r.beat.text, 70)}",
                               rows + [evidence_row(r)], "judge vs rules")
                continue
            if v == PASS and op == "strong_pass":
                source = "judge + rules"
        elif n == "ending":
            op, r = rule_ending(ctx, sig, S)
            if v == PASS and op == "strong_fail":
                out[n] = check(UNCERTAIN, f"the judge passed the ending, but the last beat {r.ref} shows a resolution: "
                                          f"{_short(r.beat.text, 70)}", rows + [evidence_row(r)], "judge vs rules")
                continue
        elif n == "transition":
            op, why = rule_transition(ctx, S)
            if v == PASS and op == "strong_fail":
                out[n] = check(UNCERTAIN, f"the judge passed the cut, but {why}", rows, "judge vs rules")
                continue
        out[n] = check(v, reason, rows, source)
    return out


# ----------------------------------------------------------------------------- decision + explanations
_UNSUPPORTED = re.compile(
    r"\balways\b|\bnever fails\b|\bguarantee\w*|\balgorithm\w*|\bfyp\b|for ?you ?page|\bviral\w*|\btik ?tok\b|"
    r"\binstagram\b|\byoutube\b|\bengagement\b|\bbait\b|\bcheap\b|\bclich[eé]\w*|\bcringe\w*|\btrash\w*|"
    r"guilty[- ]pleasure|\bvibes?\b|\bdoomscroll\w*|\bthumb\b|\blol\b|\bgonna\b|\bwanna\b|\bdude\b|\bsoap opera\b",
    re.I)


def professional(text: str) -> str:
    """'' if the text makes platform/performance claims or uses dismissive language (caller falls back)."""
    return "" if not text or _UNSUPPORTED.search(text) else text


def _beat_line(b: Beat) -> str:
    t = _short(b.text, 140)
    return f'{b.speaker}: "{t}"' if b.kind == "DIALOGUE" else t


def explain(p: Promo, ctx: ClipContext, checks: dict, sig: dict, S: Settings) -> dict:
    """Editorial text built from the validated judgment, falling back to the source lines. Returns counters."""
    secs = (p.judgment.get("sections") or {}) if p.judgment.get("for_clip") == p.key else {}
    hook, end = secs.get("hook") or {}, secs.get("ending") or {}
    opening = _opening_refs(ctx, S) or ctx.refs[:1]
    best_open = max(opening, key=lambda r: hook_strength(sig, r.beat.id)) if opening else None
    story_end = [r for r in ctx.refs if not sig[r.beat.id].get("music_cue")] or ctx.refs
    last = max(story_end, key=lambda r: r.rel_end) if story_end else None
    p.hook_text = (hook.get("description") if checks["hook"]["status"] == PASS and hook.get("description")
                   else (_beat_line(best_open.beat) if best_open else ""))
    ending_ok = checks["ending"]["status"] == PASS
    p.final_moment = end.get("final_moment") if ending_ok and end.get("final_moment") else (
        _beat_line(last.beat) if last else "")
    p.cliff_text = end.get("open_question", "") if ending_ok else ""
    replaced = 0
    why = professional(p.judgment.get("why_it_works", "")) if secs else ""
    if secs and p.judgment.get("why_it_works") and not why:
        replaced += 1
    if not why:
        parts = []
        if checks["hook"]["status"] == PASS:
            mech = hook.get("mechanism") or "a strong opening"
            parts.append(f"Opens on {mech.lower()}: {p.hook_text.rstrip('.')}.")
        if ending_ok:
            parts.append(f"Cuts on {p.final_moment.rstrip('.')[:1].lower() + p.final_moment.rstrip('.')[1:]}, "
                         f"leaving open: {p.cliff_text}")
        why = " ".join(parts) or "Not verified as a promo; see the review and failure reasons."
    p.why = why
    lim = professional(p.judgment.get("limitations", "")) if secs else ""
    if secs and p.judgment.get("limitations") and not lim:
        replaced += 1
    p.limitations = lim
    return {"explanations_replaced": replaced}


def decide(p: Promo, ep_by_num: dict[int, Episode], sig: dict, S: Settings, unavailable: str = "") -> dict:
    """Set p.decision, p.checks and the editorial text. The ONLY place a candidate's status is decided."""
    ctx = clip_context(p, ep_by_num)
    checks = {**deterministic_checks(p, ep_by_num, S), **semantic_checks(p, ctx, sig, S, unavailable)}
    fails = [k for k, c in checks.items() if c["status"] == FAIL]
    unsure = [k for k, c in checks.items() if c["status"] == UNCERTAIN]
    status = "rejected" if fails else "review" if unsure else "recommended"
    notes = []
    if p.kind == "stitched":
        notes.append("visual continuity (costume, location, lighting) across the cut can't be checked from the "
                     "script: verify it in the edit")
    secs = (p.judgment.get("sections") or {}) if p.judgment.get("for_clip") == p.key else {}
    if (secs.get("standalone") or {}).get("harmless_curiosity"):
        notes.append("open questions that add intrigue (not blocking): " + secs["standalone"]["harmless_curiosity"])
    for k, c in checks.items():
        if c["status"] == PASS and c["source"] == "judge" and k == "hook":
            op, r = rule_hook(ctx, sig, S)
            if op != "strong_pass" and r is not None and hook_strength(sig, r.beat.id) < 0.45:
                notes.append("the keyword rules rated this opening weak; it passed on the judge's cited evidence")
    if p.origin:
        notes.append(f"re-cut of {p.origin['derived_from']}: {p.origin['change']}")
    counters = explain(p, ctx, checks, sig, S)
    p.checks = checks
    p.decision = {
        "status": status,
        "failure_reasons": [f"{CHECK_LABELS.get(k, k)}: {checks[k]['reason']}" for k in fails],
        "review_reasons": [f"{CHECK_LABELS.get(k, k)}: {checks[k]['reason']}" for k in unsure],
        "notes": notes, **counters,
    }
    return p.decision


def ending_repair(p: Promo, ep_by_num: dict[int, Episode], sig: dict, S: Settings) -> tuple[float, str] | None:
    """Where to re-cut a clip whose ending failed: just before the beat that answers its own question, or
    (rules) just after the strongest earlier tension beat. Returns (absolute time in the last segment's
    episode, why) or None."""
    if p.decision.get("status") == "recommended" or p.checks.get("ending", {}).get("status") == PASS:
        return None
    ctx = clip_context(p, ep_by_num)
    last_seg = len(p.segments) - 1
    sec = ((p.judgment.get("sections") or {}).get("ending") or {}) if p.judgment.get("for_clip") == p.key else {}
    answered = [r for r in _ev(ctx, sec.get("answered_by", [])) if r.segment == last_seg]
    resolved = sec.get("final_moment_type") == "resolution"
    if (sec.get("answered_within_clip") is True or resolved) and answered:
        first = min(answered, key=lambda r: r.beat.start)
        return first.beat.start, f"end before {first.beat.id}, where the clip resolves its own tension"
    # no cited answer: if the ending failed, or the judge classed the final moment as a resolution, cut after
    # the strongest earlier tension beat (unblended rule score or miner tag, like hook_strength)
    if p.checks.get("ending", {}).get("status") != FAIL and not resolved:
        return None
    seg = p.segments[-1]
    strength = lambda b: max(sig[b.id].get("cliff_rule", sig[b.id]["cliff"]), sig[b.id].get("cliff_llm", 0.0))
    window = [b for b in ep_by_num[seg.episode].beats_between(seg.start, seg.end)
              if seg.end - 15 <= b.end <= seg.end - 2 and not sig[b.id].get("music_cue")]
    tense = [b for b in window if strength(b) >= 0.7 and sig[b.id]["resolve"] < 0.8]
    if not tense:
        return None
    b = max(tense, key=lambda b: (strength(b), b.end))
    return b.end, f"end on the earlier tension beat {b.id} instead of a settled ending"
