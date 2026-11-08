"""Final scoring, ranking, and automatic quality checks (all plain code, all explainable)."""
from __future__ import annotations

import math

from .candidates import Promo, _overlap, nms
from .config import Settings
from .parse import Beat, Episode, fmt_ts

V4 = Settings()
DEFAULT_WEIGHTS = V4.weights()


def funnel_fit(first_ep: int, episode_numbers: list[int], strength: float = 0.7) -> float:
    """Early-episode bonus: 1.0 for the first episode loaded, falling to 1 - strength for the last.
    Assumption: viewers land on Ep1, so early promos pay off sooner and spoil less.
    Uses the episode's position within the loaded range (bug O1: the v4 formula assumed episodes
    start at 1, and went negative for a subset like episodes 26-50). Identical to v4 for 1..N."""
    lo, hi = min(episode_numbers), max(episode_numbers)
    if hi <= lo:
        return 1.0
    return 1.0 - strength * (first_ep - lo) / (hi - lo)


def _judge_cliff(llm: dict, mode: str) -> float:
    score = llm["cliffhanger"] / 10
    if "check_score" not in llm or mode == "score":
        return score
    return llm["check_score"] if mode == "checks" else 0.5 * score + 0.5 * llm["check_score"]


def finalize(promos: list[Promo], sig: dict, beats: dict[str, Beat], episode_numbers: list[int],
             S: Settings = V4) -> None:
    for p in promos:
        r = p.rule
        hook, cliff, stand = r["hook"], r["cliff"], r["standalone"]
        if p.llm:
            wj = S.w_judge
            hook = (1 - wj) * hook + wj * p.llm["hook"] / 10
            cliff = (1 - wj) * cliff + wj * _judge_cliff(p.llm, S.cliff_mode)
            stand = (1 - wj) * stand + wj * p.llm["standalone"] / 10
            # v4-v6: rules and model disagreeing a lot = low confidence -> send to a human. In v7 these
            # flags are replaced by the per-check decision (eligibility.py), which says WHY.
            if S.eligibility != "verified":
                if abs(p.llm["hook"] / 10 - sig[p.hook_beat].get("hook_rule", r["hook"])) > 0.5:
                    p.flags.append("rules-vs-model disagree on hook")
                if abs(p.llm["cliffhanger"] / 10 - sig[p.cliff_beat].get("cliff_rule", r["cliff"])) > 0.5:
                    p.flags.append("rules-vs-model disagree on cliffhanger")
                if p.llm.get("confusing"):
                    p.flags.append("cold viewer confused: " + p.llm["confusing"][:90])
        funnel = funnel_fit(p.first_episode, episode_numbers, S.funnel_strength)
        p.rule["funnel"] = funnel
        if S.standalone_mode == "geometric":
            # weighted geometric mean of hook, cliffhanger AND standalone: weak on any one -> low overall
            ws = S.w_hook + S.w_cliff + S.w_standalone
            eps = 1e-3
            geo = math.exp((S.w_hook * math.log(max(hook, eps)) + S.w_cliff * math.log(max(cliff, eps))
                            + S.w_standalone * math.log(max(stand, eps))) / ws)
            p.final = ws * geo + S.w_intensity * r["intensity"] + S.w_funnel * funnel
        else:
            # v4: a promo needs a hook AND a cliffhanger -> geometric mean of the two (a great ending
            # can't rescue a weak opening); standalone is added on top
            both = (max(hook, 0.0) * max(cliff, 0.0)) ** 0.5
            p.final = ((S.w_hook + S.w_cliff) * both + S.w_standalone * stand
                       + S.w_intensity * r["intensity"] + S.w_funnel * funnel)
        p.rule["blended"] = {"hook": round(hook, 3), "cliff": round(cliff, 3), "standalone": round(stand, 3)}
        p.rule["formula"] = score_formula(S, hook, cliff, stand, r["intensity"], funnel, p.final, bool(p.llm))

        hb, cb = beats[p.hook_beat], beats[p.cliff_beat]
        p.hook_text = p.llm.get("hook_moment") or _beat_line(hb)
        p.cliff_text = p.llm.get("open_question") or _beat_line(cb)
        p.why = p.llm.get("why") or _template_why(p, sig)


def score_formula(S: Settings, hook, cliff, stand, intensity, funnel, final, judged: bool) -> str:
    """The ranking score written out with its actual numbers, so the order can be checked by hand."""
    src = (f"each of hook/cliff/standalone = {1 - S.w_judge:.1f} x rules + {S.w_judge:.1f} x judge score/10"
           if judged else "rule scores only (not judged)")
    if S.standalone_mode == "geometric":
        core = (f"{S.w_hook + S.w_cliff + S.w_standalone:.2f} x weighted-geomean(hook {hook:.2f}, cliff {cliff:.2f}, "
                f"standalone {stand:.2f})")
    else:
        core = (f"{S.w_hook + S.w_cliff:.2f} x sqrt(hook {hook:.2f} x cliff {cliff:.2f}) + "
                f"{S.w_standalone:.2f} x standalone {stand:.2f}")
    return (f"{core} + {S.w_intensity:.2f} x intensity {intensity:.2f} + {S.w_funnel:.2f} x early-episode "
            f"{funnel:.2f} = {final:.3f}  ({src})")


def _beat_line(b: Beat) -> str:
    t = b.text if len(b.text) < 140 else b.text[:137] + "..."
    return f'{b.speaker}: "{t}"' if b.kind == "DIALOGUE" else t


def _template_why(p: Promo, sig: dict) -> str:
    hr = ", ".join(dict.fromkeys(sig[p.hook_beat]["hook_reasons"]))[:60] or "tension"
    cr = ", ".join(dict.fromkeys(sig[p.cliff_beat]["cliff_reasons"]))[:60] or "unresolved beat"
    shape = "two-moment trailer" if p.kind == "stitched" else "single continuous scene"
    return f"{shape}: opens on {hr}; cuts on {cr}."


def quality_checks(p: Promo, ep_by_num: dict[int, Episode], sig: dict, S: Settings = V4) -> dict:
    """Automatic pass/fail checks an editor would otherwise eyeball."""
    c = {}
    c["length_20_60s"] = S.min_len <= p.duration <= S.max_len
    clean = True
    for s in p.segments:
        ep = ep_by_num[s.episode]
        for t in (s.start, s.end):
            if any(b.kind == "DIALOGUE" and b.start < t - 0.15 and b.end > t + 0.15 for b in ep.beats):
                clean = False
        if s.start < 0 or s.end > ep.duration + 0.01:
            clean = False
    c["clean_cuts_no_mid_line"] = clean
    first = p.segments[0]
    opening = [b for b in ep_by_num[first.episode].beats_between(first.start, first.end)
               if first.start - S.hook_lead <= b.start < first.start + S.hook_window]
    c["hook_in_first_3s"] = any(sig[b.id]["hook"] >= 0.45 for b in opening)
    last = p.segments[-1]
    closing = [b for b in ep_by_num[last.episode].beats_between(last.start, last.end) if b.end > last.end - S.cliff_window]
    c["ends_unresolved"] = max(sig[b.id]["cliff"] for b in closing) > max(sig[b.id]["resolve"] for b in closing) * 0.6
    c["has_dialogue"] = any(b.kind == "DIALOGUE" for s in p.segments
                            for b in ep_by_num[s.episode].beats_between(s.start, s.end))
    if p.llm:
        c["cold_viewer_understands"] = p.llm["standalone"] >= 5
    return c


def _needs_context(p: Promo, S: Settings) -> bool:
    stand = p.rule.get("blended", {}).get("standalone", p.rule["standalone"])
    if S.standalone_mode == "gate" and stand < S.standalone_threshold:
        return True
    return p.kind == "stitched" and S.stitched_min_standalone > 0 and stand < S.stitched_min_standalone


def rank(promos: list[Promo], ep_by_num, sig, S: Settings = V4) -> tuple[list[Promo], list[Promo]]:
    """Returns (main ranking, needs-context list). The second list is empty unless gating is on."""
    for p in promos:
        p.checks = quality_checks(p, ep_by_num, sig, S)
    hard_ok = [p for p in promos if p.checks["length_20_60s"] and p.checks["clean_cuts_no_mid_line"]]
    gated = [p for p in hard_ok if _needs_context(p, S)]
    main = [p for p in hard_ok if p not in gated]
    ranked = _pick(main, S.top_k, S)
    needs = _pick(gated, 15, S)
    for i, p in enumerate(ranked, 1):
        p.id = f"P{i:02d}"
    for i, p in enumerate(needs, 1):
        p.id = f"C{i:02d}"
        p.flags.append("needs context: a stranger may not follow it on its own")
    return ranked, needs


def _pick(pool: list[Promo], limit: int, S: Settings) -> list[Promo]:
    """Best-first with diversity: no near-duplicates, at most `max_per_episode` per episode, a cap on
    stitched promos. `limit` is a maximum, not a quota."""
    out, per_ep, stitched = [], {}, 0
    for p in nms(pool, key=lambda p: p.final, max_overlap=S.final_overlap):
        eps = {s.episode for s in p.segments}
        if any(per_ep.get(e, 0) >= S.max_per_episode for e in eps):
            continue
        if p.kind == "stitched" and S.max_stitched_in_top and stitched >= S.max_stitched_in_top:
            continue
        for e in eps:
            per_ep[e] = per_ep.get(e, 0) + 1
        stitched += p.kind == "stitched"
        out.append(p)
        if len(out) >= limit:
            break
    return out


def rank_verified(pool: list[Promo], S: Settings) -> tuple[list[Promo], list[Promo], list[Promo], dict]:
    """v7: rank ONLY candidates whose decision is 'recommended'. Review and rejected clips are listed with
    their reasons and are never promoted by score. Returns (recommended, review, rejected, selection stats)."""
    by = {st: [p for p in pool if p.decision.get("status") == st] for st in ("recommended", "review", "rejected")}
    rec = _pick(by["recommended"], S.top_k, S)
    rev_pool = [p for p in by["review"] if all(_overlap(p, k) <= S.final_overlap for k in rec)]
    # clips the final reviewer actually judged (with a specific concern) come before never-reviewed ones
    judged = lambda p: bool(p.judgment) and not p.judgment.get('unavailable')
    rev = nms(rev_pool, key=lambda p: (judged(p), p.final), max_overlap=S.final_overlap, limit=S.top_k)
    rej = sorted(by["rejected"], key=lambda p: -p.final)
    for prefix, items in (("P", rec), ("R", rev), ("X", rej)):
        for i, p in enumerate(items, 1):
            p.id = f"{prefix}{i:02d}"
    shown = {p.key: p.id for p in rec + rev}
    for p in rej + rev:                       # point a failed clip at the re-cut that replaced it
        kids = [shown[q.key] for q in rec + rev if q.origin.get("derived_from") == p.key and q.key in shown]
        if kids:
            p.decision["notes"].append(f"re-cut version listed as {', '.join(kids)}")
    stats = {"eligible": len(by["recommended"]), "eligible_returned": len(rec),
             "eligible_not_returned": len(by["recommended"]) - len(rec),
             "review": len(by["review"]), "review_returned": len(rev), "rejected": len(rej),
             "top_k_max": S.top_k}
    return rec, rev, rej, stats


def apply_rerank(ranked: list[Promo], positions: dict[int, float], top: int) -> list[Promo]:
    """Reorder the first `top` promos by average listwise position (ties: previous score)."""
    head, tail = ranked[:top], ranked[top:]
    if not positions:
        return ranked
    order = sorted(range(len(head)), key=lambda i: (positions.get(i, float(i)), -head[i].final))
    head = [head[i] for i in order]
    for i, p in enumerate(head):
        p.rule["rerank_position"] = round(positions.get(order[i], float(order[i])), 2)
    out = head + tail
    for i, p in enumerate(out, 1):
        p.id = f"P{i:02d}"
    return out


def segment_text(p: Promo, ep_by_num: dict[int, Episode]) -> list[dict]:
    """Exported segments: source precision (MM:SS.ss), seconds kept to the millisecond."""
    rows = []
    for s in p.segments:
        rows.append({"episode": s.episode, "start": fmt_ts(s.start), "end": fmt_ts(s.end),
                     "start_sec": s.start, "end_sec": s.end,
                     "duration_sec": s.dur,
                     "script": [b.render(precise=True) for b in ep_by_num[s.episode].beats if b.end > s.start and b.start < s.end]})
    return rows
