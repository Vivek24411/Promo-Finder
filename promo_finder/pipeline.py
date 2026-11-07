"""End-to-end: screenplay folder -> ranked promos. The ONE entry point used by the CLI, the web app
and the evaluation, so there is no duplicated pipeline logic anywhere.

    run(...)       the real run (progress events, rules-first preview, cancel)
    estimate(...)  a dry run in cache-only mode: how many NEW calls/tokens a run would need
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from .candidates import continuous_candidates, nms, stitched_candidates
from .config import Settings
from .llm import LLM
from .llm_stages import PROMPT_VERSIONS, blend_signals, judge, mine_moments, rerank, show_profile
from .parse import parse_show
from .rank import apply_rerank, finalize, rank
from .signals import annotate, resolve_lexicon

OUT_TOKENS = {"profile": 400, "miner": 600, "judge": 900, "rerank": 120}   # rough reply sizes for estimates


class ParseCoverageError(Exception):
    def __init__(self, report):
        super().__init__(f"only {report.coverage:.1%} of timestamp lines could be parsed")
        self.report = report


def clean_show_name(folder: str | Path) -> str:
    """'My Show-scripts (2)' -> 'My Show'. Same name everywhere = same prompts = cache hits."""
    name = Path(folder).name
    name = re.sub(r"[-_ ]*scripts?\b.*$", "", name, flags=re.I)
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)
    return name.strip(" -_") or Path(folder).name


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _rules_ranking(episodes, ep_by_num, beats, S: Settings):
    """Rules-only ranking (no LLM): used for the instant preview and for rules-only mode."""
    sig = annotate(episodes, S.lexicon, None, S.resolve_on_calm_tone)
    pool = [p for e in episodes for p in continuous_candidates(e, sig, S)] + stitched_candidates(episodes, sig, S)
    finalize(pool, sig, beats, [e.number for e in episodes], S)
    ranked, needs = rank(pool, ep_by_num, sig, S)
    return ranked, needs


def run(show_dir: str, llm: LLM, show_name: str | None = None, settings: Settings | None = None,
        episodes_filter: set[int] | None = None, progress=None, log=print, on_rules_result=None,
        force: bool = False, workers: int | None = None, extra_meta: dict | None = None) -> dict:
    S = settings or Settings()
    W = max(1, int(workers or llm.workers or 1))
    t0 = time.time()
    timings: dict[str, float] = {}
    show_name = (show_name or "").strip() or clean_show_name(show_dir)

    def emit(stage: str, done: int = 0, total: int = 0):
        if progress:
            st = llm.stats
            progress({"stage": stage, "done": done, "total": total, "calls": st["calls"],
                      "cache_hits": st["cache_hits"], "tokens_in": st["in_tokens"], "tokens_out": st["out_tokens"],
                      "retries": st["retries"], "elapsed": time.time() - t0, "est_cost": llm.est_cost()})

    def timed(name: str, t: float):
        timings[name] = round(time.time() - t, 2)

    # 1. parse (code) + coverage gate --------------------------------------------------------
    t = time.time()
    emit("parse")
    episodes, report = parse_show(show_dir, episodes_filter, S)
    timed("parse", t)
    if not episodes:
        raise SystemExit(f"No screenplay files with timestamped beats found in {show_dir}"
                         + (f" (episodes filter {sorted(episodes_filter)[:5]}...)" if episodes_filter else ""))
    if report.coverage < S.min_coverage and not force:
        raise ParseCoverageError(report)
    ep_by_num = {e.number: e for e in episodes}
    beats = {b.id: b for e in episodes for b in e.beats}
    numbers = [e.number for e in episodes]
    log(f"Parsed {len(episodes)} episodes, {len(beats)} beats, {sum(e.duration for e in episodes) / 60:.1f} min "
        f"of story · timestamp coverage {report.coverage:.1%}")

    # 2. instant rules-only preview (so the UI shows something in seconds) --------------------
    if on_rules_result is not None and llm.enabled:
        t = time.time()
        r_ranked, r_needs = _rules_ranking(episodes, ep_by_num, beats, S)
        timed("rules_preview", t)
        on_rules_result({"promos": r_ranked, "needs_context": r_needs, "ep_by_num": ep_by_num})

    # 3. show profile (1 LLM call) -> show-specific words for the rules ------------------------
    profile = {}
    if llm.enabled and S.show_profile:
        t = time.time()
        emit("profile", 0, 1)
        profile = show_profile(llm, show_name, episodes, log=log)
        emit("profile", 1, 1)
        timed("profile", t)

    # 4. rule signals (code) -----------------------------------------------------------------
    t = time.time()
    sig = annotate(episodes, S.lexicon, profile if llm.enabled else None, S.resolve_on_calm_tone)
    timed("rules", t)

    # 5. moment miner (LLM, 1 call/episode, parallel) ------------------------------------------
    miner_stats: dict = {}
    if llm.enabled:
        t = time.time()
        log(f"Stage A - moment mining with {llm.name} ({len(episodes)} calls, {W} workers, cached after first run)")
        emit("miner", 0, len(episodes))
        hooks, cliffs, miner_stats = mine_moments(llm, show_name, episodes, log=log, version=S.miner_prompt,
                                                  workers=W, tick=lambda d, n: emit("miner", d, n))
        blend_signals(sig, hooks, cliffs, S.w_llm, miner_stats.get("notes"), S.emotional_reveal_exempt)
        timed("miner", t)

    # 6. candidates (code) ---------------------------------------------------------------------
    t = time.time()
    cont = [p for e in episodes for p in continuous_candidates(e, sig, S)]
    stitched = stitched_candidates(episodes, sig, S)
    pool = cont + stitched
    timed("candidates", t)
    log(f"Generated {len(cont)} continuous + {len(stitched)} stitched candidates")

    # 7. cold-viewer judge on a shortlist (bounded cost regardless of show length) --------------
    judge_stats: dict = {}
    if llm.enabled and S.judge_enabled:
        t = time.time()
        # every episode's best clip is judged, then fill by score. Without this the rule pre-filter
        # starved whole episodes (bug B3).
        key = lambda p: p.rule["score"]
        best_per_ep = [max(g, key=key) for e in episodes
                       if (g := [p for p in cont if p.segments[0].episode == e.number])]
        rest = nms([p for p in pool if p not in best_per_ep], key=key, max_overlap=S.shortlist_overlap)
        shortlist = (best_per_ep + [p for p in rest if p not in best_per_ep])[:max(S.judge_top_n, len(best_per_ep))]
        n_batches = math.ceil(len(shortlist) / S.judge_batch)
        log(f"Stage B - cold-viewer judging {len(shortlist)} shortlisted clips ({n_batches} calls)")
        emit("judge", 0, n_batches)
        judge_stats = judge(llm, shortlist, ep_by_num, batch=S.judge_batch, log=log, version=S.judge_prompt,
                            workers=W, tick=lambda d, n: emit("judge", d, n))
        pool = [p for p in shortlist if p.llm] or shortlist
        timed("judge", t)

    # 8. blend, check, rank with diversity (code) ------------------------------------------------
    t = time.time()
    finalize(pool, sig, beats, numbers, S)
    ranked, needs = rank(pool, ep_by_num, sig, S)
    timed("rank", t)

    # 9. listwise rerank of the top N (LLM) -------------------------------------------------------
    rerank_stats: dict = {}
    if llm.enabled and S.rerank and len(ranked) > 1:
        t = time.time()
        top = ranked[:S.rerank_top]
        emit("rerank", 0, S.rerank_rounds)
        rr = rerank(llm, top, ep_by_num, rounds=S.rerank_rounds, seed=S.rerank_seed, log=log, workers=W,
                    tick=lambda d, n: emit("rerank", d, n))
        ranked = apply_rerank(ranked, rr["positions"], S.rerank_top)
        rerank_stats = {k: v for k, v in rr.items() if k != "positions"}
        timed("rerank", t)

    emit("done", 1, 1)
    judged = [p for p in pool if p.llm]
    meta = {
        "show": show_name, "episodes": len(episodes), "episode_range": [min(numbers), max(numbers)],
        "beats": len(beats), "story_minutes": round(sum(e.duration for e in episodes) / 60, 1),
        "model": llm.name, "provider": llm.provider, "workers": W,
        "settings": asdict(S), "lexicon_used": resolve_lexicon(S.lexicon),
        "prompt_versions": {k: v for k, v in PROMPT_VERSIONS.items()
                            if llm.enabled and (k in (f"miner:{S.miner_prompt}", f"judge:{S.judge_prompt}")
                                                or (k == "profile:v1" and S.show_profile)
                                                or (k == "rerank:v1" and S.rerank))},
        "candidates": {"continuous": len(cont), "stitched": len(stitched)},
        "returned": len(ranked), "needs_context": len(needs), "seconds": round(time.time() - t0, 1),
        "stage_seconds": timings,
        "llm_usage": {**dict(llm.stats), "est_cost_usd": round(llm.est_cost(), 4)},
        "miner": {k: v for k, v in miner_stats.items() if k != "notes"},
        "judge": judge_stats, "rerank": rerank_stats,
        "show_profile": {k: profile.get(k) for k in ("genre", "error") if profile.get(k)} if profile else {},
        "parse": {**report.summary(), "warnings_list": report.all_warnings[:50]},
        "check_pass_rates": _pass_rates(ranked),
        "cliff_score_spread": _spread(judged),
        **(extra_meta or {}),
    }
    return {"meta": meta, "promos": ranked, "needs_context": needs, "episodes": episodes, "ep_by_num": ep_by_num,
            "signals": sig, "miner_notes": miner_stats.get("notes", {}), "parse_report": report, "profile": profile}


def estimate(show_dir: str, llm: LLM, show_name: str | None = None, settings: Settings | None = None,
             episodes_filter: set[int] | None = None, force: bool = True) -> dict:
    """Dry run with cache_only=True: counts the calls that are NOT cached. Downstream prompts depend
    on upstream answers, so when an upstream stage has misses the downstream count is an upper bound."""
    S = settings or Settings()
    if not llm.enabled:
        return {"calls": 0, "in_tokens": 0, "out_tokens": 0, "by_stage": {}, "est_cost_usd": 0.0, "upper_bound": False}
    dry = copy.copy(llm)
    dry.cache_only, dry.log_path, dry.cancel = True, None, None
    dry.stats = {k: 0 for k in llm.stats}
    dry.misses = []
    import threading
    dry._lock = threading.Lock()
    _orig = dry.json

    def counting_json(system, user, tag="", schema=None):
        try:
            return _orig(system, user, tag, schema)
        except Exception:
            if dry.cached(system, user, schema) is None:
                dry.misses.append((tag, (len(system) + len(user)) // 4))
            raise
    dry.json = counting_json
    res = run(show_dir, dry, show_name, S, episodes_filter, log=lambda *_: None, force=force)
    by: dict[str, list] = {}
    for tag, tok in dry.misses:
        by.setdefault(tag.split(":")[0], []).append(tok)
    upstream_miss = bool(by.get("profile") or by.get("miner"))
    if upstream_miss and S.judge_enabled:       # judge/rerank prompts would change -> assume all miss
        n_j = math.ceil(min(S.judge_top_n, sum(res["meta"]["candidates"].values())) / S.judge_batch)
        by["judge"] = [3500] * n_j
        if S.rerank:
            by["rerank"] = [6000] * S.rerank_rounds
    calls = sum(len(v) for v in by.values())
    tin = sum(sum(v) for v in by.values())
    tout = sum(OUT_TOKENS.get(k, 500) * len(v) for k, v in by.items())
    return {"calls": calls, "in_tokens": tin, "out_tokens": tout,
            "by_stage": {k: len(v) for k, v in by.items()},
            "est_cost_usd": round(tin / 1e6 * llm.price_in + tout / 1e6 * llm.price_out, 4),
            "upper_bound": upstream_miss}


def _pass_rates(promos) -> dict:
    if not promos:
        return {}
    keys = sorted({k for p in promos for k in p.checks})
    return {k: round(sum(bool(p.checks.get(k)) for p in promos) / len(promos), 2) for k in keys}


def _spread(judged) -> dict:
    """How spread out are the judge's cliffhanger scores? (4b: the 0-10 score saturated at 8-9)"""
    s = [p.llm["cliffhanger"] / 10 for p in judged if "cliffhanger" in p.llm]
    c = [p.llm["check_score"] for p in judged if "check_score" in p.llm]
    out = {}
    if len(s) > 1:
        out["score_0_10"] = {"n": len(s), "mean": round(statistics.mean(s), 3), "stdev": round(statistics.pstdev(s), 3)}
    if len(c) > 1:
        out["check_based"] = {"n": len(c), "mean": round(statistics.mean(c), 3), "stdev": round(statistics.pstdev(c), 3)}
    return out
