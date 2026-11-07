"""End-to-end on the synthetic show with the mock LLM (2e), plus evaluation helpers (3a/3b)."""
import json
from pathlib import Path

import pytest

from conftest import ROOT, SYNTHETIC
from promo_finder import config as C
from promo_finder.llm import LLM
from promo_finder.pipeline import estimate, run
from promo_finder.report import write_all

OUTPUT_FILES = {"promos.json", "promos.csv", "report.md", "report.html", "run_meta.json", "parse_report.json"}


def check_outputs(res, out: Path, S):
    beat_ids = {b.id for e in res["episodes"] for b in e.beats}
    for p in res["promos"] + res["needs_context"]:
        assert S.min_len <= p.duration <= S.max_len, p.duration
        assert p.checks["clean_cuts_no_mid_line"]
        assert p.hook_beat in beat_ids and p.cliff_beat in beat_ids
        for s in p.segments:
            ep = res["ep_by_num"][s.episode]
            assert 0 <= s.start < s.end <= ep.duration + 0.01
    assert OUTPUT_FILES <= {f.name for f in out.iterdir()}
    data = json.loads((out / "promos.json").read_text())
    assert data["promos"] and data["meta"]["parse"]["coverage"] == 1.0


@pytest.mark.parametrize("preset", ["v4", "v5"])
def test_end_to_end_synthetic(make_llm, tmp_path, preset):
    S = C.load(preset=preset).settings
    llm = make_llm()
    seen = []
    res = run(str(SYNTHETIC), llm, "The CEO's Hidden Wife", S, progress=seen.append, log=lambda *_: None,
              on_rules_result=lambda r: seen.append({"stage": "rules_preview", "n": len(r["promos"])}))
    out = write_all(res, tmp_path / "out")
    check_outputs(res, out, S)
    stages = {e["stage"] for e in seen}
    assert {"parse", "miner", "judge", "done", "rules_preview"} <= stages
    m = res["meta"]
    assert m["miner"]["invalid_ids"] == 0 and m["judge"]["judged"] > 0
    assert set(m["stage_seconds"]) >= {"parse", "rules", "miner", "candidates", "judge", "rank"}
    assert m["prompt_versions"] and m["workers"] == 4
    if preset == "v5":
        assert (out / "show_profile.json").exists()
        prof = json.loads((out / "show_profile.json").read_text())
        assert "neverappearsanywhere" in prof["dropped_words"]          # invented word rejected by code
        assert m["rerank"]["valid_rounds"] == S.rerank_rounds
        assert m["judge"]["checks_answered"] > 0 and "check_based" in m["cliff_score_spread"]
        assert sum(p.kind == "stitched" for p in res["promos"]) <= S.max_stitched_in_top
        for p in res["needs_context"]:
            assert p.id.startswith("C")


def test_rules_only_uses_generic_lexicon_and_needs_no_key(tmp_path):
    S = C.load(preset="v5").settings
    res = run(str(SYNTHETIC), LLM(), settings=S, log=lambda *_: None)
    assert res["promos"] and res["meta"]["model"] == "rules-only" and not res["profile"]


def test_estimate_counts_then_zero_after_run(make_llm):
    S = C.load(preset="v4").settings
    llm = make_llm()
    est = estimate(str(SYNTHETIC), llm, "X", S)
    assert est["by_stage"]["miner"] == 10 and est["calls"] >= 10 and est["upper_bound"]
    run(str(SYNTHETIC), llm, "X", S, log=lambda *_: None)
    assert estimate(str(SYNTHETIC), llm, "X", S)["calls"] == 0


def test_episode_subset_keeps_funnel_in_range(make_llm):
    """Bug O1: on a subset like 6-10 the early-episode bonus used to go negative."""
    res = run(str(SYNTHETIC), LLM(), settings=C.load(preset="v4").settings, episodes_filter={6, 7, 8, 9, 10},
              log=lambda *_: None)
    assert {e.number for e in res["episodes"]} == {6, 7, 8, 9, 10}
    assert all(0.29 <= p.rule["funnel"] <= 1.0 for p in res["promos"])


def test_cancel_stops_further_calls(make_llm, mock_server):
    llm = make_llm("mock-latency=0.2", workers=2)

    def progress(ev):
        if ev["stage"] == "miner" and ev["done"] >= 2:
            raise KeyboardInterrupt          # what a UI cancel looks like from inside the run
    before = mock_server.calls
    with pytest.raises(KeyboardInterrupt):
        run(str(SYNTHETIC), llm, "X", C.load().settings, progress=progress, log=lambda *_: None)
    assert mock_server.calls - before < 10      # did not go on to mine all 10 episodes


def test_parallel_workers_are_faster(make_llm):
    S = C.load().settings
    import time
    t = time.time()
    run(str(SYNTHETIC), make_llm("mock-latency=0.15", workers=1), "A", S, log=lambda *_: None)
    serial = time.time() - t
    t = time.time()
    run(str(SYNTHETIC), make_llm("mock-latency=0.15", workers=5, cache_dir=str(Path(make_llm().cache_dir).parent / "c2")),
        "A", S, log=lambda *_: None)
    assert time.time() - t < serial * 0.6


# ---------------------------------------------------------------------------- evaluation helpers
def test_wilson_interval():
    from evaluate import wilson
    lo, hi = wilson(13, 20)
    assert 0.43 < lo < 0.44 and 0.81 < hi < 0.82
    assert wilson(0, 20)[0] == 0.0 and wilson(20, 20)[1] == 1.0


def test_heldout_ledger(tmp_path, monkeypatch, capsys):
    import evaluate
    monkeypatch.setattr(evaluate, "LEDGER", tmp_path / "ledger.log")
    f = tmp_path / "heldout.json"
    f.write_text('{"moments": []}')
    h = evaluate.ledger_check(f)
    evaluate.ledger_check(f)
    f.write_text('{"moments": [1]}')
    evaluate.ledger_check(f)
    out = capsys.readouterr().out
    assert "first use" in out and "HELD-OUT FILE CHANGED" in out and h


# ---------------------------------------------------------------------------- real-data v4 regression
# Paths to the confidential input pack live in a gitignored file, so the show is never named in git.
_PRIV = ROOT / "eval" / "paths.private.json"
_paths = json.loads(_PRIV.read_text()) if _PRIV.exists() else {}
REAL = Path(_paths.get("real_show", "/nonexistent"))
SAVED = Path(_paths.get("v4_promos", "/nonexistent"))


@pytest.mark.skipif(not (REAL.exists() and SAVED.exists()), reason="private input pack / v4 outputs not present")
def test_v4_reproduces_from_cache():
    cfg = C.load(preset="v4")
    llm = C.make_llm(cfg, "gemini")
    llm.cache_only = True                     # proves zero spending
    res = run(str(REAL), llm, None, cfg.settings, log=lambda *_: None)
    saved = json.loads(SAVED.read_text())["promos"]
    got = [(tuple((s.episode, round(s.start, 2), round(s.end, 2)) for s in p.segments), round(p.final, 3)) for p in res["promos"]]
    exp = [(tuple((s["episode"], s["start_sec"], s["end_sec"]) for s in p["segments"]), p["score"]) for p in saved]
    assert got == exp and llm.stats["calls"] == 0
