#!/usr/bin/env python3
"""Wall time for a 50-episode show on the mock server with simulated latency, for 1..N workers.

  python scripts/bench_mock.py --latency 1.5 --workers 1 4 8
(Builds a 50-episode show by repeating the synthetic episodes; no network, no cost.)
"""
import argparse
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from mock_llm import MockLLM  # noqa: E402
from promo_finder import config as C  # noqa: E402
from promo_finder.llm import LLM  # noqa: E402
from promo_finder.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=float, default=1.5, help="seconds per simulated model call")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--rpm", type=float, default=6000)
    a = ap.parse_args()
    src = sorted((ROOT / "tests" / "fixtures" / "synthetic_show").iterdir())
    tmp = Path(tempfile.mkdtemp(prefix="bench_show_"))
    for n in range(1, 51):
        text = src[(n - 1) % len(src)].read_text()
        (tmp / f"ep{n}.txt").write_text(re.sub(r"EPISODE NUMBER: Episode \d+", f"EPISODE NUMBER: Episode {n}", text))
    S = C.load(preset="v4").settings
    print(f"50 episodes, {a.latency}s simulated latency per call, rpm limit {a.rpm:g}")
    with MockLLM() as m:
        for w in a.workers:
            cache = Path(tempfile.mkdtemp(prefix="bench_cache_"))
            llm = LLM(provider="openai_compat", model=f"mock-latency={a.latency}", base_url=m.url, rpm=a.rpm,
                      workers=w, cache_dir=str(cache))
            t = time.time()
            res = run(str(tmp), llm, "bench", S, log=lambda *_: None)
            print(f"  workers={w:<2} {time.time() - t:6.1f}s  ({res['meta']['llm_usage']['calls']} calls)")
            shutil.rmtree(cache)
    shutil.rmtree(tmp)
    print("Real providers: time ≈ max(calls / rpm, calls × latency / workers). At Gemini free tier "
          "(~12 rpm) 62 calls take ≥ 5 min whatever the workers; on a paid tier (≥ 300 rpm) workers dominate.")


if __name__ == "__main__":
    main()
