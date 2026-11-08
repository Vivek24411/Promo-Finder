#!/usr/bin/env python3
"""Find and rank promos in a folder of microdrama screenplays.

  python run.py --show "/path/to/show-folder"                          # rules only, no API key needed
  python run.py --show ... --preset gemini                             # key from .env / GEMINI_API_KEY
  python run.py --show ... --preset gemini --scoring-preset v5         # the improved pipeline
  python run.py --show ... --preset openai --base-url https://their-gateway/v1 --model their-model
  python run.py --show ... --preset gemini --episodes 26-50 --workers 4
  python run.py --show ... --preset gemini --estimate                  # how many NEW calls? (spends nothing)

Before any run that needs uncached model calls, the estimated calls/tokens are shown and you are
asked to confirm (or pass --yes).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from promo_finder import config as C
from promo_finder.llm import PRESETS, Cancelled, LLMError
from promo_finder.parse import parse_episode_range
from promo_finder.pipeline import ParseCoverageError, clean_show_name, estimate, file_sha256, run
from promo_finder.report import write_all


def progress_bar():
    last = {"t": 0.0}

    def show(ev: dict):
        now = time.time()
        if ev["stage"] != "done" and now - last["t"] < 0.2:
            return
        last["t"] = now
        bar = ""
        if ev["total"]:
            n = int(20 * ev["done"] / ev["total"])
            bar = f"[{'#' * n}{'.' * (20 - n)}] {ev['done']}/{ev['total']} "
        cost = f" · ~${ev['est_cost']:.3f}" if ev["est_cost"] else ""
        sys.stderr.write(f"\r  {ev['stage']:<10} {bar}· calls {ev['calls']} (+{ev['cache_hits']} cached) · "
                         f"{(ev['tokens_in'] + ev['tokens_out']) / 1000:.1f}k tok · {ev['elapsed']:.0f}s{cost}   ")
        if ev["stage"] == "done":
            sys.stderr.write("\n")
        sys.stderr.flush()
    return show


def main():
    C.load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true", help="Offline source/boundary inspection; no model recommendations")
    ap.add_argument("--show", required=True, help="folder with one screenplay per episode (.txt or .md)")
    ap.add_argument("--name", help="show name (defaults to the folder name)")
    ap.add_argument("--config", help="config file (default: config.toml)")
    ap.add_argument("--scoring-preset", help="v6 (default) | v4 (submitted) | v5 | any [scoring_presets.*] in config.toml")
    ap.add_argument("--preset", help="model provider preset", choices=sorted(PRESETS))
    ap.add_argument("--model", help="override the preset's model id")
    ap.add_argument("--base-url", help="any OpenAI-compatible endpoint")
    ap.add_argument("--api-key-env", help="env var holding the key")
    ap.add_argument("--workers", type=int, help="parallel model calls (default per provider in config.toml)")
    ap.add_argument("--episodes", help="subset, e.g. 1-10 or 26-50 or 1,3,5")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--force", action="store_true", help="run even if parse coverage is below the threshold")
    ap.add_argument("--estimate", action="store_true", help="only report how many NEW calls a run would need")
    ap.add_argument("--yes", action="store_true", help="don't ask before making new model calls")
    ap.add_argument("--out", help="output folder (default outputs/<show>/<model>[__preset])")
    a = ap.parse_args()

    overrides = {"top_k": a.top_k} if a.top_k else None
    cfg = C.load(a.config, a.scoring_preset, overrides)
    show_dir = Path(a.show)
    name = a.name or clean_show_name(show_dir)
    try:
        eps = parse_episode_range(a.episodes)
    except ValueError as e:
        raise SystemExit(str(e))
    if cfg.scoring_preset == "v8":
        from promo_finder import v8
        import json
        if a.preset or a.model or a.base_url or a.episodes:
            raise SystemExit("v8 processes the complete input and configures Gemini/Claude on the server. Use an explicit legacy scoring preset for historical experiments.")
        if a.estimate:
            print(json.dumps(v8.estimate(show_dir), indent=2)); return
        try:
            result, out = v8.run(show_dir, out=a.out, inspection=a.inspect, show_name=name,
                                progress=lambda stage,done,total: print(f"{stage}: {done}/{total}"))
        except (ValueError, LLMError, OSError) as exc:
            raise SystemExit(str(exc)) from None
        print(json.dumps({"status":result["meta"]["status"],"recommended":len(result["promos"]),
                          "budget":result["meta"]["budget"],"outputs":str(out)},indent=2))
        return
    preset = a.preset or cfg.llm_preset
    llm = C.make_llm(cfg, preset, a.model, a.base_url, api_key_env=a.api_key_env, workers=a.workers)
    if llm.enabled and not llm.has_key():
        raise SystemExit(f"Set {llm.api_key_env} in .env or the environment (or use --preset none for rules only).")

    if llm.enabled:
        est = estimate(str(show_dir), llm, name, cfg.settings, eps, force=a.force)
        if est["calls"] or a.estimate:
            ub = " (upper bound)" if est["upper_bound"] else ""
            cost = f", ~${est['est_cost_usd']:.3f}" if est["est_cost_usd"] else ""
            print(f"New model calls needed: {est['calls']}{ub} {est['by_stage']} · ~{est['in_tokens'] / 1000:.0f}k in / "
                  f"~{est['out_tokens'] / 1000:.0f}k out tokens{cost} · {llm.name}")
        if a.estimate:
            return
        if est["calls"] and not a.yes:
            if not sys.stdin.isatty():
                raise SystemExit("New calls needed - re-run with --yes to allow them.")
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                raise SystemExit("Cancelled - nothing was spent.")
    elif a.estimate:
        print("Rules-only mode: no model calls.")
        return

    tag = (llm.model or "rules-only").replace("/", "_").replace(":", "_")
    if cfg.scoring_preset != "v4":
        tag += f"__{cfg.scoring_preset}"
    out = Path(a.out) if a.out else cfg.outputs_dir / _slug(name) / tag
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "llm_calls.jsonl"
    if log_path.exists():
        log_path.rename(out / f"llm_calls.{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    llm.log_path = str(log_path)

    try:
        result = run(str(show_dir), llm, show_name=name, settings=cfg.settings, episodes_filter=eps,
                     progress=progress_bar(), log=lambda m: print("\n" + m if m.startswith("Stage") else m, file=sys.stderr),
                     force=a.force, workers=a.workers,
                     extra_meta={"scoring_preset": cfg.scoring_preset,
                                 "heldout_sha256": file_sha256(cfg.heldout_path)})
    except ParseCoverageError as e:
        r = e.report
        print(f"\nStopped: only {r.parsed_lines}/{r.timestamp_lines} timestamp lines ({r.coverage:.1%}) could be "
              f"parsed. Unparsed lines:", file=sys.stderr)
        for f, n, t in r.unparsed[:15]:
            print(f"  {f}:{n}  {t}", file=sys.stderr)
        raise SystemExit("Fix the format, or re-run with --force to continue anyway.")
    except LLMError as e:
        raise SystemExit(f"\nModel error ({e.kind}): {e}")
    except (Cancelled, KeyboardInterrupt):
        raise SystemExit("\nCancelled.")
    write_all(result, out)
    m = result["meta"]
    u = m["llm_usage"]
    print(f"\nDone in {m['seconds']}s · {m['model']} · preset {cfg.scoring_preset} · calls {u['calls']} "
          f"(cache hits {u['cache_hits']}) · tokens {u['in_tokens']}/{u['out_tokens']}"
          + (f" · ~${u['est_cost_usd']:.3f}" if u["est_cost_usd"] else ""))
    for w in result["parse_report"].all_warnings[:5]:
        print(f"  warning: {w}")
    print(f"Outputs -> {out}/")
    for p in result["promos"][:10]:
        segs = " + ".join(s.label() for s in p.segments)
        print(f"  {p.id}  {p.final:.2f}  {p.duration:4.1f}s  {segs}  | {p.hook_text[:60]}")
    if result["needs_context"]:
        print(f"  (+{len(result['needs_context'])} promos in the 'needs context' list)")
    if el := m.get("eligibility"):
        s = el["status"]
        print(f"  {el['evaluated']} clips evaluated: {s['recommended']} eligible ({m['returned']} returned, max "
              f"{el['top_k_max']}), {s['review']} need review, {s['rejected']} rejected · judge: "
              f"{m['judge'].get('api_calls', 0)} new calls, {m['judge'].get('replies_valid', 0)} valid replies, "
              f"{m['judge'].get('validation_retries', 0)} validation retries, {m['judge'].get('clips_unjudged', 0)} unjudged")


def _slug(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")[:60]


if __name__ == "__main__":
    main()
