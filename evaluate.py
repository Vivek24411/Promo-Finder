#!/usr/bin/env python3
"""Score runs against the gold key moments - honestly (counts + 95% intervals, dev and held-out kept apart).

  python evaluate.py outputs/<show>/<run> [more runs ...]      # dev set (eval/gold_moments.json)
  python evaluate.py --heldout outputs/<show>/<run>            # held-out set; every look is logged
  python evaluate.py --ablation --show <folder> [--preset gemini] [--allow-api]

Metrics (K = 10 and 25):
  recall@K     share of gold moments that appear (>= half the moment, or 8 s) in at least one top-K promo
  gold_overlap@K  share of top-K promos overlapping selected positive examples; NOT precision or human acceptance
  key@K        share of gold moments whose single key instant (the slap, the line) is inside a top-K promo
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from promo_finder import config as C

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "eval" / "heldout_ledger.log"


# ---------------------------------------------------------------------------- stats
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (well-behaved for small n and for 0 or n)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n if n else 0:.2f} [{lo:.2f}-{hi:.2f}]"


# ---------------------------------------------------------------------------- matching
def covers(promo: dict, g: dict) -> bool:
    """Promo shows a substantial part of the moment: >= half of it, or >= 8 s of a long one."""
    need = min(0.5 * (g["end"] - g["start"]), 8.0)
    return any(s["episode"] == g["episode"] and
               min(s["end_sec"], g["end"]) - max(s["start_sec"], g["start"]) >= need
               for s in promo["segments"])


def shows_key(promo: dict, g: dict) -> bool:
    """Promo shows the single instant that makes the moment."""
    return "key" in g and any(s["episode"] == g["episode"] and s["start_sec"] - 0.3 <= g["key"] <= s["end_sec"] - 0.5
                              for s in promo["segments"])


def load_moments(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Moments file not found: {path}\n(see eval/gold_moments.example.json for the schema; "
                         f"set [eval] paths in config.toml)")
    data = json.loads(path.read_text())
    return data["moments"] if isinstance(data, dict) else data


def score_run(run_dir: Path, moments: list[dict]) -> dict:
    data = json.loads((run_dir / "promos.json").read_text())
    promos = data["promos"]
    n = len(moments)
    res = {"run": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
           "label": run_dir.name, "model": data["meta"].get("model"),
           "scoring_preset": data["meta"].get("scoring_preset", "v4"), "n_moments": n}
    for k in (10, 25):
        top = promos[:k]
        hit = [g for g in moments if any(covers(p, g) for p in top)]
        good = [p for p in top if any(covers(p, g) for g in moments)]
        keyh = [g for g in moments if any(shows_key(p, g) for p in top)]
        res[f"recall@{k}"] = (len(hit), n)
        res[f"gold_overlap@{k}"] = (len(good), len(top))
        res[f"key@{k}"] = (len(keyh), n)
        if k == 25:
            res["missed"] = [f"{g['id']} Ep{g['episode']}" for g in moments if g not in hit]
    res["check_pass_rates"] = data["meta"].get("check_pass_rates", {})
    return res


COLS = ["recall@10", "gold_overlap@10", "key@10", "recall@25", "gold_overlap@25", "key@25"]


def print_table(results: list[dict], title: str):
    n = results[0]["n_moments"] if results else 0
    print(f"\n{title}  ({n} moments: one moment = {100 / n:.0f} points)" if n else f"\n{title}")
    print("  values are hits/total = rate [95% Wilson interval]")
    w = max([len(r["label"]) for r in results] + [10])
    print("  " + "run".ljust(w) + "  " + "  ".join(c.ljust(26) for c in COLS))
    for r in results:
        print("  " + r["label"].ljust(w) + "  " + "  ".join(fmt_rate(*r[c]).ljust(26) for c in COLS))


def to_jsonable(r: dict) -> dict:
    out = dict(r)
    for c in COLS:
        k, n = r[c]
        lo, hi = wilson(k, n)
        out[c] = {"hits": k, "total": n, "rate": round(k / n, 3) if n else 0, "ci95": [round(lo, 3), round(hi, 3)]}
    return out


# ---------------------------------------------------------------------------- held-out ledger
def ledger_check(path: Path) -> str:
    """Record the held-out file's hash on first use; warn loudly if it ever changes."""
    from promo_finder.pipeline import file_sha256
    h = file_sha256(path)
    LEDGER.parent.mkdir(exist_ok=True)
    lines = LEDGER.read_text().splitlines() if LEDGER.exists() else []
    first = next((l for l in lines if l.startswith("INIT ")), None)
    if first is None:
        with open(LEDGER, "a") as f:
            f.write(f"INIT {h} {datetime.now().isoformat(timespec='seconds')}\n")
        print(f"[held-out] first use - hash {h[:12]}… recorded in {LEDGER.name}")
    elif first.split()[1] != h:
        print("\n" + "!" * 78 + f"\n!! HELD-OUT FILE CHANGED since first use ({first.split()[1][:12]}… -> {h[:12]}…).\n"
              "!! Results are no longer comparable with earlier held-out numbers.\n" + "!" * 78 + "\n")
    looks = sum(1 for l in lines if l.startswith("EVAL "))
    print(f"[held-out] this is look #{looks + 1} at the held-out set")
    return h


def ledger_append(h: str, r: dict):
    scores = " ".join(f"{c}={r[c][0]}/{r[c][1]}" for c in COLS)
    with open(LEDGER, "a") as f:
        f.write(f"EVAL {datetime.now().isoformat(timespec='seconds')} hash={h[:12]} preset={r['scoring_preset']} "
                f"run={r['label']} {scores}\n")


# ---------------------------------------------------------------------------- ablation
ABLATION = [
    ("1 rules only", "v4", "none"),
    ("2 AI miner only (no rule blend, no judge)", "miner_only", None),
    ("3 rules + miner, no judge", "no_judge", None),
    ("4 full v4", "v4", None),
    ("5a v4 + standalone geometric", "v4_standalone_geometric", None),
    ("5b v4 + standalone gate", "v4_standalone_gate", None),
    ("5c v4 + stitched rules", "v4_stitched_rules", None),
    ("5d v4 + calm tone != resolved", "v4_calm_not_resolved", None),
    ("5e v4 + cliffhanger checks", "v4_cliff_checks", None),
    ("5f v4 + listwise rerank", "v4_rerank", None),
    ("5g v4 + emotional reveals", "v4_emotional", None),
    ("5h v4 + show profile (generic lexicon)", "v4_show_profile", None),
    ("6 v5 (all changes)", "v5", None),
]


def ablation(a, cfg_path):
    from promo_finder.parse import parse_episode_range
    from promo_finder.pipeline import clean_show_name, estimate, run
    from promo_finder.report import write_all
    C.load_dotenv()
    base = C.load(cfg_path)
    show = Path(a.show)
    name = clean_show_name(show)
    eps = parse_episode_range(a.episodes)
    moments = load_moments(base.gold_path)
    out_root = base.outputs_dir / "ablation"
    plan = []
    for label, preset, llm_preset in ABLATION:
        cfg = C.load(cfg_path, preset)
        llm = C.make_llm(cfg, llm_preset or a.preset)
        est = estimate(str(show), llm, name, cfg.settings, eps) if llm.enabled else {"calls": 0, "in_tokens": 0, "out_tokens": 0}
        plan.append((label, preset, llm_preset or a.preset, cfg, llm, est))
    print("\nAblation plan (new = uncached model calls this variant would need):")
    for label, preset, lp, cfg, llm, est in plan:
        ub = "+" if est.get("upper_bound") else ""
        print(f"  {label:<44} preset={preset:<26} model={llm.name:<40} new calls={est['calls']}{ub} "
              f"(~{(est['in_tokens'] + est['out_tokens']) / 1000:.0f}k tokens)")
    total = sum(p[5]["calls"] for p in plan)
    print(f"  TOTAL new calls: {total} (a '+' means upper bound)")
    if total and not a.allow_api:
        print("  Variants needing new calls are SKIPPED. Re-run with --allow-api to spend them.")

    rows = []
    for label, preset, lp, cfg, llm, est in plan:
        if est["calls"] and not a.allow_api:
            rows.append({"label": label, "preset": preset, "status": f"skipped (needs {est['calls']} calls)"})
            continue
        out = out_root / preset.replace("/", "_") / (llm.model or "rules-only").replace("/", "_").replace(":", "_")
        llm.log_path = str(out / "llm_calls.jsonl")
        res = run(str(show), llm, name, cfg.settings, eps, log=lambda *_: None, force=True,
                  extra_meta={"scoring_preset": preset})
        write_all(res, out)
        r = score_run(out, moments)
        r["label"] = label
        rows.append({"label": label, "preset": preset, "status": "ok", "calls": res["meta"]["llm_usage"]["calls"],
                     **{c: r[c] for c in COLS}})
        print(f"  done: {label}")
    _write_ablation(rows, out_root, len(moments))


def _write_ablation(rows, out_root: Path, n: int):
    out_root.mkdir(parents=True, exist_ok=True)
    head = ["variant", "preset", "status"] + COLS
    md = [f"# Ablation (dev set, {n} moments: one moment = {100 / n:.0f} points)", "",
          "Values: hits/total = rate [95% Wilson interval]. Overlapping intervals = not a reliable difference.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    with open(out_root / "ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "preset", "status"] + [f"{c}_{x}" for c in COLS for x in ("hits", "total", "lo", "hi")])
        for r in rows:
            if r["status"] != "ok":
                md.append(f"| {r['label']} | {r['preset']} | {r['status']} |" + " |" * len(COLS))
                w.writerow([r["label"], r["preset"], r["status"]])
                continue
            md.append(f"| {r['label']} | {r['preset']} | ok | " + " | ".join(fmt_rate(*r[c]) for c in COLS) + " |")
            w.writerow([r["label"], r["preset"], "ok"] + [v for c in COLS for v in (*r[c], *[round(x, 3) for x in wilson(*r[c])])])
    (out_root / "ablation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nSaved {out_root / 'ablation.md'} and ablation.csv")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders (each with promos.json)")
    ap.add_argument("--config")
    ap.add_argument("--heldout", action="store_true", help="evaluate on the held-out set (logged in the ledger)")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--show", help="show folder (for --ablation)")
    ap.add_argument("--episodes")
    ap.add_argument("--preset", default="gemini", help="model preset for ablation variants")
    ap.add_argument("--allow-api", action="store_true", help="ablation: allow variants that need NEW model calls")
    a = ap.parse_args()
    cfg = C.load(a.config)
    if a.ablation:
        if not a.show:
            raise SystemExit("--ablation needs --show <folder>")
        return ablation(a, a.config)
    runs = [Path(r).resolve() for r in a.runs] or sorted(
        p.parent for p in cfg.outputs_dir.rglob("promos.json") if "ui_runs" not in p.parts)
    if not runs:
        raise SystemExit("No runs found.")
    path = cfg.heldout_path if a.heldout else cfg.gold_path
    moments = load_moments(path)
    h = ledger_check(path) if a.heldout else None
    results = [score_run(r, moments) for r in runs]
    print_table(results, "HELD-OUT set" if a.heldout else "DEV set (the original 20 moments)")
    for r in results:
        if r["missed"]:
            print(f"  [{r['label']}] missed in top-25: {', '.join(r['missed'])}")
        if h:
            ledger_append(h, r)
    out = cfg.outputs_dir / ("eval_heldout.json" if a.heldout else "eval_summary.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([to_jsonable(r) for r in results], indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    sys.exit(main())
