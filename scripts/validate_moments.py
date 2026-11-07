#!/usr/bin/env python3
"""Check a gold / held-out moments file against the schema (and optionally against the scripts).

  python scripts/validate_moments.py eval/heldout_moments.json [--show <folder>]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from promo_finder.parse import parse_show  # noqa: E402

REQUIRED = {"id": str, "episode": int, "start": (int, float), "end": (int, float), "key": (int, float), "what": str}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--show", help="also check episodes/times exist in this show")
    a = ap.parse_args()
    data = json.loads(Path(a.file).read_text())
    moments = data["moments"] if isinstance(data, dict) else data
    errors, ids = [], set()
    eps = {e.number: e for e in parse_show(a.show)[0]} if a.show else {}
    for i, m in enumerate(moments):
        where = f"moment #{i + 1} ({m.get('id', '?')})"
        for k, t in REQUIRED.items():
            if not isinstance(m.get(k), t):
                errors.append(f"{where}: '{k}' missing or wrong type")
        if errors and errors[-1].startswith(where):
            continue
        if m["id"] in ids:
            errors.append(f"{where}: duplicate id")
        ids.add(m["id"])
        if not m["start"] < m["end"]:
            errors.append(f"{where}: start must be before end")
        if not m["start"] - 0.5 <= m["key"] <= m["end"] + 0.5:
            errors.append(f"{where}: key instant {m['key']} is outside start-end")
        if eps:
            if m["episode"] not in eps:
                errors.append(f"{where}: episode {m['episode']} not in the show")
            elif m["end"] > eps[m["episode"]].duration + 1:
                errors.append(f"{where}: end {m['end']} is after the episode ends ({eps[m['episode']].duration:.1f}s)")
    print(f"{len(moments)} moments, {len(errors)} problem(s)")
    for e in errors:
        print("  -", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
