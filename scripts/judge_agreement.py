#!/usr/bin/env python3
"""Compare human ratings with the AI cold-viewer judge (3d).

  python scripts/judge_agreement.py outputs/synthetic_rating ratings_alice.csv ratings_bob.csv
CSV columns: rater,clip,hook,cliffhanger,standalone[,confused]
Reports per dimension: Spearman correlation (judge vs mean human), mean absolute difference,
share of ratings within +-2 points, and whether the judge is systematically higher or lower.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

DIMS = ("hook", "cliffhanger", "standalone")


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1          # average rank for ties
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if len(a) < 3 or statistics.pstdev(ra) == 0 or statistics.pstdev(rb) == 0:
        return float("nan")
    return statistics.correlation(ra, rb)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    key = json.loads((Path(sys.argv[1]) / "rating_key.json").read_text())
    human: dict = {}
    for f in sys.argv[2:]:
        for row in csv.DictReader(open(f)):
            if row["clip"] in key and all(row.get(d, "").strip() for d in DIMS):
                human.setdefault(row["clip"], []).append({d: float(row[d]) for d in DIMS})
    clips = [c for c in key if c in human]
    if len(clips) < 3:
        raise SystemExit("Need ratings for at least 3 clips.")
    raters = len(sys.argv) - 2
    print(f"{len(clips)} clips rated by {raters} rater file(s)\n")
    print(f"{'dimension':<12} {'spearman':>9} {'mean |diff|':>12} {'within ±2':>10} {'judge - human':>14}")
    for d in DIMS:
        j = [key[c]["judge"][d] for c in clips]
        h = [statistics.mean(r[d] for r in human[c]) for c in clips]
        allpairs = [(key[c]["judge"][d], r[d]) for c in clips for r in human[c]]
        mad = statistics.mean(abs(a - b) for a, b in allpairs)
        within = sum(abs(a - b) <= 2 for a, b in allpairs) / len(allpairs)
        bias = statistics.mean(a - b for a, b in allpairs)
        side = "judge higher" if bias > 0.5 else "judge lower" if bias < -0.5 else "no clear bias"
        print(f"{d:<12} {spearman(j, h):>9.2f} {mad:>12.2f} {within:>9.0%} {bias:>+8.2f} ({side})")


if __name__ == "__main__":
    main()
