#!/usr/bin/env python3
"""Build a BLIND rating sheet for human raters (3d) from a run's judged clips.

Use it on the SYNTHETIC show only - the real scripts are confidential and must not be shown to others.
Clips appear exactly as the judge saw them (clip-relative times, no logline, no scores), shuffled,
mixing high- and low-ranked clips.

  python run.py --show tests/fixtures/synthetic_show --preset gemini --out outputs/synthetic_rating
  python scripts/rating_sheet.py outputs/synthetic_rating --n 14
  -> outputs/synthetic_rating/rating_sheet.html (print or share) + rating_key.json (keep private)
"""
import argparse
import html
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def clip_text(p: dict) -> str:
    out, offset = [], 0.0
    for i, s in enumerate(p["segments"]):
        if i:
            out.append("[CUT TO A DIFFERENT MOMENT]")
        for line in s["script"]:
            m = re.match(r"\[\S+ (\d+):([\d.]+)-[^\]]+\] (.*)", line)
            if m:
                t = int(m.group(1)) * 60 + float(m.group(2))
                out.append(f"+{offset + max(0.0, t - s['start_sec']):04.1f}s  {m.group(3)}")
        offset += s["end_sec"] - s["start_sec"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run = Path(a.run)
    data = json.loads((run / "promos.json").read_text())
    judged = [p for p in data["promos"] + data.get("needs_context", []) if p.get("cold_viewer")]
    if len(judged) < 4:
        raise SystemExit("Need a run with judged clips (use an LLM preset).")
    judged.sort(key=lambda p: p["score"], reverse=True)
    half = a.n // 2
    pick = judged[:half] + judged[-(a.n - half):] if len(judged) > a.n else judged
    rng = random.Random(a.seed)
    rng.shuffle(pick)
    key = {f"C{i + 1:02d}": {"promo_id": p["id"], "judge": {k: p["cold_viewer"][k] for k in ("hook", "cliffhanger", "standalone")}}
           for i, p in enumerate(pick)}
    cards = "".join(f"""<section><h2>Clip {cid}</h2><pre>{html.escape(clip_text(p))}</pre>
<table><tr><td>Hook (0-10): would the first 3 s stop your scroll?</td><td class=b></td></tr>
<tr><td>Cliffhanger (0-10): do you NEED to know what happens next?</td><td class=b></td></tr>
<tr><td>Standalone (0-10): can you tell who is who and what's at stake?</td><td class=b></td></tr>
<tr><td colspan=2>What confused you? (optional)<div class=line></div></td></tr></table></section>"""
                    for cid, p in zip(key, pick))
    page = f"""<!doctype html><meta charset=utf-8><title>Promo rating sheet</title><style>
body{{font:14px/1.5 Georgia,serif;max-width:760px;margin:24px auto;padding:0 16px;color:#1b1b1b}}
section{{border:1px solid #ccc;padding:12px 16px;margin:0 0 18px;break-inside:avoid}}pre{{white-space:pre-wrap;font:12px/1.45 ui-monospace,monospace;background:#f6f4ef;padding:10px}}
table{{width:100%;border-collapse:collapse}}td{{padding:6px 4px;border-top:1px solid #eee}}td.b{{width:70px;border:1px solid #999}}.line{{border-bottom:1px solid #999;height:26px}}
</style><h1>Promo rating sheet</h1><p>Imagine scrolling TikTok. You have never seen this show. Rate each clip 0-10. Rater name: ________</p>{cards}"""
    (run / "rating_sheet.html").write_text(page)
    (run / "rating_key.json").write_text(json.dumps(key, indent=2))
    tmpl = run / "ratings_template.csv"
    tmpl.write_text("rater,clip,hook,cliffhanger,standalone,confused\n" + "".join(f"name,{c},,,,\n" for c in key))
    print(f"Wrote {run / 'rating_sheet.html'}, rating_key.json (private) and {tmpl.name} ({len(key)} clips)")


if __name__ == "__main__":
    main()
