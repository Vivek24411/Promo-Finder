"""Write run outputs: promos.json, promos.csv, report.md, report.html, run_meta.json."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .rank import segment_text


def promo_dict(p, ep_by_num) -> dict:
    return {
        "rank": int(p.id[1:]), "id": p.id, "list": "needs_context" if p.id.startswith("C") else "main", "kind": p.kind, "duration_sec": round(p.duration, 1),
        "score": round(p.final, 3),
        "segments": segment_text(p, ep_by_num),
        "hook": p.hook_text, "cliffhanger": p.cliff_text, "why_it_works": p.why,
        "hook_beat": p.hook_beat, "cliff_beat": p.cliff_beat,
        "component_scores": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.rule.items()},
        "cold_viewer": p.llm or None, "checks": p.checks, "needs_human_review": p.flags,
    }


def write_all(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ep_by_num = result["ep_by_num"]
    rows = [promo_dict(p, ep_by_num) for p in result["promos"]]
    ctx = [promo_dict(p, ep_by_num) for p in result.get("needs_context", [])]
    meta = result["meta"]

    (out / "promos.json").write_text(json.dumps({"meta": meta, "promos": rows, "needs_context": ctx},
                                                indent=2, ensure_ascii=False))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    if result.get("profile"):
        (out / "show_profile.json").write_text(json.dumps(result["profile"], indent=2, ensure_ascii=False))
    if result.get("parse_report") is not None:
        rep = result["parse_report"]
        (out / "parse_report.json").write_text(json.dumps(
            {**rep.summary(), "warnings": rep.all_warnings,
             "unparsed": [{"file": f, "line": n, "text": t} for f, n, t in rep.unparsed]}, indent=2, ensure_ascii=False))

    with open(out / "promos.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["list", "rank", "kind", "segments (episode start-end)", "duration_s", "score",
                    "hook", "cliffhanger", "why_it_works", "checks_failed", "needs_review"])
        for name, rs in (("main", rows), ("needs_context", ctx)):
            for r in rs:
                segs = " + ".join(f"Ep{s['episode']:02d} {s['start']}-{s['end']}" for s in r["segments"])
                failed = ";".join(k for k, v in r["checks"].items() if not v)
                w.writerow([name, r["rank"], r["kind"], segs, r["duration_sec"], r["score"], r["hook"],
                            r["cliffhanger"], r["why_it_works"], failed, " | ".join(r["needs_human_review"])])

    (out / "report.md").write_text(_markdown(meta, rows))
    (out / "report.html").write_text(_html(meta, rows, result["episodes"]))
    return out


def _markdown(meta, rows) -> str:
    L = [f"# Promo picks - {meta['show']}", "",
         f"Model: `{meta['model']}` · {meta['episodes']} episodes · {meta['story_minutes']} min of story · "
         f"{meta['candidates']['continuous']} + {meta['candidates']['stitched']} candidates -> top {meta['returned']}", "",
         "| # | Segments | Len | Score | Hook | Cliffhanger |", "|---|---|---|---|---|---|"]
    for r in rows:
        segs = "<br>".join(f"Ep{s['episode']:02d} {s['start']}–{s['end']}" for s in r["segments"])
        L.append(f"| {r['rank']} | {segs} | {r['duration_sec']}s | {r['score']} | {r['hook']} | {r['cliffhanger']} |")
    L.append("")
    for r in rows:
        L += [f"## {r['id']} · {r['kind']} · {r['duration_sec']}s · score {r['score']}", "",
              f"- **Hook:** {r['hook']}", f"- **Cliffhanger:** {r['cliffhanger']}",
              f"- **Why it works:** {r['why_it_works']}",
              f"- **Checks:** " + ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in r["checks"].items())]
        if r["needs_human_review"]:
            L.append(f"- **Needs review:** " + "; ".join(r["needs_human_review"]))
        for s in r["segments"]:
            L += ["", f"**Ep{s['episode']:02d} {s['start']} → {s['end']}**", "```"] + s["script"] + ["```"]
        L.append("")
    return "\n".join(L)


def _html(meta, rows, episodes) -> str:
    e = html.escape
    cards = []
    for r in rows:
        segs = "".join(f'<span class="seg">Ep{s["episode"]:02d} <b>{s["start"]} → {s["end"]}</b></span>' for s in r["segments"])
        checks = "".join(f'<span class="chk {"ok" if v else "bad"}">{"✓" if v else "✗"} {e(k.replace("_", " "))}</span>'
                         for k, v in r["checks"].items())
        flags = "".join(f'<li>{e(f)}</li>' for f in r["needs_human_review"])
        script = "".join(f'<div class="cut">Ep{s["episode"]:02d} {s["start"]} → {s["end"]}</div>' +
                         "".join(f"<div>{e(line)}</div>" for line in s["script"]) for s in r["segments"])
        cv = r["cold_viewer"]
        cvs = (f'<div class="cv">Cold viewer — hook {cv["hook"]:.0f}/10 · cliffhanger {cv["cliffhanger"]:.0f}/10 · '
               f'standalone {cv["standalone"]:.0f}/10</div>') if cv else ""
        cards.append(f"""
<article class="card">
  <header><span class="rank">#{r['rank']}</span><span class="kind {r['kind']}">{r['kind']}</span>
  <span class="dur">{r['duration_sec']}s</span><span class="score">score {r['score']:.2f}</span>
  <div class="bar"><i style="width:{min(100, r['score'] * 100):.0f}%"></i></div></header>
  <div class="segs">{segs}</div>
  <dl><dt>Hook</dt><dd>{e(r['hook'])}</dd><dt>Cliffhanger</dt><dd>{e(r['cliffhanger'])}</dd>
  <dt>Why</dt><dd>{e(r['why_it_works'])}</dd></dl>{cvs}
  <div class="checks">{checks}</div>
  {f'<ul class="flags">{flags}</ul>' if flags else ''}
  <details><summary>Script for this promo</summary><div class="script">{script}</div></details>
</article>""")

    # timeline: one row per episode, promo segments marked
    max_d = max(ep.duration for ep in episodes) or 1
    marks = {}
    for r in rows:
        for s in r["segments"]:
            marks.setdefault(s["episode"], []).append((s["start_sec"], s["end_sec"], r["rank"]))
    tl = []
    for ep in episodes:
        segs = "".join(f'<i title="#{rk}" style="left:{a / max_d * 100:.2f}%;width:{(b - a) / max_d * 100:.2f}%">{rk}</i>'
                       for a, b, rk in marks.get(ep.number, []))
        tl.append(f'<div class="tl"><span>Ep{ep.number:02d}</span><div class="track" style="width:{ep.duration / max_d * 100:.1f}%">{segs}</div></div>')

    pr = meta.get("check_pass_rates", {})
    prs = " · ".join(f"{k.replace('_', ' ')} {v * 100:.0f}%" for k, v in pr.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Promo Picks</title><style>
:root{{--bg:#f7f6f3;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e4e2dd;--acc:#c2410c;--ok:#15803d;--bad:#b91c1c;--chip:#f1efe9}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#151517;--card:#1e1e21;--ink:#ececee;--mute:#9a9aa1;--line:#2e2e33;--acc:#fb923c;--ok:#4ade80;--bad:#f87171;--chip:#26262a}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 64px}}h1{{font-size:24px;margin:0 0 4px}}.sub{{color:var(--mute);margin:0 0 20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 14px}}
header{{display:flex;flex-wrap:wrap;gap:8px;align-items:center}}.rank{{font-weight:700;font-size:18px}}
.kind,.dur,.score{{font-size:12px;padding:2px 8px;border-radius:99px;background:var(--chip)}}.kind.stitched{{color:var(--acc)}}
.bar{{flex:1 1 120px;height:6px;background:var(--chip);border-radius:9px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--acc)}}
.segs{{margin:10px 0;display:flex;gap:8px;flex-wrap:wrap}}.seg{{font:13px ui-monospace,monospace;background:var(--chip);padding:3px 8px;border-radius:6px}}
dl{{display:grid;grid-template-columns:96px 1fr;gap:4px 10px;margin:8px 0}}dt{{color:var(--mute);font-size:13px}}dd{{margin:0}}
.cv{{font-size:13px;color:var(--mute);margin:6px 0}}.checks{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}}
.chk{{font-size:12px;padding:1px 7px;border-radius:6px;border:1px solid var(--line)}}.chk.ok{{color:var(--ok)}}.chk.bad{{color:var(--bad);font-weight:600}}
.flags{{margin:8px 0 0;padding-left:18px;color:var(--bad);font-size:13px}}details{{margin-top:8px}}summary{{cursor:pointer;color:var(--mute);font-size:13px}}
.script{{font:12px/1.45 ui-monospace,monospace;background:var(--chip);padding:10px;border-radius:8px;max-height:340px;overflow:auto;margin-top:6px}}
.cut{{font-weight:700;color:var(--acc);margin:6px 0 2px}}.tl{{display:flex;align-items:center;gap:8px;margin:2px 0;font:11px ui-monospace,monospace}}
.tl span{{width:36px;color:var(--mute)}}.track{{position:relative;height:12px;background:var(--chip);border-radius:3px}}
.track i{{position:absolute;top:0;height:12px;background:var(--acc);color:#fff;font-style:normal;font-size:9px;line-height:12px;text-align:center;border-radius:3px;overflow:hidden}}
section.timeline{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:0 0 18px}}
section.timeline h2{{font-size:15px;margin:0 0 8px}}
</style></head><body><main>
<h1>Promo picks — {e(meta['show'])}</h1>
<p class="sub">{meta['episodes']} episodes · {meta['story_minutes']} min · model <code>{e(meta['model'])}</code> ·
{meta['candidates']['continuous']} continuous + {meta['candidates']['stitched']} stitched candidates → top {meta['returned']} ·
auto-check pass rates: {e(prs)}</p>
<section class="timeline"><h2>Where the promos come from</h2>{''.join(tl)}</section>
{''.join(cards)}
</main></body></html>"""
