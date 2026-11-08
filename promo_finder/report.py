"""Write run outputs: promos.json, promos.csv, report.md, report.html, run_meta.json.

Every format is rendered from ONE list of records (promo_record), so the JSON, the CSV, both reports and
the aggregate numbers in run_meta.json describe exactly the same candidates, statuses and reasons.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .rank import segment_text

LISTS = {"P": "recommended", "R": "review", "X": "rejected", "C": "needs_context"}


def _list_name(p) -> str:
    if p.decision:
        return p.decision["status"]
    return "needs_context" if p.id.startswith("C") else "main"


def promo_record(p, ep_by_num) -> dict:
    d = p.decision or {}
    verified = bool(d)
    checks = p.checks
    review = list(d.get("review_reasons", [])) if verified else list(p.flags)
    rec = {
        "clip_key": p.key, "rank": int(p.id[1:]) if p.id[1:].isdigit() else 0, "id": p.id, "list": _list_name(p),
        "status": d["status"] if verified else _list_name(p),     # v4-v6 lists were never verified
        "kind": p.kind, "duration_sec": p.duration, "score": round(p.final, 3),
        "segments": segment_text(p, ep_by_num),
        "hook": p.hook_text, "cliffhanger": p.cliff_text, "final_moment": p.final_moment or "",
        "why_it_works": p.why, "limitations": p.limitations or "",
        "failure_reasons": list(d.get("failure_reasons", [])), "review_reasons": review,
        "notes": list(d.get("notes", [])) if verified else [],
        "hook_beat": p.hook_beat, "cliff_beat": p.cliff_beat,
        "checks": checks,
        "rank_explanation": p.rule.get("formula", ""),
        "component_scores": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.rule.items()
                             if k != "formula"},
        "cold_viewer": ({**{k: p.llm[k] for k in ("hook", "cliffhanger", "standalone") if k in p.llm},
                         "judgment": {k: p.judgment.get(k) for k in ("clip_id", "sections", "scores", "errors",
                                                                     "attempts", "unavailable") if k in p.judgment}}
                        if verified and p.judgment else (p.llm or None)),
        "origin": p.origin or None,
        # kept for older readers (app, rating sheet): everything a human should look at
        "needs_human_review": (review + list(d.get("notes", []))) if verified else list(p.flags),
    }
    return rec


promo_dict = promo_record          # backwards-compatible name


def check_status(v) -> str:
    """'pass' | 'fail' | 'uncertain' for a v7 check record or a v4-v6 boolean."""
    return v.get("status", "uncertain") if isinstance(v, dict) else ("pass" if v else "fail")


def build_records(result: dict) -> dict[str, list[dict]]:
    ep = result["ep_by_num"]
    out = {"recommended": [promo_record(p, ep) for p in result["promos"]],
           "review": [promo_record(p, ep) for p in result.get("review", [])],
           "rejected": [promo_record(p, ep) for p in result.get("rejected", [])],
           "needs_context": [promo_record(p, ep) for p in result.get("needs_context", [])]}
    return out


def write_all(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = build_records(result)
    meta = result["meta"]
    meta["records"] = {k: len(v) for k, v in recs.items()}

    (out / "promos.json").write_text(json.dumps(
        {"meta": meta, "promos": recs["recommended"], "review": recs["review"], "rejected": recs["rejected"],
         "needs_context": recs["needs_context"]}, indent=2, ensure_ascii=False))
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
        w.writerow(["list", "rank", "id", "status", "kind", "segments (episode start-end)", "duration_s", "score",
                    "hook", "cliffhanger", "final_moment", "why_it_works", "limitations", "checks_failed",
                    "checks_uncertain", "failure_reasons", "review_reasons", "notes", "rank_explanation", "origin"])
        for name in ("recommended", "review", "rejected", "needs_context"):
            for r in recs[name]:
                segs = " + ".join(f"Ep{s['episode']:02d} {s['start']}-{s['end']}" for s in r["segments"])
                st = {k: check_status(v) for k, v in r["checks"].items()}
                w.writerow([r["list"], r["rank"], r["id"], r["status"], r["kind"], segs, r["duration_sec"], r["score"],
                            r["hook"], r["cliffhanger"], r["final_moment"], r["why_it_works"], r["limitations"],
                            ";".join(k for k, s in st.items() if s == "fail"),
                            ";".join(k for k, s in st.items() if s == "uncertain"),
                            " | ".join(r["failure_reasons"]), " | ".join(r["review_reasons"]), " | ".join(r["notes"]),
                            r["rank_explanation"], (r["origin"] or {}).get("change", "")])

    (out / "report.md").write_text(_markdown(meta, recs))
    (out / "report.html").write_text(_html(meta, recs, result["episodes"]))
    return out


# ----------------------------------------------------------------------------- markdown
def _seglist(r, sep=" + ") -> str:
    return sep.join(f"Ep{s['episode']:02d} {s['start']}–{s['end']}" for s in r["segments"])


def _check_line(r) -> str:
    sym = {"pass": "✓", "fail": "✗", "uncertain": "?"}
    return ", ".join(f"{k} {sym[check_status(v)]}" for k, v in r["checks"].items())


def _summary(meta) -> str:
    el = meta.get("eligibility")
    if not el:
        return ""
    s = el["status"]
    return (f"{el['evaluated']} clips evaluated: {s['recommended']} eligible, {s['review']} need review, "
            f"{s['rejected']} rejected. Returned {meta['returned']} recommended (maximum {el['top_k_max']}).")


def _markdown(meta, recs) -> str:
    rows = recs["recommended"] + recs["needs_context"]
    L = [f"# Promo picks - {meta['show']}", "",
         f"Model: `{meta['model']}` · {meta['episodes']} episodes · {meta['story_minutes']} min of story · "
         f"{meta['candidates']['continuous']} + {meta['candidates']['stitched']} candidates -> {meta['returned']} recommended",
         ""]
    if s := _summary(meta):
        L += [s, ""]
    L += ["| # | Status | Segments | Len | Score | Hook | Cliffhanger |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['id']} | {r['status']} | {_seglist(r, '<br>')} | {r['duration_sec']}s | {r['score']} | "
                 f"{r['hook']} | {r['cliffhanger']} |")
    L.append("")

    def card(r):
        out = [f"## {r['id']} · {r['status']} · {r['kind']} · {r['duration_sec']}s · score {r['score']}", "",
               f"- **Hook:** {r['hook']}", f"- **Ends on:** {r['final_moment']}" if r["final_moment"] else "",
               f"- **Cliffhanger:** {r['cliffhanger']}" if r["cliffhanger"] else "",
               f"- **Why it works:** {r['why_it_works']}",
               f"- **Limitations:** {r['limitations']}" if r["limitations"] else "",
               f"- **Checks:** {_check_line(r)}"]
        if r["rank_explanation"]:
            out.append(f"- **Score:** {r['rank_explanation']}")
        for label, key in (("Failed", "failure_reasons"), ("Needs review", "review_reasons"), ("Notes", "notes")):
            if r[key]:
                out.append(f"- **{label}:** " + "; ".join(r[key]))
        for s in r["segments"]:
            out += ["", f"**Ep{s['episode']:02d} {s['start']} → {s['end']}**", "```"] + s["script"] + ["```"]
        return [x for x in out if x != ""] + [""]

    for r in rows:
        L += card(r)
    if recs["review"]:
        L += ["# Needs review", "", "Not verified: at least one check is uncertain. Never ranked above a recommended clip.", ""]
        for r in recs["review"]:
            L += card(r)
    if recs["rejected"]:
        L += ["# Rejected", "", "| # | Segments | Len | Score | Why rejected |", "|---|---|---|---|---|"]
        for r in recs["rejected"]:
            L.append(f"| {r['id']} | {_seglist(r, '<br>')} | {r['duration_sec']}s | {r['score']} | "
                     + "; ".join(r["failure_reasons"] + r["notes"][-1:]).replace("|", "/") + " |")
        L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------- html
def _card_html(r, e) -> str:
    segs = "".join(f'<span class="seg">Ep{s["episode"]:02d} <b>{s["start"]} → {s["end"]}</b></span>' for s in r["segments"])
    sym = {"pass": "✓", "fail": "✗", "uncertain": "?"}
    checks = "".join(f'<span class="chk {check_status(v)}" title="{e(v.get("reason", "") if isinstance(v, dict) else "")}">'
                     f'{sym[check_status(v)]} {e(k.replace("_", " "))}</span>' for k, v in r["checks"].items())
    lists = ""
    for cls, label, key in (("bad", "Failed", "failure_reasons"), ("warn", "Needs review", "review_reasons"),
                            ("note", "Notes", "notes")):
        if r[key]:
            lists += f'<div class="{cls}"><b>{label}</b><ul>' + "".join(f"<li>{e(x)}</li>" for x in r[key]) + "</ul></div>"
    if not r["review_reasons"] and not r["notes"] and r["needs_human_review"] and not r["failure_reasons"]:
        lists += '<div class="warn"><ul>' + "".join(f"<li>{e(x)}</li>" for x in r["needs_human_review"]) + "</ul></div>"
    script = "".join(f'<div class="cut">Ep{s["episode"]:02d} {s["start"]} → {s["end"]}</div>' +
                     "".join(f"<div>{e(line)}</div>" for line in s["script"]) for s in r["segments"])
    cv = r["cold_viewer"] or {}
    cvs = (f'<div class="cv">Judge scores (0-10): hook {cv["hook"]:.0f} · cliffhanger {cv["cliffhanger"]:.0f} · '
           f'standalone {cv["standalone"]:.0f}</div>') if all(k in cv for k in ("hook", "cliffhanger", "standalone")) else ""
    dl = f"<dt>Hook</dt><dd>{e(r['hook'])}</dd>"
    if r["final_moment"]:
        dl += f"<dt>Ends on</dt><dd>{e(r['final_moment'])}</dd>"
    if r["cliffhanger"]:
        dl += f"<dt>Cliffhanger</dt><dd>{e(r['cliffhanger'])}</dd>"
    dl += f"<dt>Why</dt><dd>{e(r['why_it_works'])}</dd>"
    if r["limitations"]:
        dl += f"<dt>Limitations</dt><dd>{e(r['limitations'])}</dd>"
    formula = f'<div class="cv">Score: {e(r["rank_explanation"])}</div>' if r["rank_explanation"] else ""
    return f"""
<article class="card {e(r['status'])}">
  <header><span class="rank">{e(r['id'])}</span><span class="st {e(r['status'])}">{e(r['status'])}</span>
  <span class="kind {r['kind']}">{r['kind']}</span><span class="dur">{r['duration_sec']}s</span>
  <span class="score">score {r['score']:.2f}</span>
  <div class="bar"><i style="width:{min(100, r['score'] * 100):.0f}%"></i></div></header>
  <div class="segs">{segs}</div>
  <dl>{dl}</dl>{cvs}{formula}
  <div class="checks">{checks}</div>
  {lists}
  <details><summary>Script for this promo</summary><div class="script">{script}</div></details>
</article>"""


def _html(meta, recs, episodes) -> str:
    e = html.escape
    rows = recs["recommended"] + recs["needs_context"]
    cards = [_card_html(r, e) for r in rows]
    review = [_card_html(r, e) for r in recs["review"]]
    rejected = "".join(f"<tr><td>{e(r['id'])}</td><td>{e(_seglist(r))}</td><td>{r['duration_sec']}s</td>"
                       f"<td>{r['score']:.2f}</td><td>{e('; '.join(r['failure_reasons'] + r['notes'][-1:]))}</td></tr>"
                       for r in recs["rejected"])

    # timeline: one row per episode, promo segments marked
    max_d = max(ep.duration for ep in episodes) or 1
    marks = {}
    for r in rows:
        for s in r["segments"]:
            marks.setdefault(s["episode"], []).append((s["start_sec"], s["end_sec"], r["id"]))
    tl = []
    for ep in episodes:
        segs = "".join(f'<i title="{rk}" style="left:{a / max_d * 100:.2f}%;width:{(b - a) / max_d * 100:.2f}%">{rk[1:]}</i>'
                       for a, b, rk in marks.get(ep.number, []))
        tl.append(f'<div class="tl"><span>Ep{ep.number:02d}</span><div class="track" style="width:{ep.duration / max_d * 100:.1f}%">{segs}</div></div>')

    el = meta.get("eligibility")
    if el:
        sub = e(_summary(meta))
    else:
        pr = meta.get("check_pass_rates", {})
        sub = "auto-check pass rates: " + e(" · ".join(f"{k.replace('_', ' ')} {v * 100:.0f}%" for k, v in pr.items()))
    empty = '<p class="sub">No clip passed every check.</p>' if el and not rows else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Promo Picks</title><style>
:root{{--bg:#f7f6f3;--card:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e4e2dd;--acc:#c2410c;--ok:#15803d;--bad:#b91c1c;--warn:#a16207;--chip:#f1efe9}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#151517;--card:#1e1e21;--ink:#ececee;--mute:#9a9aa1;--line:#2e2e33;--acc:#fb923c;--ok:#4ade80;--bad:#f87171;--warn:#facc15;--chip:#26262a}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 64px}}h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:19px;margin:28px 0 8px}}.sub{{color:var(--mute);margin:0 0 20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 14px}}
header{{display:flex;flex-wrap:wrap;gap:8px;align-items:center}}.rank{{font-weight:700;font-size:18px}}
.kind,.dur,.score,.st{{font-size:12px;padding:2px 8px;border-radius:99px;background:var(--chip)}}.kind.stitched{{color:var(--acc)}}
.st.recommended{{color:var(--ok)}}.st.review{{color:var(--warn)}}.st.rejected{{color:var(--bad)}}
.bar{{flex:1 1 120px;height:6px;background:var(--chip);border-radius:9px;overflow:hidden}}.bar i{{display:block;height:100%;background:var(--acc)}}
.segs{{margin:10px 0;display:flex;gap:8px;flex-wrap:wrap}}.seg{{font:13px ui-monospace,monospace;background:var(--chip);padding:3px 8px;border-radius:6px}}
dl{{display:grid;grid-template-columns:96px 1fr;gap:4px 10px;margin:8px 0}}dt{{color:var(--mute);font-size:13px}}dd{{margin:0}}
.cv{{font-size:13px;color:var(--mute);margin:6px 0}}.checks{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}}
.chk{{font-size:12px;padding:1px 7px;border-radius:6px;border:1px solid var(--line)}}.chk.pass{{color:var(--ok)}}.chk.fail{{color:var(--bad);font-weight:600}}.chk.uncertain{{color:var(--warn);font-weight:600}}
.bad,.warn,.note{{font-size:13px;margin:8px 0 0}}.bad{{color:var(--bad)}}.warn{{color:var(--warn)}}.note{{color:var(--mute)}}.bad ul,.warn ul,.note ul{{margin:2px 0;padding-left:18px}}
details{{margin-top:8px}}summary{{cursor:pointer;color:var(--mute);font-size:13px}}
.script{{font:12px/1.45 ui-monospace,monospace;background:var(--chip);padding:10px;border-radius:8px;max-height:340px;overflow:auto;margin-top:6px}}
.cut{{font-weight:700;color:var(--acc);margin:6px 0 2px}}.tl{{display:flex;align-items:center;gap:8px;margin:2px 0;font:11px ui-monospace,monospace}}
.tl span{{width:36px;color:var(--mute)}}.track{{position:relative;height:12px;background:var(--chip);border-radius:3px}}
.track i{{position:absolute;top:0;height:12px;background:var(--acc);color:#fff;font-style:normal;font-size:9px;line-height:12px;text-align:center;border-radius:3px;overflow:hidden}}
section.timeline{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:0 0 18px}}
section.timeline h2{{font-size:15px;margin:0 0 8px}}
.tbl{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}}td,th{{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}}
</style></head><body><main>
<h1>Promo picks — {e(meta['show'])}</h1>
<p class="sub">{meta['episodes']} episodes · {meta['story_minutes']} min · model <code>{e(meta['model'])}</code> ·
{meta['candidates']['continuous']} continuous + {meta['candidates']['stitched']} stitched candidates → {meta['returned']} recommended · {sub}</p>
<section class="timeline"><h2>Where the recommended promos come from</h2>{''.join(tl)}</section>
{empty}{''.join(cards)}
{f'<h2>Needs review ({len(review)})</h2><p class="sub">Not verified: at least one check is uncertain.</p>' + ''.join(review) if review else ''}
{f'<h2>Rejected ({len(recs["rejected"])})</h2><div class="tbl"><table><tr><th>#</th><th>Segments</th><th>Len</th><th>Score</th><th>Why</th></tr>{rejected}</table></div>' if rejected else ''}
</main></body></html>"""
