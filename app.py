"""Promo Finder - the editor's web app.

    streamlit run app.py

Load -> Check -> Run -> Review. Calls the same pipeline.run() as the CLI (no duplicated logic).
Uploaded scripts only touch disk inside a temporary folder that is deleted right after each step;
typed API keys live only in this browser session's memory.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

from promo_finder import config as C
from promo_finder.llm import PRESETS, Cancelled, LLMError
from promo_finder.parse import fmt, parse_episode_range, parse_show
from promo_finder.pipeline import ParseCoverageError, clean_show_name, estimate, file_sha256, run
from promo_finder.report import promo_dict, write_all

HERE = Path(__file__).parent
SAMPLE = HERE / "tests" / "fixtures" / "synthetic_show"
C.load_dotenv()
BASE = C.load()
RUNS_DIR = BASE.outputs_dir / "ui_runs"
# Public mode (set PROMO_FINDER_PUBLIC=1 on a shared server): visitors must not reach the server's files,
# other visitors' runs, the shared cache, or send the server's key to an arbitrary URL.
PUBLIC = os.environ.get("PROMO_FINDER_PUBLIC", "").strip().lower() in ("1", "true", "yes")

PROVIDER_LABELS = {"none": "Rules only (no key)", "gemini": "Google Gemini", "anthropic": "Anthropic Claude",
                   "openai": "OpenAI / custom endpoint", "groq": "Groq", "openrouter": "OpenRouter", "ollama": "Ollama (local)"}
PRESET_LABELS = {"v4": "v4: submitted", "v6": "v6: v4 + safe changes", "v5": "v5: all changes"}
STAGE_LABELS = {"parse": "Reading scripts", "profile": "Profiling the show", "miner": "Finding hook and cliffhanger beats",
                "judge": "Cold-viewer review", "rerank": "Head-to-head ranking", "done": "Finishing"}

st.set_page_config(page_title="Promo Finder", page_icon=":material/movie:", layout="wide",
                   initial_sidebar_state="expanded")

st.html("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--ink:#1C1A17;--mute:#6F685E;--line:#DED8CC;--paper:#F6F4EF;--card:#FFFFFF;--accent:#B5502A;--ok:#2F6B4F;--warn:#9A6A12;--bad:#A23B2A}
[data-testid="stDecoration"]{display:none}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1140px;padding-top:2.2rem}
h1,h2,h3,.pf-serif{font-family:'Newsreader',Georgia,serif!important;font-weight:500!important;letter-spacing:-.01em}
.pf-top{display:flex;align-items:baseline;justify-content:space-between;gap:16px;border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:6px}
.pf-word{font-family:'Newsreader',Georgia,serif;font-size:34px;line-height:1;color:var(--ink)}
.pf-word em{font-style:italic;color:var(--accent)}
.pf-sub{color:var(--mute);font-size:14px}
.pf-chips{display:flex;gap:8px;flex-wrap:nowrap;white-space:nowrap}
.pf-chip{font:500 12px 'JetBrains Mono',monospace;color:var(--mute);border:1px solid var(--line);border-radius:999px;padding:3px 10px;background:var(--card)}
.pf-steps{display:flex;gap:0;margin:14px 0 22px;font-size:13px}
.pf-step{flex:1;padding:8px 0;border-top:2px solid var(--line);color:var(--mute)}
.pf-step b{font-family:'JetBrains Mono',monospace;font-weight:500;margin-right:6px}
.pf-step.on{border-top-color:var(--ink);color:var(--ink)}
.pf-step.done{border-top-color:var(--accent);color:var(--ink)}
.pf-label{font:600 11px Inter,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--mute);margin:4px 0 8px}
.pf-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:6px 0 12px}
.pf-stat{background:var(--card);padding:12px 14px}
.pf-stat .v{font-family:'Newsreader',Georgia,serif;font-size:26px;line-height:1.1;color:var(--ink)}
.pf-stat .k{font-size:12px;color:var(--mute);margin-top:2px}
.pf-note{border-left:3px solid var(--warn);background:#FBF6EA;padding:9px 12px;border-radius:0 8px 8px 0;font-size:13.5px;margin:6px 0}
.pf-note.bad{border-left-color:var(--bad);background:#FBEDEA}
.pf-note.ok{border-left-color:var(--ok);background:#EEF5F0}
.pf-head{display:flex;gap:18px;align-items:flex-start}
.pf-rank{font-family:'Newsreader',Georgia,serif;font-size:40px;line-height:.9;color:var(--accent);min-width:54px}
.pf-tc{display:inline-block;font:500 13px 'JetBrains Mono',monospace;color:var(--ink);background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:3px 8px;margin:0 6px 6px 0}
.pf-meta{font-size:12.5px;color:var(--mute)}
.pf-dl{display:grid;grid-template-columns:96px 1fr;gap:6px 14px;margin:12px 0 10px;font-size:14.5px}
.pf-dl dt{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.06em;padding-top:2px}
.pf-dl dd{margin:0;color:var(--ink)}
.pf-meters{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--mute);margin:4px 0 8px}
.pf-meter{display:flex;align-items:center;gap:6px}
.pf-meter i{display:inline-block;width:70px;height:4px;background:var(--line);border-radius:4px;position:relative;overflow:hidden}
.pf-meter i s{position:absolute;inset:0 auto 0 0;background:var(--ink);border-radius:4px;text-decoration:none}
.pf-badges{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
.pf-b{font-size:11.5px;border-radius:6px;padding:2px 7px;border:1px solid var(--line);color:var(--mute);background:var(--card)}
.pf-b.fail{color:var(--bad);border-color:#E7C4BC;background:#FBEDEA}
.pf-flags{font-size:12.5px;color:var(--warn);margin:4px 0 0;padding-left:16px}
[class*="st-key-card_"]{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px 10px}
[class*="st-key-card_"][class*="_approve"]{border-left:3px solid var(--ok)}
[class*="st-key-card_"][class*="_reject"]{opacity:.62}
[class*="st-key-prev_"]{background:transparent;border:1px dashed var(--line);border-radius:10px;padding:10px 14px}
.pf-run{font-size:13px;color:var(--mute)}
.pf-run b{color:var(--ink);font-weight:600}
.stTabs [data-baseweb="tab"]{font-size:14px}
[data-testid="stSidebar"] h3{font-size:19px!important;margin-top:.4rem}
</style>""")


# ============================================================================= helpers
def secret(name: str) -> str:
    try:
        if name and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "") if name else ""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


@contextmanager
def materialized(source: dict):
    """Yield a folder holding the scripts. Uploads are written to a temp folder that is deleted on exit."""
    if source["kind"] in ("sample", "local"):
        yield Path(source["path"])
        return
    tmp = Path(tempfile.mkdtemp(prefix="pf_upload_"))
    try:
        for name, data in source["files"]:
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for info in z.infolist():
                        base = Path(info.filename).name          # basename only: no path traversal
                        if info.is_dir() or base.startswith(("._", ".")) or Path(base).suffix.lower() not in (".txt", ".md"):
                            continue
                        target = tmp / base
                        if target.exists():
                            target = tmp / f"{len(list(tmp.iterdir())):03d}_{base}"
                        target.write_bytes(z.read(info))
            else:
                (tmp / Path(name).name).write_bytes(data)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def friendly(e: BaseException, provider: str) -> tuple[str, str]:
    p = PROVIDER_LABELS.get(provider, provider)
    if isinstance(e, LLMError):
        return {
            "auth": (f"{p} rejected the API key.", "Check the key in the sidebar. Keys are kept in memory only."),
            "rate_limit": (f"{p} kept rate-limiting the run.", "Every finished call is cached. Wait a minute and run again; it continues where it stopped."),
            "unavailable": (f"{p} is unreachable or overloaded.", "Every finished call is cached. Try again shortly, or choose 'Rules only'."),
            "bad_request": (f"{p} rejected the request.", "Check the model id and base URL in the sidebar."),
            "refusal": ("The model declined this request.", "Try another model, or run rules only."),
            "parse": ("The model's replies could not be read as JSON.", "Try a stronger model. The smoke test script checks a provider in one call."),
            "config": ("Configuration problem.", str(e)),
        }.get(e.kind, ("The model call failed.", str(e)[:200]))
    if isinstance(e, Cancelled):
        return ("Run cancelled.", "Finished calls are cached, so running again continues where it stopped.")
    if isinstance(e, SystemExit):
        return ("Nothing to run.", str(e))
    return ("Something went wrong.", f"{type(e).__name__}: {str(e)[:200]}")


def list_runs() -> list[Path]:
    dirs = [p.parent for p in BASE.outputs_dir.rglob("promos.json") if "ablation" not in p.parts]
    return sorted(dirs, key=lambda d: (d / "promos.json").stat().st_mtime, reverse=True)[:40]


def run_label(d: Path) -> str:
    try:
        m = json.loads((d / "run_meta.json").read_text())
        when = time.strftime("%d %b %H:%M", time.localtime((d / "promos.json").stat().st_mtime))
        return f"{m.get('show', d.name)[:28]} · {m.get('model', '?').split(':')[-1]} · {m.get('scoring_preset', 'v4')} · {when}"
    except Exception:
        return d.name


def load_run(d: Path) -> dict:
    data = json.loads((d / "promos.json").read_text())
    fb = d / "feedback.jsonl"
    verdicts = {}
    if fb.exists():
        for line in fb.read_text().splitlines():
            try:
                r = json.loads(line)
                verdicts[r["id"]] = r["verdict"]
            except Exception:
                pass
    return {"meta": data["meta"], "rows": data["promos"], "ctx": data.get("needs_context", []), "out": str(d),
            "verdicts": verdicts}


def set_verdict(R: dict, row: dict, verdict: str):
    R["verdicts"][row["id"]] = verdict
    with open(Path(R["out"]) / "feedback.jsonl", "a") as f:
        f.write(json.dumps({"id": row["id"], "verdict": verdict, "rank": row["rank"], "list": row.get("list", "main"),
                            "segments": [{k: s[k] for k in ("episode", "start", "end")} for s in row["segments"]],
                            "score": row["score"], "components": row["component_scores"], "ts": time.time()}) + "\n")


def stats_html(items: list[tuple[str, str]]) -> str:
    return '<div class="pf-stats">' + "".join(f'<div class="pf-stat"><div class="v">{esc(v)}</div><div class="k">{esc(k)}</div></div>'
                                             for k, v in items) + "</div>"


def meter(label: str, v: float | None) -> str:
    if v is None:
        return ""
    return f'<span class="pf-meter">{esc(label)} <i><s style="width:{max(0, min(10, v)) * 10:.0f}%"></s></i> {v:.0f}</span>'


CHECK_LABELS = {"length_20_60s": "20–60 s", "clean_cuts_no_mid_line": "clean cuts", "hook_in_first_3s": "hook in 3 s",
                "ends_unresolved": "ends unresolved", "has_dialogue": "has dialogue", "cold_viewer_understands": "stranger follows"}


def card(R: dict, r: dict, compact: bool = False):
    verdict = R["verdicts"].get(r["id"], "none") if R else "none"
    key = f"{'prev' if compact else 'card'}_{Path(R['out']).name if R else 'preview'}_{r['id']}_{verdict}"
    with st.container(key=key):
        segs = "".join(f'<span class="pf-tc">EP {s["episode"]:02d} · {esc(s["start"])} → {esc(s["end"])}</span>' for s in r["segments"])
        kind = "two-moment trailer" if r["kind"] == "stitched" else "continuous"
        mark = {"approve": " · approved", "reject": " · rejected"}.get(verdict, "")
        head = (f'<div class="pf-head"><div class="pf-rank">{r["rank"]:02d}</div><div><div>{segs}</div>'
                f'<div class="pf-meta">{r["duration_sec"]} s · {kind} · score {r["score"]:.2f}{mark}</div></div></div>')
        if compact:
            st.html(head + f'<dl class="pf-dl"><dt>Hook</dt><dd>{esc(r["hook"])}</dd></dl>')
            return
        cv = r.get("cold_viewer") or {}
        meters = "".join([meter("Hook", cv.get("hook")), meter("Cliffhanger", cv.get("cliffhanger")),
                          meter("Makes sense alone", cv.get("standalone"))])
        badges = "".join(f'<span class="pf-b{"" if ok else " fail"}">{"✓" if ok else "✗"} {esc(CHECK_LABELS.get(k, k))}</span>'
                         for k, ok in r["checks"].items())
        flags = "".join(f"<li>{esc(f)}</li>" for f in r["needs_human_review"])
        st.html(head + f'<dl class="pf-dl"><dt>Hook</dt><dd>{esc(r["hook"])}</dd><dt>Cliffhanger</dt>'
                f'<dd>{esc(r["cliffhanger"])}</dd><dt>Why</dt><dd>{esc(r["why_it_works"])}</dd></dl>'
                + (f'<div class="pf-meters">{meters}</div>' if meters else "")
                + f'<div class="pf-badges">{badges}</div>' + (f'<ul class="pf-flags">{flags}</ul>' if flags else ""))
        with st.expander("Script for this promo"):
            for s in r["segments"]:
                st.code("\n".join(s["script"]), language=None)
        a, b, _ = st.columns([1, 1, 4])
        a.button("Approve", key=f"ap_{key}", on_click=set_verdict, args=(R, r, "approve"),
                 type="primary" if verdict == "approve" else "secondary", width="stretch")
        b.button("Reject", key=f"rj_{key}", on_click=set_verdict, args=(R, r, "reject"), width="stretch")


# ============================================================================= sidebar
ss = st.session_state
with st.sidebar:
    st.markdown("### Model")
    names = list(PROVIDER_LABELS)
    default = "gemini" if secret("GEMINI_API_KEY") else "none"
    preset = st.selectbox("Provider", names, index=names.index(default), format_func=PROVIDER_LABELS.get)
    prov, d_model, d_base, env = PRESETS[preset]
    lim = C.provider_limits(BASE, preset)
    model = st.text_input("Model", d_model) if preset != "none" else ""
    base_url = d_base
    if prov == "openai_compat" and not PUBLIC:      # public: a custom URL could receive the server's key
        with st.expander("Endpoint"):
            base_url = st.text_input("Base URL (any OpenAI-compatible API)", d_base)
    key = ""
    if env:
        server_key = secret(env)
        typed = st.text_input("API key", type="password", help="Kept in this browser session only, never saved.",
                              placeholder="using the key configured on this machine" if server_key else f"paste {env}")
        key = typed or server_key
    workers = st.number_input("Parallel calls", 1, 32, int(lim["workers"]),
                              help=f"Requests are also capped at {lim['rpm']} per minute for this provider.") if preset != "none" else 1

    st.markdown("### Scope")
    presets = C.scoring_presets(BASE)
    main_presets = [p for p in presets if p in PRESET_LABELS] + [p for p in presets if p not in PRESET_LABELS]
    scoring = st.selectbox("Scoring", main_presets, index=main_presets.index(BASE.scoring_preset) if BASE.scoring_preset in main_presets else 0,
                           format_func=lambda p: PRESET_LABELS.get(p, p),
                           help="v4 = the submitted pipeline. v6 = v4 + show profile (works on any show) + needs-context list + "
                                "stitched safeguards. v5 = every change tried (several hurt on the dev set: see DECISIONS.md).")
    ep_spec = st.text_input("Episodes", "", placeholder="all  (e.g. 1-10 or 26-50)")
    top_k = st.slider("Promos to return", 5, 40, BASE.settings.top_k)

    if PUBLIC:
        st.caption("Public demo. Uploaded scripts are deleted after each step; results are kept on this server "
                   "but never listed to other visitors.")
    else:
        st.markdown("### Data")
    runs = [] if PUBLIC else list_runs()
    if runs:
        pick = st.selectbox("Previous runs", runs, format_func=run_label, index=None, placeholder="Open a saved run…")
        if pick is not None and st.button("Open run", width="stretch"):
            ss["res"] = load_run(pick)
            ss["view"] = "review"
    if not PUBLIC:
        with st.popover("Clear cache", width="stretch"):
            st.caption("Deletes saved model answers, so the next run calls the model again. "
                       "The read-only v4 snapshot is kept, so v4 stays reproducible.")
            if st.button("Delete cached answers", type="primary"):
                shutil.rmtree(BASE.cache_dir, ignore_errors=True)
                BASE.cache_dir.mkdir(parents=True, exist_ok=True)
                ss.pop("est", None)
                st.toast("Cache cleared")

# ============================================================================= header
try:
    episodes_filter = parse_episode_range(ep_spec)
    range_error = None
except ValueError as e:
    episodes_filter, range_error = None, str(e)

has_res = "res" in ss
source = ss.get("source")
step = 4 if has_res and ss.get("view") == "review" else (2 if source else 1)
chips = [PROVIDER_LABELS[preset] + (f" · {model}" if model else ""), PRESET_LABELS.get(scoring, scoring)]
st.html(f"""<div class="pf-top"><div><div class="pf-word">Promo <em>Finder</em></div>
<div class="pf-sub">Ranked 20–60 s promos from timestamped microdrama scripts. Every cut lands on a shot change, never mid-line.</div></div>
<div class="pf-chips">{''.join(f'<span class="pf-chip">{esc(c)}</span>' for c in chips)}</div></div>
<div class="pf-steps">{''.join(f'<div class="pf-step {"done" if i < step else "on" if i == step else ""}"><b>0{i}</b>{n}</div>'
                               for i, n in enumerate(["Load scripts", "Check", "Run", "Review"], 1))}</div>""")

# ============================================================================= 1. load
mode = st.segmented_control("Source", ["Files or .zip", "Folder", "Sample show"], default="Files or .zip",
                            label_visibility="collapsed") or "Files or .zip"
if mode == "Folder":
    files = st.file_uploader("Choose the show's folder", accept_multiple_files="directory", type=["txt", "md"])
elif mode == "Files or .zip":
    files = st.file_uploader("One screenplay per episode (.txt or .md), or a .zip of the show folder",
                             accept_multiple_files=True, type=["txt", "md", "zip"])
else:
    files = None
    st.caption("An original 10-episode drama written for testing (not Kuku content). Use it to rehearse the full flow.")
local = ""
if not PUBLIC:                                       # public: never read folders on the server
    with st.expander("Use a folder on this computer"):
        local = st.text_input("Folder path", "", label_visibility="collapsed", placeholder="/path/to/show-folder")

if mode == "Sample show":
    source = {"kind": "sample", "path": str(SAMPLE), "sig": ("sample",), "name": "The CEO's Hidden Wife"}
elif files:
    blobs = [(f.name, f.getvalue()) for f in files]
    zips = [n for n, _ in blobs if n.lower().endswith(".zip")]
    source = {"kind": "upload", "files": blobs, "sig": tuple((n, len(b)) for n, b in blobs),
              "name": clean_show_name(Path(zips[0]).stem) if zips else ""}
elif local.strip():
    p = Path(local.strip()).expanduser()
    source = {"kind": "local", "path": str(p), "sig": ("local", str(p)), "name": clean_show_name(p)} if p.exists() else None
    if not p.exists():
        st.error("That folder doesn't exist.")
else:
    source = None
ss["source"] = source

# ============================================================================= 2. check
if source is not None:
    if ss.get("parsed", {}).get("sig") != source["sig"]:
        with materialized(source) as folder:
            eps, rep = parse_show(folder, settings=BASE.settings)
        ss["parsed"] = {"sig": source["sig"], "episodes": [{"n": e.number, "file": Path(e.path).name, "dur": e.duration,
                                                           "beats": len(e.beats), "logline": e.logline} for e in eps],
                        "summary": rep.summary(), "warnings": rep.all_warnings, "unparsed": rep.unparsed}
        ss.pop("est", None)
    P = ss["parsed"]
    eps_meta = [e for e in P["episodes"] if not episodes_filter or e["n"] in episodes_filter]
    show_name = st.text_input("Show title", source["name"], key=f"title_{source['sig']}",
                              help="Given to the model as context (titles often hint at the premise). Keep it the same "
                                   "between runs so cached answers are reused.")
    st.markdown('<div class="pf-label">Check · before anything is spent</div>', unsafe_allow_html=True)
    s = P["summary"]
    if not P["episodes"]:
        st.html('<div class="pf-note bad">No timestamped beats found. Each beat needs a header such as '
                '<code>**NAME (Tone):**</code> followed by <code>Start: (00:11.30) | End: (00:13.10)</code>.</div>')
        st.stop()
    st.html(stats_html([("episodes" + (f" (of {len(P['episodes'])})" if episodes_filter else ""), str(len(eps_meta))),
                        ("beats", f"{sum(e['beats'] for e in eps_meta):,}"),
                        ("minutes of story", f"{sum(e['dur'] for e in eps_meta) / 60:.1f}"),
                        ("timestamp lines read", f"{s['coverage']:.1%}")]))
    force = False
    if range_error:
        st.html(f'<div class="pf-note bad">Episode range: {esc(range_error)}</div>')
    if episodes_filter and not eps_meta:
        st.html('<div class="pf-note bad">No episodes in that range.</div>')
    if s["coverage"] < BASE.settings.min_coverage:
        st.html(f'<div class="pf-note bad"><b>{s["unparsed"]} timestamp line(s) could not be read</b> '
                f'({s["parsed_lines"]}/{s["timestamp_lines"]}). Promos from those beats would be missing. '
                f'Fix the format, or run anyway.</div>')
        force = st.checkbox("Run anyway")
    for w in P["warnings"][:8]:
        st.html(f'<div class="pf-note">{esc(w)}</div>')
    with st.expander(f"Episodes detected ({len(P['episodes'])})" + (f" · {len(P['unparsed'])} unread lines" if P["unparsed"] else "")):
        st.dataframe([{"ep": e["n"], "file": e["file"], "length": fmt(e["dur"]), "beats": e["beats"],
                       "logline": e["logline"][:120]} for e in P["episodes"]], hide_index=True, width="stretch")
        if P["unparsed"]:
            st.dataframe([{"file": f, "line": n, "text": t} for f, n, t in P["unparsed"][:200]], hide_index=True, width="stretch")

    # ------------------------------------------------------------------------- 3. run
    cfg = C.load(preset=scoring, overrides={"top_k": top_k})
    llm = C.make_llm(cfg, preset, model, base_url, api_key=key, workers=int(workers))
    ready = bool(eps_meta) and not range_error and (s["coverage"] >= BASE.settings.min_coverage or force)
    est = None
    if llm.enabled and not llm.has_key():
        st.html(f'<div class="pf-note bad">Add an API key for {esc(PROVIDER_LABELS[preset])} in the sidebar, '
                f'or choose “Rules only”.</div>')
        ready = False
    elif llm.enabled and ready:
        ekey = (source["sig"], preset, model, base_url, scoring, ep_spec, show_name, top_k)
        if ss.get("est", {}).get("key") != ekey:
            with materialized(source) as folder:
                ss["est"] = {"key": ekey, "v": estimate(str(folder), llm, show_name or None, cfg.settings,
                                                         episodes_filter, force=True)}
        est = ss["est"]["v"]
    if est is not None:
        mins = max(est["calls"] / max(llm.rpm, 0.1), 0.0)
        cost = f" · about ${est['est_cost_usd']:.2f}" if est["est_cost_usd"] else (" · free tier / price not set" if est["calls"] else "")
        msg = ("Everything is cached: this run makes <b>no new model calls</b>." if not est["calls"] else
               f"This run makes <b>{est['calls']}{'+' if est['upper_bound'] else ''} new model calls</b> "
               f"(~{(est['in_tokens'] + est['out_tokens']) / 1000:.0f}k tokens) · at least {mins:.0f} min at "
               f"{llm.rpm:g} requests/min{cost}.")
        st.html(f'<div class="pf-note {"ok" if not est["calls"] else ""}">{msg}</div>')
    if ss.get("run_state") == "cancelled":
        st.html('<div class="pf-note">The last run was cancelled. Finished calls are cached, so running again continues where it stopped.</div>')
        ss["run_state"] = None

    go = st.button("Find promos", type="primary", disabled=not ready)
    if go:
        out_dir = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}__{(show_name or 'show')[:24].replace('/', '-').replace(' ', '-')}"
        llm.log_path = str(out_dir / "llm_calls.jsonl")
        ss["run_state"] = "running"
        st.button("Cancel run", key="cancel_btn")          # any click interrupts the run (Streamlit rerun)
        status = st.empty()
        bar = st.progress(0.0)
        preview = st.empty()

        def on_progress(ev: dict):
            frac = ev["done"] / ev["total"] if ev["total"] else 0.0
            bar.progress(min(1.0, frac), text=f"{STAGE_LABELS.get(ev['stage'], ev['stage'])}"
                                                 + (f" · {ev['done']}/{ev['total']}" if ev["total"] else ""))
            cost = f" · <b>${ev['est_cost']:.3f}</b>" if ev["est_cost"] else ""
            status.html(f'<div class="pf-run">calls <b>{ev["calls"]}</b> new + <b>{ev["cache_hits"]}</b> cached · '
                        f'<b>{(ev["tokens_in"] + ev["tokens_out"]) / 1000:.1f}k</b> tokens · <b>{ev["elapsed"]:.0f}s</b>'
                        f'{cost}{" · retries " + str(ev["retries"]) if ev["retries"] else ""}</div>')

        def on_rules(r: dict):
            with preview.container():
                st.markdown('<div class="pf-label">Rules-only preview · the AI ranking replaces it when ready</div>',
                            unsafe_allow_html=True)
                for p in r["promos"][:5]:
                    card(None, promo_dict(p, r["ep_by_num"]), compact=True)

        try:
            with materialized(source) as folder:
                res = run(str(folder), llm, show_name or None, cfg.settings, episodes_filter, progress=on_progress,
                          log=lambda *_: None, on_rules_result=on_rules if llm.enabled else None, force=force,
                          workers=int(workers),
                          extra_meta={"scoring_preset": scoring, "source": source["kind"],
                                      "heldout_sha256": file_sha256(BASE.heldout_path)})
                write_all(res, out_dir)
        except ParseCoverageError as e:
            ss["run_state"] = None
            st.html(f'<div class="pf-note bad">Only {e.report.coverage:.0%} of timestamp lines could be read. Tick “Run anyway” to continue.</div>')
            st.stop()
        except (LLMError, SystemExit, Exception) as e:      # readable message, never a stack trace
            ss["run_state"] = None
            title, detail = friendly(e, preset)
            st.html(f'<div class="pf-note bad"><b>{esc(title)}</b> {esc(detail)}</div>')
            st.stop()
        except BaseException:                                # Streamlit interrupt = the user pressed Cancel
            ss["run_state"] = "cancelled"
            raise
        ss["run_state"] = "done"
        ss["res"] = load_run(out_dir)
        ss["view"] = "review"
        preview.empty()
        st.rerun()
elif not has_res:
    st.html('<div class="pf-note">Upload one screenplay per episode (or a .zip), choose a folder, or open the sample '
            'show. Nothing is sent anywhere until you press <b>Find promos</b>, and you will see the number of model '
            'calls first.</div>')

# ============================================================================= 4. review
if has_res:
    R = ss["res"]
    m, out = R["meta"], Path(R["out"])
    st.markdown(f"## {esc(m.get('show', 'Results'))}")
    u = m.get("llm_usage", {})
    reviewed = len(R["verdicts"])
    approved = sum(v == "approve" for v in R["verdicts"].values())
    st.html(stats_html([("promos", str(len(R["rows"]))), ("need context", str(len(R["ctx"]))),
                        ("episodes", str(m.get("episodes", "?"))),
                        ("model calls · cached", f"{u.get('calls', 0)} · {u.get('cache_hits', 0)}"),
                        ("run time", f"{m.get('seconds', 0):.0f} s"),
                        ("approved", f"{approved}/{reviewed}" if reviewed else "–")]))
    st.caption(f"{m.get('model', '?')} · scoring {m.get('scoring_preset', 'v4')} · "
               + " · ".join(f"{CHECK_LABELS.get(k, k)} {v * 100:.0f}%" for k, v in m.get("check_pass_rates", {}).items()))
    tabs = st.tabs([f"Main ranking  {len(R['rows'])}", f"Needs context  {len(R['ctx'])}", "Report", "Table",
                    "Run details", "Downloads"])
    with tabs[0]:
        if not R["rows"]:
            st.info("No promos passed the checks.")
        for r in R["rows"]:
            card(R, r)
    with tabs[1]:
        st.caption("Strong clips that a stranger may not follow on their own. Good for retargeting viewers who already "
                   "know the show, or as a starting point an editor can extend with a few seconds of setup.")
        if not R["ctx"]:
            st.info("None. This list is used by the v5 'gate' scoring.")
        for r in R["ctx"]:
            card(R, r)
    with tabs[2]:
        rep = out / "report.html"
        if rep.exists():   # data: URL -> the report renders in its own sandboxed origin
            st.iframe(f"data:text/html;base64,{base64.b64encode(rep.read_bytes()).decode()}", height=1100)
    with tabs[3]:
        st.dataframe([{"#": r["rank"], "segments": " + ".join(f"Ep{s['episode']:02d} {s['start']}-{s['end']}" for s in r["segments"]),
                       "sec": r["duration_sec"], "score": r["score"], "hook": r["hook"], "cliffhanger": r["cliffhanger"],
                       "why": r["why_it_works"], "verdict": R["verdicts"].get(r["id"], "")} for r in R["rows"]],
                     hide_index=True, width="stretch")
    with tabs[4]:
        a, b = st.columns(2)
        with a:
            st.markdown('<div class="pf-label">Stage timings (s)</div>', unsafe_allow_html=True)
            st.dataframe([{"stage": k, "seconds": v} for k, v in m.get("stage_seconds", {}).items()], hide_index=True, width="stretch")
            st.markdown('<div class="pf-label">Model usage</div>', unsafe_allow_html=True)
            st.json({k: m.get(k) for k in ("model", "workers", "llm_usage", "miner", "judge", "rerank")}, expanded=False)
        with b:
            st.markdown('<div class="pf-label">Parse</div>', unsafe_allow_html=True)
            st.json(m.get("parse", {}), expanded=False)
            prof = out / "show_profile.json"
            if prof.exists():
                st.markdown('<div class="pf-label">Show profile</div>', unsafe_allow_html=True)
                st.json(json.loads(prof.read_text()), expanded=False)
            st.markdown('<div class="pf-label">Prompt versions</div>', unsafe_allow_html=True)
            st.json(m.get("prompt_versions", {}), expanded=False)
    with tabs[5]:
        files_ = [f for f in ("promos.csv", "promos.json", "run_meta.json", "llm_calls.jsonl", "show_profile.json",
                              "parse_report.json", "report.html", "feedback.jsonl") if (out / f).exists()]
        cols = st.columns(4)
        for i, f in enumerate(files_):
            cols[i % 4].download_button(f, (out / f).read_bytes(), f, width="stretch")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in out.iterdir():
                if p.is_file():
                    z.write(p, p.name)
        st.download_button("Everything (.zip)", buf.getvalue(), f"{out.name}.zip", type="primary")
