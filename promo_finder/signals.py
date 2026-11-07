"""Rule-based beat signals: cheap, explainable, and free to run on every beat.

Simple keyword/tone rules (lists live in lexicon.toml). They do two jobs:
  1. A no-API baseline that always works (rules-only mode).
  2. A pre-filter so the LLM only judges a short list (cost control at scale).
Every score is in [0, 1] and comes with the matched words, so an editor can see why.
"""
from __future__ import annotations

import re
import tomllib
from functools import lru_cache
from pathlib import Path

from .parse import Beat

LEXICON_PATH = Path(__file__).with_name("lexicon.toml")
PRIVATE_LEXICON = Path(__file__).with_name("lexicon_v4_legacy.private.toml")   # gitignored


@lru_cache(maxsize=None)
def _lexicon_file() -> dict:
    lx = tomllib.loads(LEXICON_PATH.read_text())
    if PRIVATE_LEXICON.exists():
        lx.update(tomllib.loads(PRIVATE_LEXICON.read_text()))
    return lx


def resolve_lexicon(name: str) -> str:
    """The lexicon actually used: v4_legacy falls back to generic when its private file is absent."""
    return name if name in _lexicon_file() else "generic"


def lexicon(name: str = "v4_legacy", show_words: dict | None = None) -> dict:
    """Merge shared lists with the named list, plus show-specific words from the show profile."""
    lx = _lexicon_file()
    name = resolve_lexicon(name) if name == "v4_legacy" else name
    if name not in lx:
        raise SystemExit(f"Unknown lexicon '{name}' (have: {', '.join(k for k in lx if k != 'shared')})")
    shared, chosen = lx["shared"], lx[name]
    out = {
        "physical": [tuple(x) for x in chosen["physical"]],
        "reveal": [tuple(x) for x in chosen["reveal"]],
        "shock_line": [tuple(x) for x in chosen["shock_line"]],
        "hot": dict(shared["hot_emotions"]),
        "calm": set(shared["calm_emotions"]),
        "camera": shared["intense_camera"],
        "cliff": [tuple(x) for x in shared["cliff_text"]],
        "resolution": shared["resolution_text"],
        "threat": shared["threat_line"],
    }
    for kind, weight in (("reveal_words", 0.9), ("threat_words", 0.8)):
        words = [w.strip() for w in (show_words or {}).get(kind, []) if isinstance(w, str) and 2 < len(w.strip()) < 40]
        if words:
            pat = r"\b(" + "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True)) + r")\b"
            out["reveal" if kind == "reveal_words" else "shock_line"].append((weight, pat))
    return out


def _lex(text: str, table) -> tuple[float, list[str]]:
    best, hits = 0.0, []
    for w, pat in table:
        if m := re.search(pat, text, re.I):
            hits.append(m.group(0).lower())
            best = max(best, w)
    return best, hits


def _emotion_heat(beat: Beat, lx: dict) -> float:
    tags = [t.strip().lower() for t in beat.emotion.split(",") if t.strip()]
    return max((lx["hot"].get(t, 0.0) for t in tags), default=0.0)


def _is_calm(beat: Beat, lx: dict) -> bool:
    return bool({t.strip().lower() for t in beat.emotion.split(",")} & lx["calm"])


def beat_signals(beat: Beat, lx: dict, resolve_on_calm_tone: bool = True) -> dict:
    """Per-beat scores + human-readable reasons."""
    text = beat.text
    phys, ph = _lex(text, lx["physical"])
    rev, rh = _lex(text, lx["reveal"])
    reasons = ph + rh
    hook = max(phys, rev * 0.9)
    intensity = max(phys, rev * 0.7)

    if beat.kind == "DIALOGUE":
        line, lh = _lex(text, lx["shock_line"])
        heat = _emotion_heat(beat, lx)
        reasons += lh + ([f"tone:{beat.emotion.lower()}"] if heat >= 0.7 else [])
        punct = 0.25 if re.search(r"[!?]", text) else 0.0
        hook = max(hook, 0.55 * line + 0.35 * heat + punct, rev)
        intensity = max(intensity, 0.6 * heat + 0.4 * line)
    elif beat.kind == "CAMERA":
        if re.search(lx["camera"], text, re.I):
            hook = max(hook * 1.1, 0.35)
            reasons.append("intense-camera")
    elif beat.kind == "SFX":
        hook = max(hook, phys)

    cliff, ch = _lex(text, lx["cliff"])
    if beat.kind == "DIALOGUE":
        if text.rstrip().endswith(("?", "—", "-", "...", "…")):
            cliff = max(cliff, 0.7)
            ch.append("open-question/interrupted")
        if re.search(lx["threat"], text, re.I):
            cliff = max(cliff, 0.8)
            ch.append("threat/ultimatum")
        cliff = max(cliff, 0.6 * _emotion_heat(beat, lx))
    cliff = max(cliff, 0.6 * rev)

    resolve = 0.0
    calm_tone = resolve_on_calm_tone and beat.kind == "DIALOGUE" and _is_calm(beat, lx)
    if re.search(lx["resolution"], text, re.I) or calm_tone:
        resolve = 0.8
    return {"hook": min(hook, 1.0), "cliff": min(cliff, 1.0), "resolve": resolve,
            "intensity": min(intensity, 1.0), "hook_reasons": reasons, "cliff_reasons": ch}


def annotate(episodes, lexicon_name: str = "v4_legacy", show_words: dict | None = None,
             resolve_on_calm_tone: bool = True) -> dict[str, dict]:
    """Signals for every beat in the show, keyed by beat id."""
    lx = lexicon(lexicon_name, show_words)
    return {b.id: beat_signals(b, lx, resolve_on_calm_tone) for ep in episodes for b in ep.beats}
