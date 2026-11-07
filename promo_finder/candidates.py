"""Generate promo candidates with code (cheap, exhaustive, timestamp-exact).

Two shapes:
  * continuous - one stretch inside an episode, starting on a shot cut and ending on a shot cut.
  * stitched   - an "opener" moment + a "closer" moment from different points in the story
                 (e.g. Ep1 humiliation -> Ep3 proposal), the classic microdrama trailer shape.

Hard rules enforced here (never left to the model):
  * total length 20-60s, cuts only on shot boundaries, never in the middle of a spoken line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .parse import Beat, Episode, fmt


@dataclass
class Segment:
    episode: int
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start

    def label(self) -> str:
        return f"Ep{self.episode:02d} {fmt(self.start)}-{fmt(self.end)}"


@dataclass
class Promo:
    id: str
    kind: str                       # continuous | stitched
    segments: list[Segment]
    hook_beat: str = ""
    cliff_beat: str = ""
    rule: dict = field(default_factory=dict)    # rule-based component scores
    llm: dict = field(default_factory=dict)     # cold-viewer judge output
    final: float = 0.0
    checks: dict = field(default_factory=dict)
    hook_text: str = ""
    cliff_text: str = ""
    why: str = ""
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.dur for s in self.segments)

    @property
    def first_episode(self) -> int:
        return self.segments[0].episode


# ---------------------------------------------------------------- cut points
def _inside_dialogue(ep: Episode, t: float, tol: float = 0.15) -> bool:
    return any(b.kind == "DIALOGUE" and b.start < t - tol and b.end > t + tol for b in ep.beats)


def cut_points(ep: Episode) -> tuple[list[float], list[float]]:
    """Shot boundaries where an editor can cleanly cut in / cut out."""
    starts = {0.0} | {b.start for b in ep.beats if b.kind == "CAMERA"}
    ends = {ep.duration} | {b.end for b in ep.beats if b.kind == "CAMERA"}
    # don't open on the black frame or close after it
    black = [b.start for b in ep.beats if b.kind == "CAMERA" and "cut to black" in b.text.lower()]
    if black:
        ends = {e for e in ends if e <= min(black) + 0.01} | {min(black)}
    ins = sorted(t for t in starts if not _inside_dialogue(ep, t) and t < ep.duration - 5)
    outs = sorted(t for t in ends if not _inside_dialogue(ep, t))
    return ins, outs


# ---------------------------------------------------------------- window scoring
V4 = Settings()


def window_parts(ep: Episode, sig: dict, s: float, e: float, S: Settings = V4) -> dict:
    beats = ep.beats_between(s, e)
    if not beats:
        return {}
    # A hook must START inside the first 3 s. A beat already in progress at the cut-in doesn't count
    # (bug B2: a clip cut ~1 s after a strike was credited with the strike it never shows).
    opening = [b for b in beats if s - S.hook_lead <= b.start < s + S.hook_window] or beats[:1]
    closing = [b for b in beats if b.end > e - S.cliff_window]
    hook_b = max(opening, key=lambda b: sig[b.id]["hook"])
    cliff_b = max(closing, key=lambda b: sig[b.id]["cliff"])
    hook = sig[hook_b.id]["hook"]
    cliff = sig[cliff_b.id]["cliff"]
    if e >= ep.duration - 2.0 or any("cut to black" in b.text.lower() for b in ep.beats if abs(b.start - e) < 1.2):
        cliff = min(1.0, cliff + S.end_of_episode_bonus)    # writers built this as the episode cliffhanger
    resolve = max(sig[b.id]["resolve"] for b in closing)
    ints = sorted((sig[b.id]["intensity"] for b in beats), reverse=True)
    intensity = sum(ints[: max(1, len(ints) // 2)]) / max(1, len(ints) // 2)
    speakers = {b.speaker for b in beats if b.kind == "DIALOGUE"}
    n_lines = sum(b.kind == "DIALOGUE" for b in beats)
    scenes = {b.scene for b in beats}
    # crude standalone proxy: needs some dialogue, few characters, one or two locations
    standalone = 1.0
    if n_lines < 3:
        standalone -= 0.35
    if len(speakers) > 4:
        standalone -= 0.15 * (len(speakers) - 4)
    if len(scenes) > 2:
        standalone -= 0.2
    return {"hook": hook, "cliff": max(0.0, cliff - S.resolve_penalty * resolve), "intensity": intensity,
            "standalone": max(0.0, standalone), "resolve": resolve,
            "hook_beat": hook_b.id, "cliff_beat": cliff_b.id}


def rule_score(p: dict, dur: float) -> float:
    sweet = 1.0 if 28 <= dur <= 50 else 0.85
    return sweet * (0.35 * p["hook"] + 0.35 * p["cliff"] + 0.2 * p["intensity"] + 0.1 * p["standalone"])


def _overlap(a: Promo, b: Promo) -> float:
    """Fraction of the shorter promo that overlaps the other one in story time."""
    inter = 0.0
    for x in a.segments:
        for y in b.segments:
            if x.episode == y.episode:
                inter += max(0.0, min(x.end, y.end) - max(x.start, y.start))
    return inter / max(1e-6, min(a.duration, b.duration))


def nms(promos: list[Promo], key, max_overlap: float = 0.4, limit: int | None = None) -> list[Promo]:
    """Greedy non-max suppression so we don't return 10 near-identical clips."""
    kept: list[Promo] = []
    for p in sorted(promos, key=key, reverse=True):
        if all(_overlap(p, k) <= max_overlap and p.hook_beat != k.hook_beat for k in kept):
            kept.append(p)
            if limit and len(kept) >= limit:
                break
    return kept


# ---------------------------------------------------------------- generators
def continuous_candidates(ep: Episode, sig: dict, S: Settings = V4) -> list[Promo]:
    ins, outs = cut_points(ep)
    out: list[Promo] = []
    for s in ins:
        for e in outs:
            d = e - s
            if d < S.min_len or d > S.max_len:
                continue
            parts = window_parts(ep, sig, s, e, S)
            if not parts:
                continue
            p = Promo(id="", kind="continuous", segments=[Segment(ep.number, s, e)],
                      hook_beat=parts.pop("hook_beat"), cliff_beat=parts.pop("cliff_beat"))
            p.rule = parts
            p.rule["score"] = rule_score(parts, d)
            out.append(p)
    return nms(out, key=lambda p: p.rule["score"], max_overlap=S.candidate_overlap, limit=S.per_episode)


def _moments(ep: Episode, sig: dict, role: str, S: Settings, min_len=6.0, max_len=22.0, per_episode=2) -> list[Promo]:
    """Short sub-clips that work as an opener (strong hook) or closer (strong cliffhanger)."""
    ins, outs = cut_points(ep)
    found = []
    for s in ins:
        for e in outs:
            if not (min_len <= e - s <= max_len):
                continue
            parts = window_parts(ep, sig, s, e, S)
            if not parts:
                continue
            p = Promo(id="", kind="moment", segments=[Segment(ep.number, s, e)],
                      hook_beat=parts.pop("hook_beat"), cliff_beat=parts.pop("cliff_beat"))
            p.rule = parts
            p.rule["score"] = (0.7 * parts["hook"] + 0.3 * parts["intensity"]) if role == "open" \
                else (0.7 * parts["cliff"] + 0.3 * parts["intensity"])
            found.append(p)
    return nms(found, key=lambda p: p.rule["score"], limit=per_episode)


def stitched_candidates(episodes: list[Episode], sig: dict, S: Settings = V4) -> list[Promo]:
    if not S.stitched_enabled:
        return []
    openers, closers = [], []
    for ep in episodes:
        openers += _moments(ep, sig, "open", S)
        closers += _moments(ep, sig, "close", S)
    openers = sorted(openers, key=lambda p: p.rule["score"], reverse=True)[:S.stitched_top_moments]
    closers = sorted(closers, key=lambda p: p.rule["score"], reverse=True)[:S.stitched_top_moments]
    out = []
    for o in openers:
        for c in closers:
            so, sc = o.segments[0], c.segments[0]
            if sc.episode < so.episode or sc.episode - so.episode > S.stitched_max_episode_gap:
                continue
            if sc.episode == so.episode and sc.start < so.end + 15:   # that's just a continuous clip
                continue
            d = so.dur + sc.dur
            if not (S.min_len <= d <= S.max_len):
                continue
            standalone = min(o.rule["standalone"], c.rule["standalone"]) - 0.1   # every cut costs clarity
            parts = {"hook": o.rule["hook"], "cliff": c.rule["cliff"],
                     "intensity": (o.rule["intensity"] + c.rule["intensity"]) / 2,
                     "standalone": max(0.0, standalone), "resolve": c.rule["resolve"]}
            p = Promo(id="", kind="stitched", segments=[so, sc], hook_beat=o.hook_beat, cliff_beat=c.cliff_beat)
            p.rule = parts
            p.rule["score"] = rule_score(parts, d) * S.stitched_penalty
            p.flags.append("cross-episode cut: check visual continuity (costume/location)")
            out.append(p)
    return nms(out, key=lambda p: p.rule["score"], limit=S.stitched_limit)
