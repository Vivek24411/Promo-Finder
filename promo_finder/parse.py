"""Parse timestamped screenplays into structured beats - plus a coverage report.

Pure code, no AI: timestamps are facts, so they must never come from a model.
Every downstream stage refers to beats by ID and looks timestamps up here.

Accepted format (per beat):
    **SPEAKER (Tone, Tone):**          <- or plain `SPEAKER (Tone):`, or **CAMERA:** / **ACTION:** / **SOUND FX:**
    Start: (00:11.30) | End: (00:13.10)  <- brackets optional; mm:ss.s, mm:ss.ss or h:mm:ss.ss; End optional
    "The line or description."
A header line counts as a beat header only if the next non-empty line is a Start line, so prose
that happens to end with a colon is never mistaken for a speaker.
"""
from __future__ import annotations

import re
from decimal import Decimal
from dataclasses import dataclass, field
from pathlib import Path

# h:mm:ss.ss | mm:ss.ss  (the hour group only matches when two colons are present)
_T = r"(?:(\d{1,2}):(?=\d{1,2}:))?(\d{1,3}):(\d{2}(?:[.,]\d+)?)"
SPAN_RE = re.compile(
    rf"^[\W_]*start\s*[:=\-]?\s*[\(\[]?\s*{_T}\s*[\)\]]?\s*"
    rf"(?:(?:\||,|;|/|-|–|—|\bto\b)?\s*end\s*[:=\-]?\s*[\(\[]?\s*{_T}\s*[\)\]]?)?\s*$", re.I)
HEADER_RE = re.compile(r"^\**\s*([^:*\n]{1,80}?)\s*\**\s*:\s*\**\s*$")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
SCENE_HEADING_RE = re.compile(r"^Scene heading:\s*(.*)$", re.I)
SCENE_TIME_RE = re.compile(rf"^Scene (start|end):\s*[\(\[]?\s*{_T}", re.I)
META_RE = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES):\s*(.*)$", re.I)
PAREN_RE = re.compile(r"\(([^)]*)\)")
ANY_TIME_RE = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?:[.,]\d+)?(?![\d:])")  # broad: anything that looks like a time
SECTION_WORDS = {"ACTIONS AND DIALOGUE", "SCENE HEADING", "SCENE START", "SCENE END", "SCRIPT START",
                 "SCRIPT END", "EPISODE NUMBER", "EPISODE LOGLINE", "CONTINUITY NOTES"}
EP_NUM_RE = re.compile(r"(?:episode|ep)[\s_-]*0*(\d+)", re.I)
SCRIPT_EXTS = {".txt", ".md"}


def to_sec(h: str | None, m: str, s: str) -> float:
    seconds = Decimal(s.replace(",", "."))
    if not 0 <= seconds < 60 or (h is not None and not 0 <= int(m) < 60) or -seconds.as_tuple().exponent > 9:
        raise ValueError("invalid timestamp component or more than nine fractional digits")
    return float(Decimal(int(h) * 3600 if h else 0) + Decimal(int(m) * 60) + Decimal(s.replace(",", ".")))


def _mmss(t: float, places: int) -> str:
    m, s = divmod(max(t, 0.0), 60)
    txt = f"{s:0{3 + places}.{places}f}"
    if txt.startswith("60"):              # 59.96 s formatted to 1 place would read '00:60.0'
        m, txt = m + 1, f"{0:0{3 + places}.{places}f}"
    return f"{int(m):02d}:{txt}"


def fmt(t: float) -> str:
    """Seconds -> MM:SS.s, used in model prompts. Byte-identical to v4 (cache keys) except that the
    minute now rolls over correctly (59.96 s -> '01:00.0', not '00:60.0')."""
    return _mmss(t, 1)


def fmt_ts(t: float) -> str:
    """Seconds -> MM:SS.ss: the precision the scripts are written in. Used for every exported timestamp."""
    return _mmss(t, max(2, min(9, -Decimal(str(t)).as_tuple().exponent)))


@dataclass
class Beat:
    id: str            # "e03b041" - stable reference used by the LLM
    episode: int
    kind: str          # CAMERA | ACTION | DIALOGUE | SFX
    start: float
    end: float
    text: str
    speaker: str = ""
    emotion: str = ""  # parenthetical tags, e.g. "Cold, Commanding"
    scene: int = 0
    timing_inferred: bool = False
    offscreen: bool = False

    def render(self, precise: bool = False) -> str:
        """Compact one-line form used in LLM prompts (precise=False, cache-stable) and exports (precise=True)."""
        f = fmt_ts if precise else fmt
        t = f"[{self.id} {f(self.start)}-{f(self.end)}]"
        if self.kind == "DIALOGUE":
            tag = f" ({self.emotion})" if self.emotion else ""
            return f"{t} {self.speaker}{tag}: {self.text}"
        return f"{t} {self.kind}: {self.text}"


@dataclass
class Scene:
    num: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    number: int
    path: str
    logline: str = ""
    continuity: str = ""
    scenes: list[Scene] = field(default_factory=list)
    beats: list[Beat] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [b.end for b in self.beats] + [s.end for s in self.scenes]
        return max(ends) if ends else 0.0

    def beats_between(self, t0: float, t1: float) -> list[Beat]:
        """Beats that are visible/audible inside [t0, t1]."""
        return [b for b in self.beats if b.end > t0 + 0.05 and b.start < t1 - 0.05]


@dataclass
class FileReport:
    path: str
    timestamp_lines: int = 0           # lines that LOOK like they hold a time (broad regex)
    parsed_lines: int = 0              # of those, lines the parser actually used
    unparsed: list[tuple[int, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fractions: list[str] = field(default_factory=list)


@dataclass
class ParseReport:
    files: list[FileReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def timestamp_lines(self) -> int:
        return sum(f.timestamp_lines for f in self.files)

    @property
    def parsed_lines(self) -> int:
        return sum(f.parsed_lines for f in self.files)

    @property
    def coverage(self) -> float:
        return self.parsed_lines / self.timestamp_lines if self.timestamp_lines else 0.0

    @property
    def unparsed(self) -> list[tuple[str, int, str]]:
        return [(Path(f.path).name, n, t) for f in self.files for n, t in f.unparsed]

    @property
    def all_warnings(self) -> list[str]:
        return self.warnings + [f"{Path(f.path).name}: {w}" for f in self.files for w in f.warnings]

    def max_fraction(self) -> str:
        fr = [f for fr in self.files for f in fr.fractions]
        return max(fr, key=lambda x: float("0." + x)) if fr else ""

    def unit_note(self, frame_rate: int) -> str | None:
        fr = [f for fr in self.files for f in fr.fractions if len(f) == 2]
        if fr and max(int(f) for f in fr) < frame_rate:
            return (f"Every 2-digit fraction is below {frame_rate} - '.{max(fr)}' might be FRAMES, not hundredths "
                    f"of a second. Assuming decimal seconds.")
        return None

    def summary(self) -> dict:
        return {"files": len(self.files), "timestamp_lines": self.timestamp_lines,
                "parsed_lines": self.parsed_lines, "coverage": round(self.coverage, 4),
                "unparsed": len(self.unparsed), "warnings": len(self.all_warnings),
                "largest_fraction": self.max_fraction(), "timestamp_unit": "decimal seconds (assumed)"}


def _classify(header: str) -> tuple[str, str, str, bool]:
    """'NAME (Tone, Tone)' -> (DIALOGUE, NAME, 'Tone, Tone', False)."""
    name = PAREN_RE.sub("", header).strip()
    tags = [p.strip() for p in PAREN_RE.findall(header)]
    up = name.upper()
    if up == "CAMERA":
        return "CAMERA", "", "", False
    if up == "ACTION":
        return "ACTION", "", "", False
    if up in ("SOUND FX", "SFX", "SOUND"):
        return "SFX", "", "", False
    off_words = ("off-screen", "o.s.", "v.o.", "voice-over", "offscreen")
    off = any(t.lower() in off_words for t in tags)
    emo = ", ".join(t for t in tags if t.lower() not in off_words)
    return "DIALOGUE", name, emo, off


def _is_header(s: str, nxt: str) -> str | None:
    """Return the header text if `s` is a beat header (its next non-empty line is a Start line)."""
    if not SPAN_RE.match(nxt):
        return None
    m = HEADER_RE.match(s)
    if not m:
        return None
    text = m.group(1).strip()
    return None if PAREN_RE.sub("", text).strip().upper() in SECTION_WORDS else text


def parse_episode(path: str | Path, force_number: int | None = None,
                  report: FileReport | None = None) -> Episode:
    path = Path(path)
    rep = report if report is not None else FileReport(str(path))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    num_guess = EP_NUM_RE.search(path.name)
    ep = Episode(number=int(num_guess.group(1)) if num_guess else 0, path=str(path))
    nxt_nonblank = [""] * len(lines)
    following = ""
    for i in range(len(lines) - 1, -1, -1):           # next non-empty line after each line
        nxt_nonblank[i] = following
        if lines[i].strip():
            following = lines[i].strip()

    cur_scene: Scene | None = None
    pending: dict | None = None
    raw: list[dict] = []
    used: set[int] = set()

    def flush():
        nonlocal pending
        if pending and pending.get("start") is not None:
            pending["text"] = " ".join(pending["text"]).strip().strip('"').strip()
            raw.append(pending)
        pending = None

    for i, line in enumerate(lines):
        s = line.strip()
        if m := META_RE.match(s):
            key, val = m.group(1).upper(), m.group(2)
            if key == "EPISODE NUMBER" and (n := re.search(r"\d+", val)):
                ep.number = int(n.group())
            elif key == "EPISODE LOGLINE":
                ep.logline = val.strip()
            elif key == "CONTINUITY NOTES":
                ep.continuity = val.strip()
            used.add(i)
            continue
        if m := SCENE_RE.match(s):
            flush()
            cur_scene = Scene(num=int(m.group(1)), heading="", start=0.0, end=0.0)
            ep.scenes.append(cur_scene)
            continue
        if cur_scene and (m := SCENE_HEADING_RE.match(s)):
            cur_scene.heading = m.group(1).strip()
            continue
        if cur_scene and (m := SCENE_TIME_RE.match(s)):
            try:
                t = to_sec(m.group(2), m.group(3), m.group(4))
            except ValueError as exc:
                rep.warnings.append(f"line {i+1}: {exc}")
                continue
            if m.group(1).lower() == "start":
                cur_scene.start = t
            else:
                cur_scene.end = t
            used.add(i)
            continue
        if (hdr := _is_header(s, nxt_nonblank[i])) is not None:
            flush()
            kind, spk, emo, off = _classify(hdr)
            pending = {"kind": kind, "speaker": spk, "emotion": emo, "offscreen": off, "line": i + 1,
                       "start": None, "end": None, "text": [], "scene": cur_scene.num if cur_scene else 0}
            continue
        if pending is not None:
            if pending["start"] is None and (m := SPAN_RE.match(s)):
                g = m.groups()
                try:
                    start = to_sec(g[0], g[1], g[2])
                    end = to_sec(g[3], g[4], g[5]) if g[4] is not None else None
                except ValueError as exc:
                    rep.warnings.append(f"line {i+1}: {exc}")
                    pending = None
                    continue
                pending["start"], pending["end"] = start, end
                for frac in (g[2], g[5]):
                    if frac and re.search(r"[.,](\d+)$", frac):
                        rep.fractions.append(re.search(r"[.,](\d+)$", frac).group(1))
                used.add(i)
                continue
            if not s or s.startswith("--- SCRIPT"):
                flush()
                continue
            pending["text"].append(s)
    flush()
    if force_number is not None:
        ep.number = force_number

    # coverage: every line that looks like it holds a time must have been used
    for i, line in enumerate(lines):
        if ANY_TIME_RE.search(line):
            rep.timestamp_lines += 1
            if i in used:
                rep.parsed_lines += 1
            else:
                rep.unparsed.append((i + 1, line.strip()[:160]))

    # missing End -> next beat's start (warn); End <= Start -> warn
    order = sorted(range(len(raw)), key=lambda k: raw[k]["start"])
    for pos, k in enumerate(order):
        r = raw[k]
        r["timing_inferred"] = r["end"] is None or r["end"] <= r["start"]
        if r["end"] is None:
            later = [raw[j]["start"] for j in order[pos + 1:] if raw[j]["start"] > r["start"]]
            r["end"] = later[0] if later else r["start"] + 2.0
            rep.warnings.append(f"line {r['line']}: no End time - used {fmt(r['end'])} (next beat's start)")
        elif r["end"] <= r["start"] and r["end"] != r["start"]:
            rep.warnings.append(f"line {r['line']}: End {fmt(r['end'])} is before Start {fmt(r['start'])}")
    back = sum(1 for a, b in zip(raw, raw[1:]) if b["start"] < a["start"] - 1.0)
    if back:
        rep.warnings.append(f"{back} beat(s) start more than 1 s earlier than the beat above them (out of order)")

    raw.sort(key=lambda r: (r["start"], r["end"]))
    for i, r in enumerate(raw):
        ep.beats.append(Beat(id=f"e{ep.number:02d}b{i:03d}", episode=ep.number, kind=r["kind"],
                             start=r["start"], end=max(r["end"], r["start"]), text=r["text"],
                             speaker=r["speaker"], emotion=r["emotion"], scene=r["scene"],
                             offscreen=r["offscreen"], timing_inferred=r["timing_inferred"]))
    return ep


def script_files(folder: str | Path) -> list[Path]:
    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in SCRIPT_EXTS and p.is_file()
             and not p.name.startswith((".", "._")) and p.name.lower() not in ("readme.md",)]
    return sorted(files, key=lambda p: _natural(p.name))


def parse_show(folder: str | Path, episodes: set[int] | None = None,
               settings=None) -> tuple[list[Episode], ParseReport]:
    """Parse every script in a folder. Returns (episodes ordered by number, report)."""
    report = ParseReport()
    files = script_files(folder)
    parsed = []
    for f in files:
        fr = FileReport(str(f))
        report.files.append(fr)
        e = parse_episode(f, report=fr)
        if e.beats:
            parsed.append(e)
        else:
            fr.warnings.append("no timestamped beats found - file skipped")
    nums = [e.number for e in parsed]
    if 0 in nums or len(set(nums)) != len(nums):
        dup = sorted({n for n in nums if nums.count(n) > 1})
        report.warnings.append("episode numbers missing or duplicated" + (f" ({dup})" if dup else "")
                               + " - numbered by file name order instead")
        renum = []
        for i, e in enumerate(parsed):
            fr = next(r for r in report.files if r.path == e.path)
            fresh = FileReport(e.path)
            renum.append(parse_episode(e.path, force_number=i + 1, report=fresh))
            fr.warnings = fresh.warnings
        parsed = renum
    parsed.sort(key=lambda e: e.number)
    nums = [e.number for e in parsed]
    if nums:
        gaps = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
        if gaps:
            report.warnings.append(f"missing episode number(s): {gaps[:20]}")
    if settings is not None:
        for e in parsed:
            if not settings.episode_min_sec <= e.duration <= settings.episode_max_sec:
                report.warnings.append(f"episode {e.number}: duration {fmt(e.duration)} is outside the expected "
                                       f"{settings.episode_min_sec:.0f}-{settings.episode_max_sec:.0f} s")
        if note := report.unit_note(settings.frame_rate):
            report.warnings.append(note)
    if episodes:
        parsed = [e for e in parsed if e.number in episodes]
    return parsed, report


def load_show(folder: str | Path) -> list[Episode]:
    """Backwards-compatible: just the episodes."""
    return parse_show(folder)[0]


def parse_episode_range(spec: str | None) -> set[int] | None:
    """'1-10,15,20-22' -> {1..10, 15, 20, 21, 22}. Empty/None -> None (all episodes)."""
    if not spec or not str(spec).strip():
        return None
    out: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"bad episode range '{part}'")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def _natural(name: str):
    """'ep2' sorts before 'ep10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]
