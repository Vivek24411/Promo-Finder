"""Parser hardening (2b). All fixtures are synthetic - no Kuku content."""
import pytest

from conftest import SYNTHETIC, write
from promo_finder.config import Settings
from promo_finder.parse import parse_episode, parse_episode_range, parse_show, to_sec

HEAD = "EPISODE NUMBER: Episode {n}\nEPISODE LOGLINE: test\n\n"


def beat(header, span, text):
    return f"{header}\n{span}\n{text}\n\n"


def one(tmp_path, span, header="**ANNA (Cold):**", name="ep1.txt"):
    p = write(tmp_path, name, HEAD.format(n=1) + beat(header, span, '"Hello."'))
    return parse_episode(p)


@pytest.mark.parametrize("span,start,end", [
    ("Start: (00:11.30) | End: (00:13.10)", 11.3, 13.1),     # the Kuku format
    ("Start: 00:11.30 | End: 00:13.10", 11.3, 13.1),         # no brackets
    ("start: 00:11.3 - end: 00:13.1", 11.3, 13.1),           # lowercase, dash, one decimal
    ("START:[00:11.30], END:[00:13.10]", 11.3, 13.1),        # caps, square brackets, comma
    ("Start: 0:00:11.30 | End: 0:00:13.10", 11.3, 13.1),     # h:mm:ss.ss
    ("Start: 1:02:03.50 | End: 1:02:05.00", 3723.5, 3725.0), # real hours
    ("Start: (00:11,30) | End: (00:13,10)", 11.3, 13.1),     # decimal comma
    ("Start:00:11.30 to End:00:13.10", 11.3, 13.1),          # 'to' separator, no spaces
])
def test_span_variations(tmp_path, span, start, end):
    ep = one(tmp_path, span)
    assert len(ep.beats) == 1
    assert ep.beats[0].start == pytest.approx(start) and ep.beats[0].end == pytest.approx(end)


@pytest.mark.parametrize("header,kind,speaker,emotion", [
    ("**ANNA (Cold, Firm):**", "DIALOGUE", "ANNA", "Cold, Firm"),
    ("**ANNA:**", "DIALOGUE", "ANNA", ""),
    ("ANNA (Cold):", "DIALOGUE", "ANNA", "Cold"),            # plain, no bold
    ("**ANNA (Cold)**:", "DIALOGUE", "ANNA", "Cold"),         # colon outside the bold
    ("Anna (Off-screen) (Whispering):", "DIALOGUE", "Anna", "Whispering"),
    ("**CAMERA:**", "CAMERA", "", ""),
    ("SOUND FX:", "SFX", "", ""),
])
def test_header_variations(tmp_path, header, kind, speaker, emotion):
    b = one(tmp_path, "Start: (00:01.00) | End: (00:02.00)", header=header).beats[0]
    assert (b.kind, b.speaker, b.emotion) == (kind, speaker, emotion)


def test_colon_prose_is_not_a_header(tmp_path):
    txt = HEAD.format(n=1) + "Actions and dialogue:\n\n" + beat("**CAMERA:**", "Start: (00:00.00) | End: (00:02.00)", "Wide.")
    ep = parse_episode(write(tmp_path, "e.txt", txt))
    assert [b.kind for b in ep.beats] == ["CAMERA"]


def test_missing_end_uses_next_start_and_warns(tmp_path):
    txt = HEAD.format(n=1) + beat("**A:**", "Start: (00:01.00)", '"x"') + beat("**B:**", "Start: (00:04.50) | End: (00:06.00)", '"y"')
    show, rep = parse_show(_folder(tmp_path, {"ep1.txt": txt}))
    assert show[0].beats[0].end == pytest.approx(4.5)
    assert any("no End time" in w for w in rep.all_warnings)


def test_end_before_start_and_out_of_order_warn(tmp_path):
    txt = (HEAD.format(n=1) + beat("**A:**", "Start: (00:10.00) | End: (00:09.00)", '"x"')
           + beat("**B:**", "Start: (00:02.00) | End: (00:03.00)", '"y"'))
    _, rep = parse_show(_folder(tmp_path, {"ep1.txt": txt}))
    w = " ".join(rep.all_warnings)
    assert "before Start" in w and "out of order" in w


def test_natural_sort_and_header_numbers(tmp_path):
    files = {f"show ep{n}.txt": HEAD.format(n=n) + beat("**A:**", "Start: (00:01.00) | End: (00:02.00)", '"x"')
             for n in (1, 2, 10)}
    show, rep = parse_show(_folder(tmp_path, files))
    assert [e.number for e in show] == [1, 2, 10]
    assert any("missing episode number" in w for w in rep.all_warnings)       # 3..9 are gaps


def test_duplicate_numbers_renumber_by_file_order(tmp_path):
    files = {n: HEAD.format(n=1) + beat("**A:**", "Start: (00:01.00) | End: (00:02.00)", '"x"')
             for n in ("part a.txt", "part b.txt")}
    show, rep = parse_show(_folder(tmp_path, files))
    assert [e.number for e in show] == [1, 2] and show[1].beats[0].id == "e02b000"
    assert any("duplicated" in w for w in rep.all_warnings)


def test_coverage_report_lists_unparsed_lines(tmp_path):
    txt = (HEAD.format(n=1) + beat("**A:**", "Start: (00:01.00) | End: (00:02.00)", '"x"')
           + "**B:**\nStrat: (00:03.00) | End: (00:04.00)\n\"typo in Start\"\n")
    _, rep = parse_show(_folder(tmp_path, {"ep1.txt": txt}))
    assert rep.timestamp_lines == 2 and rep.parsed_lines == 1 and rep.coverage == 0.5
    assert rep.unparsed[0][1] == 9 and "Strat" in rep.unparsed[0][2]      # 3 header lines + 4-line beat + "**B:**"


def test_coverage_gate_stops_the_pipeline(tmp_path):
    from promo_finder.llm import LLM
    from promo_finder.pipeline import ParseCoverageError, run
    txt = (HEAD.format(n=1) + "".join(beat("**CAMERA:**", f"Start: (00:{i * 5:02d}.00) | End: (00:{i * 5 + 5:02d}.00)", "shot") for i in range(6))
           + "**B:**\nBegin 00:31.00 until 00:33.00\n\"bad\"\n")
    folder = _folder(tmp_path, {"ep1.txt": txt})
    with pytest.raises(ParseCoverageError):
        run(folder, LLM(), log=lambda *_: None)
    run(folder, LLM(), log=lambda *_: None, force=True)        # --force continues


def test_frame_rate_warning(tmp_path):
    txt = HEAD.format(n=1) + "".join(beat("**A:**", f"Start: (00:0{i}.{f:02d}) | End: (00:0{i + 1}.{f:02d})", '"x"')
                                     for i, f in enumerate([5, 12, 24]))
    _, rep = parse_show(_folder(tmp_path, {"ep1.txt": txt}), settings=Settings())
    assert any("FRAMES" in w for w in rep.all_warnings)


def test_md_files_and_episode_ranges(tmp_path):
    files = {f"ep{n}.md": HEAD.format(n=n) + beat("**A:**", "Start: (00:01.00) | End: (00:02.00)", '"x"') for n in (1, 2, 3)}
    show, _ = parse_show(_folder(tmp_path, files), episodes={2, 3})
    assert [e.number for e in show] == [2, 3]
    assert parse_episode_range("1-3,7") == {1, 2, 3, 7} and parse_episode_range("") is None
    with pytest.raises(ValueError):
        parse_episode_range("5-2")


def test_to_sec():
    assert to_sec(None, "01", "02.5") == 62.5 and to_sec("1", "00", "00") == 3600


def test_synthetic_show_parses_fully():
    show, rep = parse_show(SYNTHETIC, settings=Settings())
    assert [e.number for e in show] == list(range(1, 11))
    assert rep.coverage == 1.0
    assert all(40 <= e.duration <= 120 for e in show)


def _folder(tmp_path, files: dict) -> str:
    d = tmp_path / "show"
    d.mkdir(exist_ok=True)
    for name, txt in files.items():
        (d / name).write_text(txt)
    return str(d)
