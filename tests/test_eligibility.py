"""v7 verified eligibility: one decision per clip, evidence-checked judge verdicts, re-cuts that are judged
again, and exports that agree with each other.

The fixtures are small invented episodes. Each one reproduces the STRUCTURE of a failure seen on the real
input pack (an ending the clip answers itself, a visual hook the rules missed, an unexplained cross-cut...),
with generic content, because the real scripts are confidential and stay out of the repo.
"""
import csv
import json

import pytest

from conftest import SYNTHETIC
from promo_finder import config as C
from promo_finder.candidates import Promo, Segment, end_before_variant, opening_variant, rescore
from promo_finder.eligibility import decide, ending_repair
from promo_finder.llm import LLM
from promo_finder.llm_stages import blend_signals, clip_context, judge_v7, render_clip_v7, validate_judgment
from promo_finder.parse import Beat, Episode, fmt, fmt_ts, parse_show
from promo_finder.pipeline import estimate, run
from promo_finder.rank import quality_checks, rank_verified
from promo_finder.report import write_all
from promo_finder.signals import annotate

S = C.load(preset="v7").settings


def episode(number, rows, logline="") -> Episode:
    """rows: (kind, start, end, text[, speaker, emotion])"""
    e = Episode(number=number, path=f"ep{number}.txt", logline=logline)
    for i, r in enumerate(sorted(rows, key=lambda r: (r[1], r[2]))):
        kind, s, t, text, *rest = r
        e.beats.append(Beat(id=f"e{number:02d}b{i:03d}", episode=number, kind=kind, start=s, end=t, text=text,
                            speaker=rest[0] if rest else "", emotion=rest[1] if len(rest) > 1 else ""))
    return e


def setup(*eps):
    E = {e.number: e for e in eps}
    sig = annotate(list(eps), S.lexicon, None, S.resolve_on_calm_tone, S.count_production_cues)
    return E, sig


def promo(E, sig, *segs, kind=None) -> Promo:
    p = Promo("", kind or ("stitched" if len(segs) > 1 else "continuous"), [Segment(*s) for s in segs])
    return rescore(p, E, sig, S)


def ref_of(ctx, beat_id) -> str:
    return next(r.ref for r in ctx.refs if r.beat.id == beat_id)


def judge_with(p, E, hook=None, ending=None, standalone=None, transition=None, scores=None, drop=(), **extra):
    """Attach a validated judgment built from section overrides (refs given as beat ids)."""
    ctx = clip_context(p, E)
    conv = lambda ids: [ref_of(ctx, i) for i in ids]
    first, last = ctx.refs[0].beat.id, ctx.refs[-1].beat.id
    data = {"clip_id": ctx.clip_id,
            "hook": {"verdict": "pass", "evidence": [first], "description": "opening action", "mechanism": "danger"},
            "ending": {"verdict": "pass", "evidence": [last], "final_moment": "final beat",
                       "central_question": "will it work?", "final_moment_type": "suspense",
                       "open_question": "will she get away?", "answered_within_clip": False, "answered_by": [],
                       "reason": "open"},
            "standalone": {"verdict": "pass", "evidence": [first], "conflict": "a clear conflict",
                           "blocking_confusion": "", "harmless_curiosity": ""},
            "scores": {"hook": 7, "cliffhanger": 7, "standalone": 7},
            "why_it_works": "Opens on immediate danger and cuts before the answer.", "limitations": ""}
    if p.kind == "stitched":
        sides = [next(r.beat.id for r in ctx.refs if r.segment == i) for i in range(len(p.segments))]
        data["transition"] = {"verdict": "pass", "evidence": sides, "connection": "same person, next step", "reason": ""}
    for name, over in (("hook", hook), ("ending", ending), ("standalone", standalone), ("transition", transition)):
        if over:
            data[name] = {**data[name], **over}
    for name in ("hook", "ending", "standalone", "transition"):
        if name in data:
            for f in ("evidence", "answered_by"):
                if f in data[name]:
                    data[name][f] = conv(data[name][f])
    if scores:
        data["scores"].update(scores)
    data.update(extra)
    for k in drop:
        data.pop(k, None)
    j, errors = validate_judgment(data, ctx)
    j["errors"] = errors
    p.judgment = j
    return errors


# ----------------------------------------------------------------------------- fixtures
SEARCH = episode(1, [   # an object is searched for, found, then grasped before the clip ends
    ("CAMERA", 0.0, 4.0, "Wide shot of a frozen field as Mara digs through the snow with bare hands."),
    ("ACTION", 0.5, 4.0, "Mara scrapes at the ice, her knuckles raw."),
    ("DIALOGUE", 4.2, 6.0, "Stop! You will freeze!", "OWEN", "Desperate"),
    ("CAMERA", 6.0, 12.0, "Close-up of Mara's raw hands scraping the ice."),
    ("CAMERA", 12.0, 20.0, "Owen strains against the guard holding him back."),
    ("DIALOGUE", 13.0, 15.0, "Let me go!", "OWEN", "Furious"),
    ("CAMERA", 20.0, 30.0, "Mara crawls deeper into the drift."),
    ("DIALOGUE", 30.0, 33.0, "It was right here, I know it.", "MARA", "Panting"),
    ("CAMERA", 33.0, 36.0, "Extreme close-up of a brass key glinting under the ice."),
    ("CAMERA", 36.0, 38.0, "Close-up of Mara's wide eyes."),
    ("ACTION", 38.0, 41.0, "Mara closes her bleeding fingers around the brass key."),
    ("CAMERA", 41.0, 44.0, "Macro shot of her fist gripping the key as the episode cuts to black."),
])

ORDER = episode(2, [    # a mid-clip order, then a different, final threat
    ("CAMERA", 0.0, 3.0, "Lady Vane strikes the maid across the back with a cane."),
    ("DIALOGUE", 3.0, 6.0, "Know your place.", "LADY VANE", "Cold"),
    ("CAMERA", 6.0, 10.0, "The maid steadies herself against the bedpost."),
    ("DIALOGUE", 10.0, 14.0, "Polish every boot in the house.", "LADY VANE", "Harsh"),
    ("CAMERA", 14.0, 24.0, "The maid kneels by a pile of muddy boots."),
    ("DIALOGUE", 24.0, 26.5, "My lady—", "PIP", "Pleading"),
    ("ACTION", 27.0, 29.0, "Lady Vane turns and raises the cane at Pip."),
    ("DIALOGUE", 29.0, 32.0, "Shall I teach you the same lesson?", "LADY VANE", "Furious, Threatening"),
    ("CAMERA", 32.0, 34.0, "Pip flinches and lowers her eyes."),
])

TENDER = episode(3, [   # a tender ending whose only "cliffhanger" is a music cue
    ("CAMERA", 0.0, 3.0, "Ben pins the burglar against the fence."),
    ("DIALOGUE", 1.0, 3.0, "Who hired you?", "BEN", "Roaring"),
    ("CAMERA", 3.0, 12.0, "The burglar runs off down the alley."),
    ("DIALOGUE", 12.0, 15.0, "Where did you learn to fight like that?", "ANNA", "Wondering"),
    ("DIALOGUE", 15.0, 18.0, "Somewhere I would rather not remember.", "BEN", "Somber"),
    ("CAMERA", 18.0, 26.0, "Anna and Ben stand close under a streetlight."),
    ("SFX", 22.0, 26.0, "Soft piano music rises into a dramatic cliffhanger sting."),
])

FIRE = episode(4, [     # a purely visual hook: no dialogue in the first 3 s
    ("CAMERA", 0.0, 2.5, "The hallway is a wall of flames; the only door is blocked."),
    ("ACTION", 1.0, 4.0, "Nora shields her face from the heat."),
    ("DIALOGUE", 3.0, 5.0, "The door is blocked!", "NORA", "Panicked"),
    ("DIALOGUE", 5.5, 7.0, "The window, now!", "TOM", "Resolute"),
    ("CAMERA", 7.0, 15.0, "Tom drags Nora toward the bedroom window."),
    ("CAMERA", 15.0, 20.0, "A photo album burns on the dresser."),
    ("DIALOGUE", 20.0, 21.5, "Our photos!", "NORA", "Frantic"),
    ("CAMERA", 21.5, 26.0, "A beam cracks overhead."),
])

RING = episode(9, [     # a later, unrelated-looking interrogation
    ("CAMERA", 0.0, 3.0, "Nora's aunt seizes her by the wrist."),
    ("DIALOGUE", 1.0, 3.0, "Stay right there!", "AUNT", "Desperate"),
    ("DIALOGUE", 3.5, 7.0, "Who gave you that locket? Answer me!", "AUNT", "Demanding"),
    ("CAMERA", 7.0, 12.0, "Nora freezes, the locket between their hands."),
])

ACCUSE = episode(5, [
    ("DIALOGUE", 0.0, 2.0, "Someone drugged my tea!", "GRANDMOTHER", "Shouting"),
    ("DIALOGUE", 2.5, 4.5, "And you let it happen!", "GRANDMOTHER", "Accusing"),
    ("CAMERA", 4.5, 10.0, "The old woman slumps back into her armchair."),
])
RETURN = episode(12, [
    ("CAMERA", 0.0, 6.0, "Henry loads a rifle in a dark barn."),
    ("DIALOGUE", 6.0, 9.0, "Saturday.", "HENRY", "Chilling"),
    ("DIALOGUE", 9.0, 14.0, "They will not see me coming.", "HENRY", "Cold"),
    ("CAMERA", 14.0, 20.0, "Henry walks out into the rain."),
])

SHOVE = episode(7, [    # the advertised action happens just BEFORE the obvious cut-in
    ("DIALOGUE", 0.0, 2.0, "Move, or else!", "THUG", "Menacing"),
    ("DIALOGUE", 2.0, 2.8, "No.", "IRIS", "Steely"),
    ("CAMERA", 2.0, 3.5, "Extreme close-up on Iris's eyes."),
    ("ACTION", 2.9, 5.0, "The thug shoves Iris hard; she crashes onto the pavement."),
    ("CAMERA", 5.0, 9.0, "Iris's phone skids across the pavement."),
    ("ACTION", 5.5, 9.0, "Iris crawls toward it, her palm bleeding."),
    ("DIALOGUE", 9.0, 11.0, "That was my dad's phone...", "IRIS", "Sobbing"),
    ("CAMERA", 11.0, 20.0, "The gang laughs and circles her."),
    ("CAMERA", 20.0, 30.0, "A car door slams; someone walks up behind them."),
    ("DIALOGUE", 30.0, 33.0, "You lost, old man?", "THUG", "Arrogant"),
    ("CAMERA", 33.0, 36.0, "Iris looks up, breath held."),
])


# ----------------------------------------------------------------------------- answered cliffhanger
def test_answered_cliffhanger_is_rejected_and_recut_before_the_answer():
    E, sig = setup(SEARCH)
    p = promo(E, sig, (1, 0.0, 44.0))
    judge_with(p, E, ending={"verdict": "pass", "open_question": "Did she find the key?",
                             "answered_within_clip": True, "answered_by": ["e01b010"]})
    d = decide(p, E, sig, S)
    assert d["status"] == "rejected"
    assert p.checks["ending"]["status"] == "fail" and "answers" in p.checks["ending"]["reason"]
    assert p.checks["ending"]["evidence"][0]["beat"] == "e01b010"

    t, why = ending_repair(p, E, sig, S)
    assert t == 38.0
    q = end_before_variant(p, E, sig, S, t, why)
    assert q is not None and q.segments[0].end <= 38.0 and q.key != p.key
    assert q.origin["derived_from"] == p.key and "e01b010" in q.origin["change"]


def test_resolved_final_moment_fails_even_when_the_judge_finds_a_follow_on_question():
    """The judge can always find SOME question ('what will she do now?'). The final moment must itself be
    suspense; a search that succeeds is a resolution, and the re-cut ends before it."""
    E, sig = setup(SEARCH)
    p = promo(E, sig, (1, 0.0, 44.0))
    judge_with(p, E, ending={"verdict": "pass", "final_moment_type": "resolution", "answered_by": ["e01b008"],
                             "open_question": "What will she do with the key?"})
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "fail" and "releases the clip's tension" in p.checks["ending"]["reason"]
    t, _ = ending_repair(p, E, sig, S)
    assert t == 33.0                                            # before the key is first seen, not just grasped

    judge_with(p, E, ending={"verdict": "pass", "final_moment_type": "resolution", "answered_by": []})
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "uncertain"          # contradictory and unsupported: review


def test_resolution_without_a_cited_beat_is_recut_on_an_earlier_tension_beat():
    """The judge classes the final moment as a resolution but cites no answering beat: the clip goes to
    review AND an earlier cut is proposed on the last strong tension beat (it is then judged on its own)."""
    touch = episode(11, [
        ("CAMERA", 0.0, 4.0, "Wide shot of a frozen field as Mara digs through the snow with bare hands."),
        ("DIALOGUE", 4.2, 6.0, "Stop! You will freeze!", "OWEN", "Desperate"),
        ("CAMERA", 6.0, 30.0, "Mara crawls deeper into the drift."),
        ("ACTION", 30.0, 33.0, "Her glove snags on something cold; she freezes in shock."),
        ("CAMERA", 33.0, 36.0, "Extreme close-up of a brass key glinting under the ice."),
        ("ACTION", 36.0, 40.0, "Mara closes her fingers around the brass key."),
        ("CAMERA", 40.0, 44.0, "Macro shot of her fist gripping the key."),
    ])
    E, sig = setup(touch)
    p = promo(E, sig, (11, 0.0, 44.0))
    judge_with(p, E, ending={"verdict": "pass", "final_moment_type": "resolution", "answered_by": []})
    decide(p, E, sig, S)
    assert p.decision["status"] == "review"
    t, why = ending_repair(p, E, sig, S)
    assert t == 33.0 and "tension beat" in why
    q = end_before_variant(p, E, sig, S, t, why)
    assert q.segments[0].end == 33.0 and q.judgment == {}


def test_source_bounds_allow_display_rounding_only():
    E, sig = setup(SEARCH)                                       # episode ends at 44.00
    from promo_finder.eligibility import deterministic_checks
    ok = Promo("", "continuous", [Segment(1, 20.0, 44.02)])     # '00:44.0' typed for a source 00:43.98-ish end
    far = Promo("", "continuous", [Segment(1, 20.0, 44.5)])
    assert deterministic_checks(ok, E, S)["source_bounds"]["status"] == "pass"
    assert deterministic_checks(far, E, S)["source_bounds"]["status"] == "fail"


def test_question_that_presupposes_a_resolution_cannot_pass():
    E, sig = setup(SEARCH)
    p = promo(E, sig, (1, 0.0, 44.0))
    judge_with(p, E, ending={"open_question": "What will she do now that she has the key?"})
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "uncertain" and "presupposes" in p.checks["ending"]["reason"]


def test_neutral_final_moment_is_not_an_unresolved_ending():
    E, sig = setup(TENDER)
    p = promo(E, sig, (3, 0.0, 26.0))
    judge_with(p, E, ending={"verdict": "fail", "final_moment_type": "neutral", "evidence": ["e03b005"],
                             "final_moment": "they look at each other"})
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "fail" and p.decision["status"] == "rejected"
    judge_with(p, E, ending={"verdict": "pass", "final_moment_type": "neutral", "evidence": ["e03b005"]})
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "uncertain"
    _, errors = validate_judgment({"clip_id": clip_context(p, E).clip_id,
                                   "ending": {"verdict": "pass", "evidence": [], "final_moment_type": "dramatic"}},
                                  clip_context(p, E))
    assert "ending.final_moment_type must be suspense, resolution or neutral" in errors


def test_answered_flag_without_a_cited_beat_is_uncertain_not_pass():
    E, sig = setup(SEARCH)
    p = promo(E, sig, (1, 0.0, 44.0))
    judge_with(p, E, ending={"answered_within_clip": True, "answered_by": []})
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "uncertain" and p.decision["status"] == "review"


# ----------------------------------------------------------------------------- ending grounded in the endpoint
def test_ending_evidence_must_come_from_the_final_seconds():
    E, sig = setup(ORDER)
    p = promo(E, sig, (2, 0.0, 34.0))
    judge_with(p, E, ending={"evidence": ["e02b003"], "open_question": "Will she polish the boots?"})   # +10 s
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "uncertain" and "final" in p.checks["ending"]["reason"]
    assert p.cliff_text == ""                                           # the mid-clip question is not exported
    assert "flinches" in p.final_moment                                 # the export describes the real endpoint

    judge_with(p, E, ending={"evidence": ["e02b007"], "final_moment": "Lady Vane threatens Pip with the same lesson",
                             "open_question": "Will Pip be punished too?"})
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "pass" and p.decision["status"] == "recommended"
    assert p.cliff_text == "Will Pip be punished too?"


def test_music_cue_and_the_word_cliffhanger_are_not_ending_evidence():
    E, sig = setup(TENDER)
    cue = "e03b006"
    legacy = annotate([TENDER], "generic", None, True, True)
    assert legacy[cue]["cliff"] >= 0.8                       # v4-v6: the stage direction alone scored as a cliffhanger
    assert sig[cue]["cliff"] == 0.0 and sig[cue]["music_cue"]
    p = promo(E, sig, (3, 0.0, 26.0))
    judge_with(p, E, ending={"evidence": [cue], "open_question": "Will he hurt her?"})
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "uncertain" and "music or sound cue" in p.checks["ending"]["reason"]


def test_calm_tone_alone_does_not_make_rules_overrule_the_judge():
    E, sig = setup(TENDER)
    assert all(sig[b.id]["resolve"] == 0.0 for b in TENDER.beats if b.kind == "DIALOGUE")   # tones like Somber/Wondering
    p = promo(E, sig, (3, 0.0, 26.0))
    judge_with(p, E, ending={"evidence": ["e03b005"], "open_question": "Who is he really?"})
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "pass"


# ----------------------------------------------------------------------------- visual hooks
def test_clear_visual_hook_is_adjudicated_by_evidence_not_rejected_by_rules():
    E, sig = setup(FIRE)
    p = promo(E, sig, (4, 0.0, 26.0))
    # the v4-v6 rule check failed it: an untagged beat is capped at 0.4 x rule after blending
    legacy = annotate([FIRE], "generic", None, True, True)
    blend_signals(legacy, {}, {}, 0.6)
    assert not quality_checks(p, E, legacy, C.load(preset="v6").settings)["hook_in_first_3s"]

    judge_with(p, E, hook={"verdict": "pass", "evidence": ["e04b000"], "description": "Flames seal the only exit"})
    decide(p, E, sig, S)
    assert p.checks["hook"]["status"] == "pass" and p.checks["hook"]["source"] == "judge + rules"
    assert p.checks["hook"]["evidence"][0]["clip_time"].startswith("+0.0")

    judge_with(p, E, hook={"verdict": "fail", "evidence": [], "description": "nothing happens"}, scores={"hook": 3})
    decide(p, E, sig, S)
    assert p.checks["hook"]["status"] == "uncertain" and p.decision["status"] == "review"
    assert "flames" in p.checks["hook"]["reason"]


def test_hook_evidence_must_start_in_the_first_three_seconds():
    E, sig = setup(FIRE)
    p = promo(E, sig, (4, 0.0, 26.0))
    judge_with(p, E, hook={"evidence": ["e04b005"]})                     # a beat at +15 s
    decide(p, E, sig, S)
    assert p.checks["hook"]["status"] == "uncertain" and "first 3 s" in p.checks["hook"]["reason"]


# ----------------------------------------------------------------------------- stitched continuity
def test_cut_needs_a_stated_connection_with_evidence_from_both_sides():
    E, sig = setup(FIRE, RING)
    p = promo(E, sig, (4, 0.0, 16.0), (9, 0.0, 12.0))
    judge_with(p, E, transition={"evidence": ["e09b001"]})              # only the second segment cited
    decide(p, E, sig, S)
    assert p.checks["transition"]["status"] == "uncertain"

    judge_with(p, E, transition={"verdict": "fail", "evidence": [], "connection": "",
                                 "reason": "nothing explains the jump from the fire to the questioning"})
    decide(p, E, sig, S)
    assert p.decision["status"] == "rejected" and "cut makes sense" in p.decision["failure_reasons"][0]

    judge_with(p, E)                                                    # pass, a beat from each side, a connection
    decide(p, E, sig, S)
    assert p.decision["status"] == "recommended"
    assert any("visual continuity" in n for n in p.decision["notes"])  # still flagged for a human, non-blocking


def test_judge_pass_on_a_cut_with_no_shared_character_goes_to_review():
    E, sig = setup(ACCUSE, RETURN)
    p = promo(E, sig, (5, 0.0, 10.0), (12, 0.0, 20.0))
    judge_with(p, E)
    decide(p, E, sig, S)
    assert p.checks["transition"]["status"] == "uncertain" and "share no named character" in p.checks["transition"]["reason"]


def test_missing_transition_answer_is_uncertain():
    E, sig = setup(FIRE, RING)
    p = promo(E, sig, (4, 0.0, 16.0), (9, 0.0, 12.0))
    errors = judge_with(p, E, drop=("transition",))
    assert any("transition" in e for e in errors)
    decide(p, E, sig, S)
    assert p.checks["transition"]["status"] == "uncertain"


# ----------------------------------------------------------------------------- judge contract
def test_missing_or_malformed_judge_fields_stay_uncertain():
    E, sig = setup(ORDER)
    p = promo(E, sig, (2, 0.0, 34.0))
    errors = judge_with(p, E, drop=("ending",), hook={"verdict": "maybe"}, scores={"standalone": "high"})
    assert {"section 'ending' is missing", "hook.verdict must be pass, fail or uncertain",
            "scores.standalone must be a number from 0 to 10"} <= set(errors)
    decide(p, E, sig, S)
    assert p.checks["ending"]["status"] == "uncertain" and p.checks["hook"]["status"] == "uncertain"
    assert p.decision["status"] == "review"


def test_unknown_refs_and_wrong_clip_id_are_structural_errors():
    E, sig = setup(ORDER)
    p = promo(E, sig, (2, 0.0, 34.0))
    ctx = clip_context(p, E)
    _, errors = validate_judgment({"clip_id": "K000000", "hook": {"verdict": "pass", "evidence": ["B99"]}}, ctx)
    assert any(e.startswith("clip_id") for e in errors) and any("B99" in e for e in errors)


def test_verdict_that_contradicts_its_own_score_is_not_trusted():
    E, sig = setup(ORDER)
    p = promo(E, sig, (2, 0.0, 34.0))
    judge_with(p, E, ending={"evidence": ["e02b007"]}, scores={"standalone": 4})   # "pass" but 4/10
    decide(p, E, sig, S)
    assert p.checks["standalone"]["status"] == "uncertain" and "4/10" in p.checks["standalone"]["reason"]


def test_judge_v7_retries_invalid_replies_then_keeps_missing_answers_uncertain(make_llm):
    eps, _ = parse_show(SYNTHETIC)
    E, sig = {e.number: e for e in eps}, annotate(eps, S.lexicon, None, False, False)
    p = rescore(Promo("", "continuous", [Segment(1, 0.0, 30.3)]), E, sig, S)

    stats = judge_v7(make_llm("mock-v7wrongid"), [p], E, retries=2, log=lambda *_: None)
    assert stats["validation_retries"] == 1 and stats["replies_invalid"] == 1 and stats["clips_valid"] == 1
    assert stats["api_calls"] == 2 and stats["replies_parsed"] == 2

    q = rescore(Promo("", "continuous", [Segment(1, 0.0, 30.3)]), E, sig, S)
    stats = judge_v7(make_llm("mock-v7missing"), [q], E, retries=2, log=lambda *_: None)
    assert stats["validation_retries"] == 2 and stats["clips_partial"] == 1 and stats["answers"]["ending"]["missing"] == 1
    decide(q, E, sig, S)
    assert q.checks["ending"]["status"] == "uncertain" and q.decision["status"] == "review"


def test_judge_sees_only_the_selected_footage_in_playback_order(make_llm, mock_server):
    E, sig = setup(SHOVE)
    SHOVE.logline = "SECRET-LOGLINE the gang works for her landlord"
    SHOVE.continuity = "SECRET-CONTINUITY"
    p = promo(E, sig, (7, 5.0, 36.0))
    text = render_clip_v7(clip_context(p, E))
    assert "SECRET" not in text and "shoves Iris" not in text      # outside the clip: never shown
    kinds = [line.split()[2].rstrip(":") for line in text.splitlines()]
    assert {"CAMERA", "ACTION", "IRIS"} <= set(kinds)
    times = [float(line.split()[1][1:].split("-")[0]) for line in text.splitlines()]
    assert times == sorted(times)
    judge_v7(make_llm(), [p], E, log=lambda *_: None)
    body = json.dumps(mock_server.bodies[-1]["body"])
    assert "SECRET" not in body


# ----------------------------------------------------------------------------- boundaries
def test_opening_recut_captures_the_action_and_is_judged_again():
    E, sig = setup(SHOVE)
    p = promo(E, sig, (7, 5.0, 36.0))                                   # opens on the aftermath of the shove
    judge_with(p, E)
    q = opening_variant(p, E, sig, S)
    assert q is not None and q.segments[0].start <= 2.9
    shove = next(b for b in SHOVE.beats if "shoves" in b.text)
    assert q.segments[0].start - S.hook_lead <= shove.start < q.segments[0].start + S.hook_window
    # a re-cut is a new clip: nothing is carried over from the parent's judgment
    assert q.judgment == {} and q.llm == {} and q.key != p.key
    assert q.origin["derived_from"] == p.key and abs(q.duration - (36.0 - q.segments[0].start)) < 1e-6
    decide(q, E, sig, S)
    assert q.decision["status"] == "review"
    q.judgment = p.judgment                                             # even if someone copies it over...
    decide(q, E, sig, S)
    assert q.decision["status"] == "review" and "different footage" in q.decision["review_reasons"][0]


def test_no_recut_when_the_opening_already_shows_the_action():
    E, sig = setup(SHOVE)
    assert opening_variant(promo(E, sig, (7, 2.0, 36.0)), E, sig, S) is None


def test_deterministic_checks_catch_bad_boundaries():
    E, sig = setup(SEARCH, ORDER)
    p = Promo("", "continuous", [Segment(1, 14.0, 44.0)])               # cut-in inside a spoken line (13-15 s)
    rescore(p, E, sig, S)
    decide(p, E, sig, S)
    assert p.checks["dialogue_boundaries"]["status"] == "fail" and p.decision["status"] == "rejected"
    p = Promo("", "stitched", [Segment(2, 10.0, 20.0), Segment(1, 0.0, 12.0)])   # plays backwards in story time
    rescore(p, E, sig, S)
    decide(p, E, sig, S)
    assert p.checks["segment_order"]["status"] == "fail"


def test_duration_uses_a_rounding_tolerance_and_formats_roll_over():
    E, sig = setup(SEARCH)
    p = Promo("", "continuous", [Segment(1, 0.0, 60.04)])
    from promo_finder.eligibility import deterministic_checks
    assert deterministic_checks(p, {1: episode(1, [("CAMERA", 0, 61, "x")])}, S)["duration"]["status"] == "pass"
    assert fmt(59.96) == "01:00.0" and fmt_ts(119.999) == "01:59.999" and fmt_ts(67.45) == "01:07.45"


# ----------------------------------------------------------------------------- ranking
def test_scores_never_override_a_failed_check_and_top_k_is_a_maximum():
    E, sig = setup(ORDER, FIRE, SEARCH)
    good = promo(E, sig, (2, 0.0, 34.0))
    judge_with(good, E, ending={"evidence": ["e02b007"]})
    loud = promo(E, sig, (4, 0.0, 26.0))
    judge_with(loud, E, standalone={"verdict": "fail", "blocking_confusion": "who are these people?"},
               scores={"standalone": 2})
    for p in (good, loud):
        decide(p, E, sig, S)
    good.final, loud.final = 0.10, 0.99
    rec, rev, rej, stats = rank_verified([good, loud], S)
    assert rec == [good] and rej == [loud] and stats["top_k_max"] == S.top_k and len(rec) < S.top_k


def test_unsupported_performance_claims_are_replaced():
    E, sig = setup(ORDER)
    p = promo(E, sig, (2, 0.0, 34.0))
    judge_with(p, E, ending={"evidence": ["e02b007"]}, why_it_works="Classic bait that always performs well on TikTok.")
    d = decide(p, E, sig, S)
    assert "TikTok" not in p.why and d["explanations_replaced"] == 1 and p.why.startswith("Opens on")


# ----------------------------------------------------------------------------- end to end
def test_v7_end_to_end_lists_and_exports_agree(make_llm, tmp_path):
    res = run(str(SYNTHETIC), make_llm(), "The CEO's Hidden Wife", S, log=lambda *_: None)
    out = write_all(res, tmp_path / "out")
    m = res["meta"]
    data = json.loads((out / "promos.json").read_text())
    for r in data["promos"]:
        assert r["status"] == "recommended" and not r["failure_reasons"] and not r["review_reasons"]
        assert all(c["status"] == "pass" for c in r["checks"].values())
        assert r["rank_explanation"] and "TikTok" not in r["why_it_works"]
    for r in data["review"]:
        assert r["status"] == "review" and r["review_reasons"] and not r["failure_reasons"]
    for r in data["rejected"]:
        assert r["status"] == "rejected" and r["failure_reasons"]
    assert len(data["promos"]) <= S.top_k
    # one set of records behind every format
    with open(out / "promos.csv") as f:
        csv_rows = {(r["id"], r["status"]) for r in csv.DictReader(f)}
    json_rows = {(r["id"], r["status"]) for k in ("promos", "review", "rejected") for r in data[k]}
    assert csv_rows == json_rows
    md, html_ = (out / "report.md").read_text(), (out / "report.html").read_text()
    for r in data["promos"] + data["review"]:
        assert r["id"] in md and r["id"] in html_
    el = m["eligibility"]
    assert el["status"]["rejected"] == len(data["rejected"]) == m["rejected"]
    assert el["status"]["recommended"] >= len(data["promos"]) == m["returned"]
    assert el["returned_all_checks_pass"] and set(m["check_pass_rates"].values()) <= {1.0}
    # diagnostics distinguish calls, parsed replies, valid replies and explicit answers
    j = m["judge"]
    assert j["prompt"] == "judge:v7" and j["api_calls"] >= j["clips"] - j["cache_hits"]
    assert j["replies_parsed"] == j["replies_valid"] + j["replies_invalid"]
    assert sum(j["answers"]["ending"].values()) == j["clips_valid"] + j["clips_partial"]
    assert "judge:v7" in m["prompt_versions"] and "judge:v4" not in m["prompt_versions"]
    assert m["cliff_score_spread"]["judge_cliffhanger"]["scale"] == "0-10"
    assert m["cliff_score_spread"]["judge_cliffhanger"]["mean"] > 1          # on the 0-10 scale, not 0-1
    # a re-run is fully cached, including the repair rounds
    assert estimate(str(SYNTHETIC), make_llm(), "The CEO's Hidden Wife", S)["calls"] == 0


def test_v7_rules_only_recommends_nothing_unverified():
    res = run(str(SYNTHETIC), LLM(), settings=S, log=lambda *_: None)
    assert res["promos"] == [] and res["review"]
    assert all("not verified" in r for p in res["review"] for r in p.decision["review_reasons"])


def test_v7_estimate_counts_one_call_per_clip_plus_repairs(make_llm):
    est = estimate(str(SYNTHETIC), make_llm(), "X", S)
    assert est["by_stage"]["judge"] > 10 and est["by_stage"].get("judge_repairs", 0) > 0 and est["upper_bound"]


@pytest.mark.parametrize("preset", ["v4", "v6"])
def test_legacy_presets_keep_their_behaviour(make_llm, preset):
    """v4-v6 stay reproducible: no v7 statuses, boolean checks, needs-context list as before."""
    res = run(str(SYNTHETIC), make_llm(), "The CEO's Hidden Wife", C.load(preset=preset).settings, log=lambda *_: None)
    assert res["promos"] and not res["review"] and not res["rejected"]
    assert all(isinstance(v, bool) for p in res["promos"] for v in p.checks.values())
    assert "eligibility" not in res["meta"]


def test_review_list_puts_clips_with_a_specific_concern_first():
    E, sig = setup(ORDER, FIRE)
    flagged = promo(E, sig, (2, 0.0, 34.0))
    judge_with(flagged, E, ending={"evidence": ["e02b003"]})          # judged; evidence not in the final seconds
    unreviewed = promo(E, sig, (4, 0.0, 26.0))                        # never sent to the final reviewer
    for p in (flagged, unreviewed):
        decide(p, E, sig, S, unavailable="not selected for final review")
    flagged.final, unreviewed.final = 0.1, 0.9
    _, rev, _, _ = rank_verified([flagged, unreviewed], S)
    assert rev == [flagged, unreviewed]
