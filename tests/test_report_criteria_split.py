"""Tests for the side-by-side criteria report.

The driver's job is joining three files that were written by three different
scripts at three different times. The failures worth guarding are all joins:
reading the wrong population size, matching an anonymized system against the
wrong original, and silently dropping a criterion because its key did not
match. The last is the dangerous one, since a dropped criterion looks exactly
like a criterion that was addressed.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "report_criteria_split",
    Path(__file__).resolve().parents[1] / "scripts" / "report_criteria_split.py",
)
rcs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rcs)

ASRBN = "asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1"


def _sweep(n, singling_out, linkability):
    return {"n": n, "singling_out": singling_out, "linkability": linkability,
            "srd_idr": linkability, "lid_pdr": linkability}


def _comparison():
    """Two populations, so reading the wrong one is detectable."""
    return {"systems": {
        "cv27 original": {"by_n": [
            _sweep(100, math.exp(-1) + 0.40, 0.01 + 0.80),
            _sweep(9779, math.exp(-1) + 0.02, 1 / 9779 + 0.80),
        ]},
        "cv27 mcadams": {"by_n": [
            _sweep(100, math.exp(-1) + 0.40, 0.01 + 0.80),
            _sweep(9779, math.exp(-1) + 0.02, 1 / 9779 + 0.80),
        ]},
        f"cv27 {ASRBN}": {"by_n": [
            _sweep(100, math.exp(-1) + 0.10, 0.01 + 0.20),
            _sweep(9779, math.exp(-1) + 0.005, 1 / 9779 + 0.20),
        ]},
        # A different corpus, which must not leak into a cv27 report.
        "libri_test original": {"by_n": [_sweep(29, 0.9, 0.9)]},
    }}


def _trial():
    return {
        "cv27 original": {"eer": 0.0670, "cllr_min": 0.2508,
                          "gomez_barrero_dsys": 0.8284,
                          "zebra_population": 0.5312},
        "cv27 mcadams": {"eer": 0.0994, "cllr_min": 0.3506,
                         "gomez_barrero_dsys": 0.7560,
                         "zebra_population": 0.4575},
        f"cv27 {ASRBN}": {"eer": 0.3000, "cllr_min": 0.9000,
                          "gomez_barrero_dsys": 0.2000,
                          "zebra_population": 0.1000},
    }


def _inference():
    return {
        "cv27_enrolls": {"advantage": 0.6859, "utterances_per_speaker": 6.0},
        "cv27_enrolls_mcadams": {"advantage": 0.6919,
                                 "utterances_per_speaker": 6.0},
        f"cv27_enrolls_{ASRBN}": {"advantage": 0.6500,
                                  "utterances_per_speaker": 6.0},
        "cv27_trials_mixed": {"advantage": 0.7604,
                              "utterances_per_speaker": 20.0},
        "cv27_trials_mixed_mcadams": {"advantage": 0.7846,
                                      "utterances_per_speaker": 20.0},
        f"cv27_trials_mixed_{ASRBN}": {"advantage": 0.7300,
                                       "utterances_per_speaker": 20.0},
    }


def _collect(n=100):
    return rcs.collect(_comparison(), _trial(), _inference(), "cv27", n)


# The join itself.

def test_the_original_is_not_reported_as_a_system():
    """It is the reference every row is measured against."""
    assert "original" not in _collect()


def test_another_corpus_does_not_leak_in():
    collected = _collect()
    assert set(collected) == {"mcadams", ASRBN}


def test_the_requested_population_is_the_one_read():
    """Linkability's chance moves with N, so reading the wrong row would
    silently change every chance level in the report."""
    at_100 = _collect(100)[ASRBN]
    at_9779 = _collect(9779)[ASRBN]
    link_100 = next(r for m, r in at_100["rows"] if m == "linkability")
    link_9779 = next(r for m, r in at_9779["rows"] if m == "linkability")
    assert link_100.chance == pytest.approx(0.01)
    assert link_9779.chance == pytest.approx(1 / 9779)


def test_a_population_that_was_not_run_is_refused():
    with pytest.raises(SystemExit, match="N=7"):
        rcs.collect(_comparison(), _trial(), _inference(), "cv27", 7)


def test_a_corpus_with_no_original_is_refused():
    """Without a reference there is no room, and every ratio would be void."""
    comparison = {"systems": {"cv27 mcadams": {"by_n": [_sweep(100, 0.5, 0.5)]}}}
    with pytest.raises(SystemExit, match="no reference"):
        rcs.collect(comparison, {}, {}, "cv27", 100)


# Every criterion has to arrive, because a missing one reads as an addressed one.

def test_all_three_criteria_are_populated_when_the_inputs_are_complete():
    summary = _collect()[ASRBN]["summary"]
    assert summary["unmeasured"] == []
    for criterion in ("singling_out", "linkability", "inference"):
        assert summary["readings_per_criterion"][criterion] > 0


def test_both_evidence_levels_reach_the_inference_criterion():
    rows = _collect()[ASRBN]["rows"]
    inference = [m for m, r in rows if r.criterion == "inference"]
    assert len(inference) == 2
    assert any("6 utt/spk" in m for m in inference)
    assert any("20 utt/spk" in m for m in inference)


def test_a_missing_inference_file_leaves_the_criterion_unmeasured():
    """Which is what a verification-only report looks like, and it must not be
    mistaken for a system that addressed inference."""
    collected = rcs.collect(_comparison(), _trial(), {}, "cv27", 100)
    summary = collected[ASRBN]["summary"]
    assert "inference" in summary["unmeasured"]
    assert summary["median_reduction"]["inference"] is None


def test_a_missing_trial_metrics_file_still_leaves_linkability_measured():
    """The sweep carries linkability readings of its own."""
    collected = rcs.collect(_comparison(), {}, _inference(), "cv27", 100)
    summary = collected[ASRBN]["summary"]
    assert summary["readings_per_criterion"]["linkability"] > 0


# Directions and magnitudes, on the numbers the project actually measured.

def test_mcadams_removes_nothing_and_raises_inference():
    """The measured CV27 result: the sweep readings do not move at all, the
    trial-score metrics move a little, and the inference advantage goes the
    wrong way.

    The linkability criterion therefore reads as slightly addressed, which is
    correct and is why the median over its seven readings is reported rather
    than any single one. Singling out and inference are the two that are not
    addressed at all.
    """
    body = _collect()["mcadams"]
    link = next(r for m, r in body["rows"] if m == "linkability")
    assert link.reduction == pytest.approx(0.0)
    inference = [r for m, r in body["rows"] if r.criterion == "inference"]
    assert all(r.reduction < 0 for r in inference)
    summary = body["summary"]
    assert summary["unaddressed"] == ["singling_out", "inference"]
    # Small, and from the trial-score metrics alone.
    assert 0.0 < summary["median_reduction"]["linkability"] < 0.15


def test_a_system_that_moves_verification_but_not_inference_shows_spread():
    body = _collect()[ASRBN]
    summary = body["summary"]
    assert summary["median_reduction"]["linkability"] > 0.4
    assert summary["median_reduction"]["inference"] < 0.2
    assert summary["spread"] > 0.3


def test_the_collapsed_singling_out_room_is_marked_at_the_full_population():
    body = _collect(9779)[ASRBN]
    singling = next(r for m, r in body["rows"] if m == "singling_out")
    assert singling.room == pytest.approx(0.02)
    assert not singling.reliable
    assert "singling_out" in body["summary"]["unmeasured"]


def test_singling_out_is_still_readable_in_the_usable_window():
    body = _collect(100)[ASRBN]
    singling = next(r for m, r in body["rows"] if m == "singling_out")
    assert singling.reliable
    assert body["summary"]["readings_per_criterion"]["singling_out"] == 1


# The available-counts helper, which chooses the default populations.

def test_available_counts_are_those_every_system_shares():
    counts = rcs._available_counts(_comparison(), "cv27")
    assert counts == [100, 9779]


def test_available_counts_ignore_other_corpora():
    assert rcs._available_counts(_comparison(), "libri_test") == [29]


def test_an_absent_corpus_has_no_counts():
    assert rcs._available_counts(_comparison(), "nope") == []


# End to end, because the printing is where a format string can raise.

def test_the_whole_driver_runs_and_writes_the_counts_it_reported(tmp_path):
    comparison = tmp_path / "c.json"
    trial = tmp_path / "t.json"
    inference = tmp_path / "i.json"
    out = tmp_path / "split.json"
    comparison.write_text(json.dumps(_comparison()))
    trial.write_text(json.dumps(_trial()))
    inference.write_text(json.dumps(_inference()))
    code = rcs.main(["--comparison", str(comparison),
                     "--trial-metrics", str(trial),
                     "--inference", str(inference),
                     "--base", "cv27", "--output", str(out)])
    assert code == 0
    written = json.loads(out.read_text())
    # The default picks the usable window and the full population.
    assert sorted(written) == ["100", "9779"]
    assert sorted(written["100"]) == sorted(["mcadams", ASRBN])


def test_an_unknown_corpus_exits_non_zero(tmp_path, capsys):
    comparison = tmp_path / "c.json"
    comparison.write_text(json.dumps(_comparison()))
    empty = tmp_path / "e.json"
    empty.write_text("{}")
    code = rcs.main(["--comparison", str(comparison), "--trial-metrics",
                     str(empty), "--inference", str(empty), "--base", "nope"])
    assert code == 1
