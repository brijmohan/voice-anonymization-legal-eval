"""Tests for putting the three Article 29 criteria on one scale.

Everything here is pinned against a value that is known without running the
code: a system that changed nothing removes none of the room, a system at
chance removes all of it, and the two inverted families invert. The inversion
tests matter most, because getting the direction wrong does not blur a
conclusion, it reverses one.
"""

from __future__ import annotations

import math

import pytest

from legal_eval.criteria import (
    CRITERIA,
    CRITERION_OF_METRIC,
    FIXED_CHANCE,
    MIN_ROOM,
    chance_for,
    criteria_split,
    exposure,
    read_criterion,
)

# Chance levels, which are what make the metrics comparable at all.

def test_singling_out_chance_does_not_depend_on_the_population():
    """The one metric whose chance level is a constant, which is why its room
    collapses as N grows while Linkability's does not."""
    assert chance_for("singling_out", 2) == pytest.approx(math.exp(-1))
    assert chance_for("singling_out", 9779) == pytest.approx(math.exp(-1))


def test_population_metrics_have_chance_one_over_n():
    for metric in ("linkability", "srd_idr", "lid_pdr"):
        assert chance_for(metric, 100) == pytest.approx(0.01)


def test_a_population_metric_without_n_is_refused():
    """Returning a default would silently compare against the wrong chance."""
    with pytest.raises(ValueError, match="n is required"):
        chance_for("linkability")


def test_fixed_chance_metrics_need_no_population():
    assert chance_for("eer") == 0.5
    assert chance_for("cllr_min") == 1.0
    assert chance_for("inference_advantage") == 0.0


def test_an_unknown_metric_is_refused():
    with pytest.raises(ValueError, match="no chance level"):
        chance_for("made_up_metric", 100)


# Exposure, including the two inverted families.

def test_exposure_is_zero_at_chance():
    assert exposure("linkability", 0.01, 100) == pytest.approx(0.0)
    assert exposure("singling_out", math.exp(-1)) == pytest.approx(0.0)
    assert exposure("inference_advantage", 0.0) == pytest.approx(0.0)


def test_eer_exposure_runs_the_other_way():
    """A high EER is privacy, so it must produce a low exposure."""
    assert exposure("eer", 0.5) == pytest.approx(0.0)
    assert exposure("eer", 0.0) == pytest.approx(0.5)
    assert exposure("eer", 0.067) == pytest.approx(0.433)


def test_cllr_exposure_runs_the_other_way():
    """One bit is an attacker who learned nothing; less is an attacker who did."""
    assert exposure("cllr_min", 1.0) == pytest.approx(0.0)
    assert exposure("cllr_min", 0.25) == pytest.approx(0.75)


def test_a_reading_below_chance_is_kept_rather_than_clipped():
    """Clipping would hide a system that reads below chance for a bad reason."""
    assert exposure("linkability", 0.005, 100) == pytest.approx(-0.005)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_measurement_is_refused(bad):
    with pytest.raises(ValueError, match="not a measurement"):
        exposure("linkability", bad, 100)


# The reduction, which is the quantity the report is built on.

def test_a_system_that_changed_nothing_removes_none_of_the_room():
    reading = read_criterion("linkability", original=0.9131, system=0.9131, n=29)
    assert reading.reduction == pytest.approx(0.0)
    assert reading.reliable


def test_a_system_at_chance_removes_all_of_the_room():
    reading = read_criterion("linkability", original=0.9131, system=1 / 29, n=29)
    assert reading.reduction == pytest.approx(1.0)


def test_half_the_room_reads_as_half():
    chance = 1 / 100
    original = chance + 0.8
    reading = read_criterion("linkability", original=original,
                             system=chance + 0.4, n=100)
    assert reading.reduction == pytest.approx(0.5)


def test_an_anonymized_copy_that_discloses_more_reads_negative():
    """McAdams raised the inference advantage on CV27, and a report that
    rendered that as zero would be hiding the result."""
    reading = read_criterion("inference_advantage",
                             original=0.6859, system=0.6919)
    assert reading.reduction < 0


def test_eer_reduction_respects_the_inversion():
    """EER 0.067 to 0.5 is the whole room removed, not a rise in exposure."""
    reading = read_criterion("eer", original=0.067, system=0.5)
    assert reading.reduction == pytest.approx(1.0)


def test_a_collapsed_room_is_marked_unreliable():
    """The Singling Out room at N=9779 is 0.0247, below the threshold."""
    original = math.exp(-1) + 0.0247
    reading = read_criterion("singling_out", original=original,
                             system=math.exp(-1) + 0.0005, n=9779)
    assert reading.room < MIN_ROOM
    assert not reading.reliable


def test_room_exactly_at_the_threshold_is_reliable():
    """A boundary that moves with a rewrite is a boundary that is not pinned."""
    reading = read_criterion("linkability", original=1 / 100 + MIN_ROOM,
                             system=1 / 100, n=100)
    assert reading.room == pytest.approx(MIN_ROOM)
    assert reading.reliable


def test_no_room_gives_no_reduction_rather_than_an_infinity():
    reading = read_criterion("linkability", original=1 / 29, system=1 / 29, n=29)
    assert reading.room == pytest.approx(0.0)
    assert reading.reduction == 0.0
    assert not reading.reliable


def test_a_metric_with_no_criterion_is_refused():
    """Silently assigning one would misattribute evidence to a criterion."""
    with pytest.raises(ValueError, match="Article 29"):
        read_criterion("srd_mean", original=0.8, system=0.4, n=100)


def test_every_mapped_metric_names_a_real_criterion():
    assert set(CRITERION_OF_METRIC.values()) <= set(CRITERIA)


def test_exactly_one_mapped_metric_speaks_to_inference():
    """This imbalance is the finding the module exists to expose. If a second
    inference metric is added this test should be updated deliberately, not
    discovered to have broken."""
    inference = [m for m, c in CRITERION_OF_METRIC.items() if c == "inference"]
    assert inference == ["inference_advantage"]


# The summary across criteria.

def _readings(link_reduction: float, inference_reduction: float):
    chance = 1 / 100
    room = 0.8
    return [
        read_criterion("linkability", original=chance + room,
                       system=chance + room * (1 - link_reduction), n=100),
        read_criterion("inference_advantage", original=room,
                       system=room * (1 - inference_reduction)),
    ]


def test_a_split_shows_up_as_spread():
    summary = criteria_split(_readings(link_reduction=0.6,
                                       inference_reduction=0.0))
    assert summary["median_reduction"]["linkability"] == pytest.approx(0.6)
    assert summary["median_reduction"]["inference"] == pytest.approx(0.0)
    assert summary["spread"] == pytest.approx(0.6)
    assert summary["unaddressed"] == ["inference"]


def test_treating_both_criteria_alike_shows_no_spread():
    summary = criteria_split(_readings(link_reduction=0.6,
                                       inference_reduction=0.6))
    assert summary["spread"] == pytest.approx(0.0)
    assert summary["unaddressed"] == []


def test_a_criterion_with_no_reading_is_reported_as_unmeasured():
    """A verification-only report, which is the common case in the field."""
    chance = 1 / 100
    summary = criteria_split([
        read_criterion("linkability", original=chance + 0.8,
                       system=chance + 0.3, n=100),
    ])
    assert summary["readings_per_criterion"]["inference"] == 0
    assert "inference" in summary["unmeasured"]
    assert "singling_out" in summary["unmeasured"]
    assert summary["spread"] is None


def test_unreliable_readings_do_not_reach_the_summary():
    original = math.exp(-1) + 0.001
    summary = criteria_split([
        read_criterion("singling_out", original=original,
                       system=math.exp(-1), n=9779),
    ])
    assert summary["readings_per_criterion"]["singling_out"] == 0
    assert "singling_out" in summary["unmeasured"]


def test_the_median_is_used_so_one_metric_cannot_carry_a_criterion():
    chance = 1 / 100
    rows = [
        read_criterion(m, original=chance + 0.8, system=chance + s, n=100)
        for m, s in (("linkability", 0.8), ("srd_idr", 0.4), ("lid_pdr", 0.0))
    ]
    # Reductions are 0.0, 0.5 and 1.0, so the median is 0.5 whichever order
    # they arrive in.
    summary = criteria_split(rows)
    assert summary["median_reduction"]["linkability"] == pytest.approx(0.5)


def test_fixed_chance_agrees_with_preflight_where_both_define_a_metric():
    """Two chance levels for one metric would let a report and its preflight
    disagree about what chance is."""
    from legal_eval.preflight import chance_levels
    levels = chance_levels(100)
    shared = set(FIXED_CHANCE) & set(levels)
    assert shared, "nothing shared, so this test stopped checking anything"
    for metric in shared:
        assert FIXED_CHANCE[metric] == pytest.approx(levels[metric]), metric


# Reading past chance, which the CV27 measurements forced.

def test_a_system_past_chance_is_flagged_rather_than_scored_above_100_percent():
    """ASR-BN reads 8.2 standard errors below the Singling Out chance level on
    CV27, which is systematic. Reporting that as 192% of the room removed
    would put arithmetic where an explanation belongs."""
    chance = math.exp(-1)
    reading = read_criterion("singling_out", original=chance + 0.10,
                             system=chance - 0.02)
    assert reading.beyond_chance
    assert reading.reduction > 1.0
    assert "at or beyond chance" in str(reading)
    assert "not meaningful" in str(reading)


def test_an_ordinary_reduction_is_not_flagged():
    chance = math.exp(-1)
    reading = read_criterion("singling_out", original=chance + 0.10,
                             system=chance + 0.04)
    assert not reading.beyond_chance
    assert "of the room removed" in str(reading)


def test_exactly_at_chance_is_not_past_it():
    """A boundary that drifts is a boundary that is not pinned."""
    chance = math.exp(-1)
    reading = read_criterion("singling_out", original=chance + 0.10,
                             system=chance)
    assert not reading.beyond_chance
    assert reading.reduction == pytest.approx(1.0)


def test_a_reading_past_chance_does_not_inflate_the_median():
    """It removed all of the room and then read further. Carrying 1.92 into an
    average of fractions would make a criterion look better than fully
    addressed, which is not a state that exists."""
    chance = math.exp(-1)
    past = read_criterion("singling_out", original=chance + 0.10,
                          system=chance - 0.09)
    assert past.reduction > 1.5
    summary = criteria_split([past])
    assert summary["median_reduction"]["singling_out"] == pytest.approx(1.0)
