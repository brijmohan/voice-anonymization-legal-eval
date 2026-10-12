"""Tests for Cllr, Gomez-Barrero linkability and ZEBRA.

These three differ from SRD and LID in one important way: the authors' own
reference implementations ship with the VoicePrivacy Challenge, so these are
their metrics rather than our reading of their papers.
`scripts/verify_trial_metrics.py` checks every function against that reference
and reports agreement to 3e-15 or better, with PAV and `D_sys` exact.

That check needs the challenge checked out, so it cannot run in CI. These tests
pin the same behaviour against theoretical baselines instead, which is the
standard this repository holds itself to: an uninformative attacker's Cllr is
exactly 1 bit, indistinguishable distributions give `D_sys` of zero, and
perfect privacy gives a ZEBRA profile of zeros.
"""

from __future__ import annotations

import numpy as np
import pytest

from legal_eval.trial_metrics import (
    ENFSI_TAG_BOUNDS,
    ZebraProfile,
    category_tag,
    cllr,
    gomez_barrero_linkability,
    min_cllr,
    optimal_llr,
    pav,
    zebra,
)

# Pool adjacent violators.

def test_pav_leaves_an_already_monotone_vector_alone():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    fitted, _, _ = pav(y)
    np.testing.assert_allclose(fitted, y)


def test_pav_averages_a_violating_pair():
    """The defining behaviour: a decrease is replaced by the pooled mean."""
    fitted, _, _ = pav(np.array([0.0, 1.0, 0.0, 1.0]))
    assert fitted[0] == 0.0
    np.testing.assert_allclose(fitted[1:3], [0.5, 0.5])
    assert fitted[3] == 1.0


def test_pav_output_is_always_non_decreasing():
    rng = np.random.default_rng(0)
    for _ in range(50):
        fitted, _, _ = pav(rng.normal(0, 1, int(rng.integers(2, 200))))
        assert np.all(np.diff(fitted) >= -1e-12)


def test_pav_blocks_sum_to_the_input_length():
    _, width, height = pav(np.array([0.0, 1.0, 0.0, 1.0, 1.0]))
    assert width.sum() == 5
    assert len(width) == len(height)


@pytest.mark.parametrize("bad,match", [
    (np.zeros((2, 2)), "1-D"),
    (np.array([]), "non-empty"),
])
def test_pav_refuses_bad_input(bad, match):
    with pytest.raises(ValueError, match=match):
        pav(bad)


# Cllr. The theoretical baseline is exact and worth pinning.

def test_cllr_of_an_uninformative_attacker_is_exactly_one_bit():
    """All LLRs zero means every posterior is 0.5, so the cost is 1 bit.

    This is the value to compare a reported Cllr against. Below 1 bit the
    attacker has learned something; at 1 bit it has not.
    """
    assert cllr(np.zeros(100), np.zeros(100)) == pytest.approx(1.0)


def test_cllr_falls_as_the_attacker_improves():
    rng = np.random.default_rng(0)
    weak = cllr(rng.normal(0.5, 1, 500), rng.normal(0.0, 1, 500))
    strong = cllr(rng.normal(6.0, 1, 500), rng.normal(-6.0, 1, 500))
    assert strong < weak < 1.5


def test_cllr_refuses_llrs_so_extreme_the_cost_is_unbounded():
    with pytest.raises(ValueError, match="unbounded"):
        cllr(np.array([-10000.0]), np.array([0.0]))


def test_min_cllr_is_never_worse_than_cllr_on_calibrated_scores():
    """Minimum Cllr is the cost after optimal calibration, so it is a floor."""
    rng = np.random.default_rng(1)
    mated = rng.normal(2.0, 1.0, 400)
    non = rng.normal(0.0, 1.0, 400)
    t_llr, n_llr = optimal_llr(mated, non)
    assert min_cllr(mated, non) <= cllr(t_llr, n_llr) + 1e-12


def test_min_cllr_of_indistinguishable_scores_is_about_one_bit():
    rng = np.random.default_rng(2)
    value = min_cllr(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000))
    assert 0.95 < value <= 1.01


# Gomez-Barrero D_sys. Not this package's Linkability.

def test_dsys_is_zero_when_the_distributions_are_indistinguishable():
    """Nothing links, so the measure is zero."""
    rng = np.random.default_rng(0)
    value = gomez_barrero_linkability(rng.normal(0, 1, 4000), rng.normal(0, 1, 4000))
    assert value == pytest.approx(0.0, abs=0.05)


def test_dsys_approaches_one_when_the_distributions_separate():
    rng = np.random.default_rng(0)
    value = gomez_barrero_linkability(rng.normal(20, 0.5, 4000),
                                      rng.normal(0, 0.5, 4000))
    assert value > 0.9


def test_dsys_rises_with_separation():
    rng = np.random.default_rng(0)
    non = rng.normal(0, 1, 4000)
    values = [gomez_barrero_linkability(rng.normal(d, 1, 4000), non)
              for d in (0.5, 2.0, 5.0)]
    assert values[0] < values[1] < values[2]


def test_dsys_refuses_too_few_mated_scores():
    """The reference histograms with len/10 bins, so under ten gives none."""
    with pytest.raises(ValueError, match="too few"):
        gomez_barrero_linkability(np.array([1.0, 2.0]), np.zeros(100))


def test_dsys_refuses_a_non_positive_prior_ratio():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="omega must be positive"):
        gomez_barrero_linkability(rng.normal(2, 1, 200), rng.normal(0, 1, 200),
                                  omega=0.0)


# ZEBRA: a population figure and a worst-case individual one.

def test_zebra_reports_perfect_privacy_for_indistinguishable_scores():
    rng = np.random.default_rng(0)
    profile = zebra(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000))
    assert profile.population == pytest.approx(0.0, abs=0.05)
    assert profile.worst_case < 1.0
    assert profile.tag in {"0", "A"}


def test_zebra_reports_disclosure_for_separable_scores():
    rng = np.random.default_rng(0)
    profile = zebra(rng.normal(8, 1, 2000), rng.normal(-8, 1, 2000))
    assert profile.population > 0.5
    assert profile.worst_case > 1.0
    assert profile.tag not in {"0", "A"}


def test_zebra_worst_case_is_finite_even_when_classes_separate_completely():
    """This is why the calibration uses Laplace padding.

    Without it the optimal LLRs are infinite on separable scores and the
    worst-case figure, being a maximum, would be infinite for every strongly
    separating system, which would make it useless for comparison.
    """
    profile = zebra(np.full(500, 50.0), np.full(500, -50.0))
    assert np.isfinite(profile.worst_case)
    assert np.isfinite(profile.population)


def test_zebra_str_shows_both_figures_and_the_tag():
    profile = ZebraProfile(population=0.25, worst_case=1.5, tag="B")
    assert str(profile) == "ZEBRA (0.250, 1.500, B)"


# The categorical scale, which is what makes a result legible.

@pytest.mark.parametrize("value,expected", [
    (0.0, "0"), (0.5, "A"), (1.0, "B"), (1.9, "B"),
    (2.0, "C"), (3.9, "C"), (4.0, "D"), (4.5, "D"),
    (5.0, "E"), (5.9, "E"), (6.0, "F"), (100.0, "F"),
])
def test_category_tag_boundaries(value, expected):
    assert category_tag(value) == expected


def test_category_tag_covers_every_value_without_a_gap():
    """A gap would silently drop a result out of the scale."""
    bounds = [(low, high) for _, low, high in ENFSI_TAG_BOUNDS]
    for (_, high), (low, _) in zip(bounds, bounds[1:], strict=False):
        assert high == low


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_category_tag_refuses_impossible_values(bad):
    with pytest.raises(ValueError):
        category_tag(bad)


# Shared input validation.

@pytest.mark.parametrize("fn", [optimal_llr, gomez_barrero_linkability, zebra])
def test_an_empty_score_set_is_refused(fn):
    with pytest.raises(ValueError):
        fn(np.array([]), np.zeros(100))


def test_optimal_llr_preserves_the_input_order():
    """A caller matches LLRs back to trials by position, so order matters."""
    rng = np.random.default_rng(0)
    mated = rng.normal(2, 1, 50)
    non = rng.normal(0, 1, 200)
    t_llr, n_llr = optimal_llr(mated, non)
    assert t_llr.shape == mated.shape
    assert n_llr.shape == non.shape

    # A higher score can never receive a lower calibrated LLR. Without Laplace
    # padding the extremes are infinite, and differencing infinities gives nan,
    # so monotonicity is checked on the finite part and the infinities are
    # checked for position instead.
    order = np.argsort(mated)
    ordered = t_llr[order]
    finite = np.isfinite(ordered)
    assert np.all(np.diff(ordered[finite]) >= -1e-9)
    if (~finite).any():
        # An infinite LLR may only sit at an end of the ordering, never inside
        # it, or the calibration is not monotone.
        positions = np.flatnonzero(~finite)
        assert np.all((positions == np.arange(len(positions)))
                      | (positions == np.arange(
                          len(ordered) - len(positions), len(ordered))))


def test_laplace_padding_removes_the_infinities():
    """Which is the whole reason ZEBRA calibrates with it."""
    rng = np.random.default_rng(0)
    mated = rng.normal(8, 1, 200)
    non = rng.normal(-8, 1, 200)
    t_raw, n_raw = optimal_llr(mated, non, laplace=False)
    t_pad, n_pad = optimal_llr(mated, non, laplace=True)
    assert not np.all(np.isfinite(np.concatenate([t_raw, n_raw])))
    assert np.all(np.isfinite(np.concatenate([t_pad, n_pad])))
