"""ROCCH-EER correctness.

The implementation is independent of BOSARIS, so it is pinned here against
properties and against a brute-force search over randomised decision rules.
"""

import numpy as np
import pytest
from scipy.stats import norm

from legal_eval.metrics.eer import one_minus_eer, roc_convex_hull, rocch_eer


def brute_force_rocch_eer(target, nontarget, n_grid=2001):
    """ROCCH-EER by direct search, used as an oracle.

    Builds every single-threshold operating point, then takes the best achievable
    Pmiss for a grid of Pfa values by mixing pairs of thresholds (which is what
    the convex hull expresses), and reports where the two error rates meet.
    """
    target = np.asarray(target, float)
    nontarget = np.asarray(nontarget, float)
    thresholds = np.unique(np.concatenate([target, nontarget, [-np.inf, np.inf]]))
    pmiss = np.array([(target < t).mean() for t in thresholds])
    pfa = np.array([(nontarget >= t).mean() for t in thresholds])

    grid = np.linspace(0.0, 1.0, n_grid)
    best = np.ones_like(grid)
    # Every mixture of two operating points is achievable, so sweep all pairs.
    for i in range(len(thresholds)):
        for j in range(len(thresholds)):
            lo, hi = pfa[i], pfa[j]
            if lo == hi:
                reachable = np.isclose(grid, lo)
                candidate = np.where(reachable, min(pmiss[i], pmiss[j]), np.inf)
            else:
                weight = (grid - lo) / (hi - lo)
                inside = (weight >= 0) & (weight <= 1)
                candidate = np.where(
                    inside, pmiss[i] + weight * (pmiss[j] - pmiss[i]), np.inf
                )
            best = np.minimum(best, candidate)
    difference = best - grid
    crossing = np.argmin(np.abs(difference))
    return float(grid[crossing])


def test_identical_distributions_give_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=4000)
    assert rocch_eer(scores[:2000], scores[2000:]) == pytest.approx(0.5, abs=0.02)


def test_perfect_separation_gives_zero():
    assert rocch_eer(np.array([5.0, 6.0, 7.0]), np.array([0.0, 1.0, 2.0])) == 0.0


def test_matches_analytic_eer_for_separated_gaussians():
    # For equal-variance Gaussians separated by d, the EER is Phi(-d/2).
    rng = np.random.default_rng(1)
    separation = 2.0
    target = rng.normal(loc=separation, scale=1.0, size=200_000)
    nontarget = rng.normal(loc=0.0, scale=1.0, size=200_000)
    expected = norm.cdf(-separation / 2.0)
    assert rocch_eer(target, nontarget) == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force_oracle(seed):
    rng = np.random.default_rng(seed)
    target = rng.normal(loc=1.0, size=12)
    nontarget = rng.normal(loc=0.0, size=15)
    assert rocch_eer(target, nontarget) == pytest.approx(
        brute_force_rocch_eer(target, nontarget), abs=1e-3
    )


def test_rocch_eer_never_exceeds_empirical_eer():
    # The convex hull can only improve on single-threshold operating points.
    rng = np.random.default_rng(2)
    for _ in range(20):
        target = rng.normal(loc=1.0, size=60)
        nontarget = rng.normal(loc=0.0, size=60)
        thresholds = np.unique(np.concatenate([target, nontarget]))
        pmiss = np.array([(target < t).mean() for t in thresholds])
        pfa = np.array([(nontarget >= t).mean() for t in thresholds])
        empirical = np.max(np.minimum(pmiss, pfa))
        assert rocch_eer(target, nontarget) <= empirical + 1e-9


def test_hull_is_monotone_and_spans_the_pfa_range():
    rng = np.random.default_rng(3)
    pfa, pmiss = roc_convex_hull(rng.normal(loc=1.0, size=50), rng.normal(size=50))
    assert np.all(np.diff(pfa) > 0)
    assert np.all(np.diff(pmiss) <= 1e-12)
    # The hull keeps only non-dominated operating points, so it starts at the
    # lowest Pmiss reachable with no false alarms rather than at (0, 1).
    assert pfa[0] == pytest.approx(0.0)
    assert pfa[-1] == pytest.approx(1.0)
    assert pmiss[-1] == pytest.approx(0.0)
    assert 0.0 < pmiss[0] <= 1.0


def test_hull_brackets_the_eer_crossing():
    # Pmiss - Pfa must start positive and end negative for the EER to exist.
    rng = np.random.default_rng(7)
    pfa, pmiss = roc_convex_hull(rng.normal(loc=1.2, size=70), rng.normal(size=70))
    assert pmiss[0] - pfa[0] > 0
    assert pmiss[-1] - pfa[-1] < 0


def test_hull_is_convex():
    rng = np.random.default_rng(4)
    pfa, pmiss = roc_convex_hull(rng.normal(loc=0.8, size=80), rng.normal(size=80))
    slopes = np.diff(pmiss) / np.diff(pfa)
    assert np.all(np.diff(slopes) >= -1e-9)


def test_one_minus_eer_is_complement():
    rng = np.random.default_rng(5)
    target, nontarget = rng.normal(loc=1.0, size=40), rng.normal(size=40)
    assert one_minus_eer(target, nontarget) == pytest.approx(
        1.0 - rocch_eer(target, nontarget)
    )


def test_rejects_empty_and_non_finite_input():
    with pytest.raises(ValueError):
        rocch_eer(np.array([]), np.array([1.0]))
    with pytest.raises(ValueError):
        rocch_eer(np.array([np.nan]), np.array([1.0]))
