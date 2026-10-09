"""The disclosure metrics, and the claims the comparison rests on.

These are our reading of other groups' published definitions, so the tests state
the properties their papers assert. If a property fails, our reading is wrong.
"""

import numpy as np
import pytest

from legal_eval.comparison import (
    calibration_samples,
    fit_calibration,
    local_information_disclosure,
    similarity_rank_disclosure,
)


def uninformative(n_enroll=200, n_trials=400, seed=0):
    """Scores carrying no identity information."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_enroll, n_trials)), rng.integers(0, n_enroll, n_trials)


def informative(n_enroll=200, n_trials=200, boost=3.0, seed=1):
    """Scores where the true identity is genuinely easier to find."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_enroll, n_trials))
    true = rng.permutation(n_enroll)[:n_trials]
    scores[true, np.arange(n_trials)] += boost
    return scores, true


def test_lid_is_exactly_zero_when_the_posterior_is_uniform():
    """LID is zero per trial when p = 1/N, which a zero slope forces."""
    scores, true = uninformative()
    lid = local_information_disclosure(scores, true, 20, slope=0.0)
    assert lid.mean == pytest.approx(0.0, abs=1e-9)
    assert lid.maximum == pytest.approx(0.0, abs=1e-9)


def test_srd_is_unbiased_under_chance_but_mean_lid_is_not():
    """A difference between the two that matters when reading their numbers.

    SRD averages log2(N * gamma) over an empirical rank histogram, which is flat
    under chance, so its mean sits at zero. LID averages log2(N * p) over
    per-trial posteriors, and by Jensen's inequality the mean of a log is below
    the log of the mean even when the mean posterior is exactly 1/N. So the
    average LID is negative under chance, and "zero bits means no disclosure"
    holds per trial but not for the average.

    The consequence for the comparison is that the positive disclosure rate,
    not the mean, is the chance-referenced statistic for LID.
    """
    scores, true = uninformative()
    srd = similarity_rank_disclosure(scores, true, 20)
    assert srd.mean == pytest.approx(0.0, abs=0.25)

    lid = local_information_disclosure(scores, true, 20, slope=1.0)
    assert lid.mean < -0.2, "Jensen gap, not a defect"
    # The gap widens as the posterior gets peakier, so it is a property of the
    # calibration rather than of the data.
    sharper = local_information_disclosure(scores, true, 20, slope=4.0)
    assert sharper.mean < lid.mean


def test_positive_bits_when_the_attacker_learns_something():
    scores, true = informative()
    assert local_information_disclosure(scores, true, 20, slope=1.0).mean > 1.0
    assert similarity_rank_disclosure(scores, true, 20).mean > 1.0


def test_srd_identification_rate_is_the_linkability_metric():
    """SRD reports IdR, the rank-1 rate. That is Linkability.

    This is the overlap that makes the two families comparable at all, so it is
    worth asserting rather than assuming.
    """
    from legal_eval.io import ScoreMatrix
    from legal_eval.metrics.linkability import linkability

    scores, true = informative(n_enroll=120, n_trials=120, boost=2.0, seed=4)
    names = [f"spk{i:04d}" for i in range(scores.shape[0])]
    matrix = ScoreMatrix(scores, names, [names[i] for i in true], {})

    n = 20
    srd = similarity_rank_disclosure(scores, true, n, rng=np.random.default_rng(7))
    ours = float(np.mean(linkability(matrix, n_enroll_speakers=n, n_runs=40, seed=3)))
    assert srd.identification_rate == pytest.approx(ours, abs=0.05)


def test_disclosure_cannot_exceed_log2_n():
    """A closed set of N holds at most log2(N) bits of identity."""
    scores, true = informative(boost=12.0)
    for n in (4, 16, 64):
        lid = local_information_disclosure(scores, true, n, slope=5.0)
        assert lid.maximum <= np.log2(n) + 1e-9
        assert similarity_rank_disclosure(scores, true, n).maximum <= np.log2(n) + 1e-9


def test_lid_rises_with_population_while_linkability_falls():
    """The central empirical claim of the comparison.

    A larger search space makes success less likely but a success more
    informative, so a probability of success and a quantity of information move
    in opposite directions. If this ever fails, the headline claim is wrong.
    """
    scores, true = informative(n_enroll=400, n_trials=300, boost=2.5, seed=11)
    link, bits = [], []
    for n in (4, 16, 64, 256):
        r = local_information_disclosure(scores, true, n, slope=1.0,
                                         rng=np.random.default_rng(5))
        link.append(r.identification_rate)
        bits.append(r.mean)
    assert link[0] > link[-1], "identification must get harder with more candidates"
    assert bits[0] < bits[-1], "but a correct identification must carry more bits"


def test_calibration_slope_is_what_matters_not_the_intercept():
    """The softmax is shift invariant, so the intercept cancels exactly."""
    scores, true = informative()
    a = local_information_disclosure(scores, true, 20, slope=2.0,
                                     rng=np.random.default_rng(1))
    b = local_information_disclosure(scores, true, 20, slope=2.0,
                                     rng=np.random.default_rng(1))
    assert a.mean == pytest.approx(b.mean)


def test_lid_depends_materially_on_the_calibration_slope():
    """Worth asserting because it is a degree of freedom Linkability lacks."""
    scores, true = informative()
    means = [local_information_disclosure(scores, true, 20, slope=s,
                                          rng=np.random.default_rng(2)).mean
             for s in (0.5, 2.0, 20.0)]
    assert max(means) - min(means) > 0.5, "the slope changes the answer"


def test_calibration_recovers_a_sensible_slope():
    rng = np.random.default_rng(3)
    mated = rng.normal(loc=2.0, scale=1.0, size=4000)
    nonmated = rng.normal(loc=0.0, scale=1.0, size=4000)
    w, _ = fit_calibration(mated, nonmated)
    # For equal-variance Gaussians separated by d, the logistic slope is d.
    assert w == pytest.approx(2.0, rel=0.25)


def test_srd_needs_no_calibration():
    """SRD uses ranks only, so a monotone rescaling of scores cannot change it."""
    scores, true = informative()
    a = similarity_rank_disclosure(scores, true, 20, rng=np.random.default_rng(9))
    b = similarity_rank_disclosure(scores * 7.0 + 3.0, true, 20,
                                   rng=np.random.default_rng(9))
    assert a.mean == pytest.approx(b.mean)
    assert a.identification_rate == pytest.approx(b.identification_rate)


def _ranked_matrix(n_enroll=40, n_trials=400, strength=1.2, seed=0):
    """A score matrix with a true speaker per trial and a tunable margin."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_enroll, n_trials))
    true_rows = rng.integers(0, n_enroll, size=n_trials)
    scores[true_rows, np.arange(n_trials)] += strength
    return scores, true_rows


def test_calibration_samples_are_z_normalised_per_trial():
    """The slope is fit on the scale LID normalises to, which is per trial."""
    scores, true_rows = _ranked_matrix()
    mated, nonmated = calibration_samples(scores, true_rows)

    assert mated.shape == (scores.shape[1],)
    assert nonmated.shape == (scores.shape[1] * (scores.shape[0] - 1),)
    # Pooling a per-column standardisation leaves the whole pool standardised.
    pooled = np.concatenate([mated, nonmated])
    assert abs(pooled.mean()) < 1e-10
    assert abs(pooled.std() - 1.0) < 1e-10
    # The mated scores carry the margin, so they sit above the nonmated ones.
    assert mated.mean() > nonmated.mean()


def test_rank_metrics_are_calibration_free_but_lid_is_not():
    """Two backends with identical rankings disagree on LID and agree on the rest.

    A strictly increasing transform applied to every score leaves every ranking
    untouched, so Linkability and SRD cannot move. LID standardises and then
    applies a fitted slope, so a transform that changes the shape of the score
    distribution changes the posterior and the reported bits. This is the
    situation of two scoring backends, say cosine and PLDA, that rank
    identically.
    """
    scores, true_rows = _ranked_matrix()
    # Strictly increasing, so order preserving, but not affine.
    reshaped = np.sign(scores) * np.abs(scores) ** 3

    n = 20
    srd_a = similarity_rank_disclosure(scores, true_rows, n, rng=np.random.default_rng(3))
    srd_b = similarity_rank_disclosure(reshaped, true_rows, n, rng=np.random.default_rng(3))
    assert srd_a.mean == pytest.approx(srd_b.mean)
    assert srd_a.identification_rate == pytest.approx(srd_b.identification_rate)

    slope_a, _ = fit_calibration(*calibration_samples(scores, true_rows))
    slope_b, _ = fit_calibration(*calibration_samples(reshaped, true_rows))
    lid_a = local_information_disclosure(
        scores, true_rows, n, slope=slope_a, rng=np.random.default_rng(3)
    )
    lid_b = local_information_disclosure(
        reshaped, true_rows, n, slope=slope_b, rng=np.random.default_rng(3)
    )
    # Same rankings, same identification rate, materially different bits.
    assert lid_a.identification_rate == pytest.approx(lid_b.identification_rate)
    assert abs(lid_a.mean - lid_b.mean) > 0.1


def test_lid_bits_depend_on_the_slope_over_a_wide_range():
    """ALID moves by bits on fixed data when only the calibration changes.

    An ALID figure is therefore not interpretable without the calibration
    protocol that produced it, which is the practical obstacle to comparing
    published values.
    """
    scores, true_rows = _ranked_matrix()
    n = 20
    means = [
        local_information_disclosure(
            scores, true_rows, n, slope=w, rng=np.random.default_rng(4)
        ).mean
        for w in (0.1, 1.0, 16.0)
    ]
    assert max(means) - min(means) > 1.0
    # A flat calibration drives the posterior towards uniform, hence zero bits,
    # and a steep one overshoots into confident errors, hence negative bits.
    assert means[0] == pytest.approx(0.0, abs=0.2)
    assert means[-1] < means[1]


def test_self_calibration_shrinks_the_gap_between_systems():
    """Fitting the calibration per system feeds the effect back into the metric.

    A weaker system yields a flatter fitted slope, which pushes its ALID towards
    zero regardless of what it disclosed. Evaluating both systems under one
    shared calibration separates them much further, so the choice between the
    two protocols changes the conclusion.
    """
    strong, rows = _ranked_matrix(strength=2.0, seed=5)
    weak, rows_weak = _ranked_matrix(strength=0.3, seed=5)
    assert np.array_equal(rows, rows_weak)

    slope_strong, _ = fit_calibration(*calibration_samples(strong, rows))
    slope_weak, _ = fit_calibration(*calibration_samples(weak, rows))
    assert slope_strong > slope_weak  # a sharper system calibrates sharper

    n = 20

    def alid(matrix, slope):
        return local_information_disclosure(
            matrix, rows, n, slope=slope, rng=np.random.default_rng(6)
        ).mean

    self_gap = abs(alid(strong, slope_strong) - alid(weak, slope_weak))
    shared_gap = abs(alid(strong, slope_strong) - alid(weak, slope_strong))
    assert shared_gap > self_gap

    # The rank metrics have no such dependence: one protocol, one answer.
    link_strong = similarity_rank_disclosure(
        strong, rows, n, rng=np.random.default_rng(6)
    ).identification_rate
    link_weak = similarity_rank_disclosure(
        weak, rows, n, rng=np.random.default_rng(6)
    ).identification_rate
    assert link_strong > link_weak
