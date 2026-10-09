r"""Comparing the recent privacy metrics on a common footing.

Four proposals have replaced the equal error rate as the measure of residual
re-identification risk, and none has been run against the others on the same
data. This module computes them from one artifact so that any difference is the
metric rather than the pipeline.

The unifying observation
------------------------
Three of them are functionals of the same object: the attacker's probability
that a trial belongs to the true identity, within a closed set of ``N``
enrolled speakers. Writing that probability ``q``, each reports a form of

.. math:: \log_2(N q)

which is zero when ``q = 1/N``, that is when the attacker has learned nothing.
They differ in **how ``q`` is estimated** and **how it is summarised**:

===================  ==========================  ==========================
Metric               ``q`` is                    summarised as
===================  ==========================  ==========================
Linkability          empirical, rank 1 only      probability the true
                                                 identity ranks first
SRD                  empirical rank histogram    mean and max of
                                                 :math:`\log_2(N q)`, plus
                                                 the rank-1 rate
LID                  per-trial calibrated        mean, max, and the share
                     posterior over scores       with :math:`\log_2(N q)>0`
===================  ==========================  ==========================

Two consequences worth stating. SRD's "identification rate", the probability
that the true reference ranks first, **is** the Linkability metric, so the two
families already overlap in a published statistic. And Linkability is a
zero-one functional of the ranking while LID and SRD retain the margin, so they
can and do move in opposite directions as ``N`` grows: a larger search space
makes success less likely while making a success more informative.

Singling Out is not of this family. It asks whether a calibrated predicate
isolates exactly one record from a set, which is a different legal criterion,
and its baseline is ``exp(-1)`` rather than ``1/N``.

Provenance
----------
The LID and SRD implementations here are **our reading of the published
definitions**, written so that a comparison can be run without asking their
authors to do the work. They are not authoritative, and anywhere our reading is
wrong the comparison is wrong. They are deliberately small and commented so that
checking them is a short task rather than a reimplementation.

* LID: Bäckström et al., "Goodbye Equal Error Rate, Hello Local Information
  Disclosure", arXiv:2607.06259.
* SRD: Chandra, Pettenò, Evans, Panariello, Todisco, Bäckström, Kolossa,
  Martin, Stafylakis, Gengembre, arXiv:2605.07291.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DisclosureResult:
    """Aggregates reported by an information-theoretic disclosure metric.

    Attributes:
        mean: Mean disclosure in bits. ``ALID`` for LID, ``MeanD`` for SRD.
        maximum: Worst-case disclosure in bits.
        positive_rate: Share of trials with disclosure above zero, which is the
            share where the attacker learned more than nothing. ``PDR`` for LID.
        identification_rate: Share of trials where the true identity ranks
            first. This is the Linkability metric, reported here because SRD
            reports it too.
        per_trial: Disclosure per trial, for distribution analysis.
    """

    mean: float
    maximum: float
    positive_rate: float
    identification_rate: float
    per_trial: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def _closed_set_scores(
    scores: np.ndarray,
    true_rows: np.ndarray,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a closed set of ``n_candidates`` per trial, true identity first.

    Returns an array of shape ``(n_trials, n_candidates)`` whose column 0 holds
    the true identity's score. Every metric here operates on such a set, so
    drawing it once keeps them strictly comparable.
    """
    n_enroll, n_trials = scores.shape
    out = np.empty((n_trials, n_candidates), dtype=np.float64)
    for column in range(n_trials):
        true = int(true_rows[column])
        others = rng.choice(n_enroll - 1, size=n_candidates - 1, replace=False)
        others = np.where(others >= true, others + 1, others)
        out[column, 0] = scores[true, column]
        out[column, 1:] = scores[others, column]
    return out


def calibration_samples(
    scores: np.ndarray, true_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mated and nonmated samples on the scale LID calibrates on.

    LID z-normalises each trial's scores across the enrolled identities and
    only then fits the Platt mapping, so the slope is meaningful relative to
    that normalisation. Fitting on raw scores instead produces a slope on the
    wrong scale, and because the slope is the only thing that survives into the
    posterior that error passes straight through to the reported bits.

    Args:
        scores: Enrollment-by-trial score matrix.
        true_rows: Row index of the true identity for each trial.

    Returns:
        ``(mated, nonmated)`` z-normalised scores, pooled over trials.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n_enroll, n_trials = scores.shape
    mean = scores.mean(axis=0, keepdims=True)
    std = scores.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    normalised = (scores - mean) / std

    columns = np.arange(n_trials)
    mated = normalised[true_rows, columns]
    mask = np.ones_like(normalised, dtype=bool)
    mask[true_rows, columns] = False
    return mated, normalised[mask]


def fit_calibration(
    mated: np.ndarray, nonmated: np.ndarray
) -> tuple[float, float]:
    """Fit the logistic slope and intercept LID uses to turn scores into LLRs.

    LID maps z-normalised scores to log-likelihood ratios by Platt scaling,
    ``LLR = w * s_hat + b``, with ``w`` and ``b`` from univariate logistic
    regression on development trials.

    Only ``w`` affects the result. The softmax that follows is shift invariant,
    so ``b`` and the prior-ratio term cancel exactly, leaving ``w`` as an
    inverse temperature. We fit both and return them, but the comparison depends
    on ``w`` alone, which is worth knowing when judging how sensitive LID is to
    its calibration.

    Args:
        mated: Scores of same-speaker development trials.
        nonmated: Scores of different-speaker development trials.

    Returns:
        ``(w, b)``.
    """
    x = np.concatenate([np.asarray(mated, float), np.asarray(nonmated, float)])
    y = np.concatenate([np.ones(len(mated)), np.zeros(len(nonmated))])
    # Newton steps on the univariate logistic likelihood. Small and dependency
    # free, so this file can be checked without reading a library.
    w, b = 1.0, 0.0
    for _ in range(100):
        z = w * x + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        g = np.array([np.sum((p - y) * x), np.sum(p - y)])
        s = p * (1.0 - p)
        h = np.array([[np.sum(s * x * x), np.sum(s * x)],
                      [np.sum(s * x), np.sum(s)]])
        h[0, 0] += 1e-9
        h[1, 1] += 1e-9
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        w -= step[0]
        b -= step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w), float(b)


def local_information_disclosure(
    scores: np.ndarray,
    true_rows: np.ndarray,
    n_candidates: int,
    slope: float,
    rng: np.random.Generator | None = None,
) -> DisclosureResult:
    r"""Local Information Disclosure, our reading of arXiv:2607.06259.

    Per trial: z-normalise the scores across the ``N`` enrolled identities, map
    them to log-likelihood ratios by Platt scaling, take a softmax to obtain a
    posterior, and report :math:`\log_2(N p)` in bits for the true identity.

    Args:
        scores: Enrollment-by-trial score matrix.
        true_rows: Row index of the true identity for each trial.
        n_candidates: Closed set size ``N``.
        slope: The logistic slope ``w`` from :func:`fit_calibration`. The
            intercept is omitted because the softmax cancels it.
        rng: Generator for the closed-set draw.

    Returns:
        The :class:`DisclosureResult`.
    """
    rng = rng or np.random.default_rng(0)
    candidate = _closed_set_scores(scores, true_rows, n_candidates, rng)

    mean = candidate.mean(axis=1, keepdims=True)
    std = candidate.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    normalised = (candidate - mean) / std

    llr = slope * normalised
    llr -= llr.max(axis=1, keepdims=True)  # stability; softmax is shift invariant
    weights = np.exp(llr)
    posterior = weights / weights.sum(axis=1, keepdims=True)

    p_true = np.clip(posterior[:, 0], 1e-300, 1.0)
    bits = np.log2(n_candidates * p_true)
    return DisclosureResult(
        mean=float(bits.mean()),
        maximum=float(bits.max()),
        positive_rate=float(np.mean(bits > 0)),
        identification_rate=float(np.mean(candidate[:, 0] >= candidate[:, 1:].max(axis=1))),
        per_trial=bits,
    )


def similarity_rank_disclosure(
    scores: np.ndarray,
    true_rows: np.ndarray,
    n_candidates: int,
    rng: np.random.Generator | None = None,
) -> DisclosureResult:
    r"""Similarity Rank Disclosure, our reading of arXiv:2605.07291.

    Rank the ``N`` enrolled identities by similarity to each trial, record the
    rank of the true one, and build the empirical distribution of that rank over
    trials. Disclosure at rank ``j`` is :math:`\log_2(N \tilde p_j)`, the
    reduction from the :math:`\log_2 N` bits a uniform prior would need.

    Unlike LID this needs no calibration: it uses the ranking only, which is why
    it is described as threshold independent. The cost is that the probability
    is a population statistic rather than a per-trial one, so every trial whose
    true identity lands at the same rank is assigned the same disclosure.

    Args:
        scores: Enrollment-by-trial score matrix.
        true_rows: Row index of the true identity for each trial.
        n_candidates: Closed set size ``N``.
        rng: Generator for the closed-set draw.

    Returns:
        The :class:`DisclosureResult`, where ``identification_rate`` is the
        paper's ``IdR``, which is the Linkability metric.
    """
    rng = rng or np.random.default_rng(0)
    candidate = _closed_set_scores(scores, true_rows, n_candidates, rng)

    # Rank 1 is the most similar. The true identity sits in column 0.
    better = (candidate[:, 1:] > candidate[:, :1]).sum(axis=1)
    rank = better + 1

    histogram = np.bincount(rank, minlength=n_candidates + 1)[1:]
    gamma = histogram / histogram.sum()

    # Ranks never observed carry no trials, so they contribute nothing.
    bits_by_rank = np.full(n_candidates, -np.inf)
    seen = gamma > 0
    bits_by_rank[seen] = np.log2(n_candidates * gamma[seen])
    per_trial = bits_by_rank[rank - 1]

    return DisclosureResult(
        mean=float(per_trial.mean()),
        maximum=float(bits_by_rank[seen].max()),
        positive_rate=float(np.mean(per_trial > 0)),
        identification_rate=float(gamma[0]),
        per_trial=per_trial,
    )
