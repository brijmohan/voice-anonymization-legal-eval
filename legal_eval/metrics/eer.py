"""ROCCH-EER, the equal error rate read off the ROC convex hull.

The paper reports ``1 - EER`` as the conventional baseline that the two legal
metrics are compared against, using the ROCCH-EER rather than the raw empirical
EER. Taking the convex hull of the ROC removes the dependence on the particular
score quantisation and gives the error rate of the best calibrated decision rule
achievable from those scores, which is the right thing to compare a privacy
claim against.

This is an independent implementation. The BOSARIS toolkit, which the original
experiments used through Sidekit, is licensed for non-commercial use only and so
cannot be redistributed here. Correctness is pinned by the test suite against a
brute-force search over randomised decision rules and against the analytic EER
of separated Gaussians.

Definitions, for a threshold ``t``:

* ``Pmiss(t) = P(target score < t)``, the fraction of same-speaker trials rejected
* ``Pfa(t) = P(nontarget score >= t)``, the fraction of different-speaker trials accepted

Sweeping ``t`` traces the ROC in ``(Pfa, Pmiss)`` space from ``(1, 0)`` to
``(0, 1)``. Randomised rules that mix two thresholds reach any point on a segment
between them, so the achievable region is bounded by the lower-left convex hull
of those points. The ROCCH-EER is where that hull meets ``Pmiss = Pfa``.
"""

from __future__ import annotations

import numpy as np


def roc_convex_hull(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the lower-left convex hull of the ROC.

    Args:
        target_scores: Scores of same-speaker (mated) trials.
        nontarget_scores: Scores of different-speaker (nonmated) trials.

    Returns:
        Arrays ``(pfa, pmiss)`` of the hull vertices, ordered by increasing
        ``pfa``, running from ``(0, 1)`` to ``(1, 0)``.

    Raises:
        ValueError: If either score set is empty or contains non-finite values.
    """
    target = np.asarray(target_scores, dtype=np.float64).ravel()
    nontarget = np.asarray(nontarget_scores, dtype=np.float64).ravel()
    if target.size == 0 or nontarget.size == 0:
        raise ValueError("both target and nontarget scores are required")
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(nontarget))):
        raise ValueError("scores must be finite")

    n_target = target.size
    n_nontarget = nontarget.size

    scores = np.concatenate([target, nontarget])
    is_target = np.concatenate(
        [np.ones(n_target, dtype=bool), np.zeros(n_nontarget, dtype=bool)]
    )
    order = np.argsort(scores, kind="mergesort")
    is_target = is_target[order]

    # Sweeping the threshold upward past the first k sorted scores rejects every
    # target among them and accepts every nontarget above them.
    cumulative_targets = np.concatenate([[0], np.cumsum(is_target)])
    cumulative_nontargets = np.concatenate([[0], np.cumsum(~is_target)])
    pmiss = cumulative_targets / n_target
    pfa = (n_nontarget - cumulative_nontargets) / n_nontarget

    # Points come out with pfa decreasing; reverse so pfa increases.
    points = np.column_stack([pfa[::-1], pmiss[::-1]])
    hull = _lower_left_hull(points)
    return hull[:, 0], hull[:, 1]


def _lower_left_hull(points: np.ndarray) -> np.ndarray:
    """Lower hull of points sorted by increasing x, via Andrew's monotone chain.

    Keeps only vertices that are not above the segment joining their neighbours,
    which is exactly the set of operating points no randomised mixture improves
    upon.
    """
    order = np.lexsort((points[:, 1], points[:, 0]))
    ordered = points[order]

    hull: list[np.ndarray] = []
    for point in ordered:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) <= 0.0:
            hull.pop()
        hull.append(point)
    return np.array(hull)


def _cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """Z-component of the cross product of ``first - origin`` and ``second - origin``."""
    return float(
        (first[0] - origin[0]) * (second[1] - origin[1])
        - (first[1] - origin[1]) * (second[0] - origin[0])
    )


def rocch_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> float:
    """Equal error rate on the ROC convex hull.

    Args:
        target_scores: Scores of same-speaker (mated) trials.
        nontarget_scores: Scores of different-speaker (nonmated) trials.

    Returns:
        The ROCCH-EER, in ``[0, 0.5]``.

    Raises:
        ValueError: If either score set is empty or contains non-finite values.
    """
    pfa, pmiss = roc_convex_hull(target_scores, nontarget_scores)

    # Walk the hull and find where pmiss - pfa changes sign. The hull starts at
    # (0, 1) where the difference is positive and ends at (1, 0) where it is
    # negative, so a crossing always exists.
    difference = pmiss - pfa
    for i in range(len(difference) - 1):
        left, right = difference[i], difference[i + 1]
        if left == 0.0:
            return float(pfa[i])
        if left > 0.0 >= right:
            # Linear interpolation along the segment to the crossing point.
            weight = left / (left - right)
            return float(pfa[i] + weight * (pfa[i + 1] - pfa[i]))
    return float(pfa[-1])


def one_minus_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> float:
    """Return ``1 - ROCCH-EER``, the quantity plotted in the paper.

    Plotting ``1 - EER`` puts the conventional metric on the same axis as the two
    legal metrics, where a higher value means a higher re-identification risk.
    """
    return 1.0 - rocch_eer(target_scores, nontarget_scores)
