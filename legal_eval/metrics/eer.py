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
    hull = _lower_left_hull(_staircase_vertices(is_target[::-1], points))
    return hull[:, 0], hull[:, 1]


def _staircase_vertices(
    is_target_reversed: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Drop the ROC points that lie inside a straight segment.

    Stepping the threshold past a nontarget moves ``pfa`` and leaves ``pmiss``
    alone; stepping past a target does the reverse. So a run of equal labels
    traces a straight line, and every point strictly inside such a run is
    collinear with its neighbours. Collinear points are never hull vertices,
    because the monotone chain pops on a cross product of zero, so removing
    them first cannot change the hull.

    It changes the cost enormously. The full point set has
    ``n_target + n_nontarget + 1`` rows, which for a population of 9,779
    speakers scored against itself is about 95.6 million. The number of label
    runs is at most ``2 * n_target + 1``, about 19,559 for the same case, so
    this is close to a five thousandfold reduction in what the Python hull loop
    has to walk.

    Without it, one EER on that matrix takes minutes and the twenty replicate
    bootstrap behind its standard error takes hours. A run at 9,779 speakers
    was abandoned after four hours inside this loop, which is how the problem
    was found: the reference implementation was written against the paper's
    29-speaker evaluation sets, where the point set is 43,500 rows and the
    difference does not show.

    Args:
        is_target_reversed: Sorted target mask, reversed to match ``points``.
        points: ``(n, 2)`` array of ``(pfa, pmiss)``, pfa increasing.

    Returns:
        The subset of ``points`` at run boundaries, always including the first
        and last row.
    """
    n = points.shape[0]
    if n <= 3:
        return points

    # points has one more row than the mask: row i is the operating point
    # reached after stepping past i sorted scores. A run boundary in the mask
    # therefore bounds a straight segment in points.
    labels = np.asarray(is_target_reversed, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    if labels.size:
        changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        # Keep the point on each side of every transition.
        keep[changes] = True
        keep[np.minimum(changes + 1, n - 1)] = True
    return points[keep]


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


def standard_error_of_mean(values) -> float | None:
    """Standard error of the mean over replicates, or None below two.

    Two metrics whose means differ by less than their combined standard errors
    have not been shown to differ, and at a metric's ceiling that is the usual
    case. Reporting an ordering without this invites a conclusion the data does
    not support.

    A result of exactly ``0.0`` means the replicates were identical, which is
    almost always a fault in how they were drawn rather than a precise
    estimate. :func:`legal_eval.preflight.run_preflight` checks for it.

    Args:
        values: Replicate values, or None.

    Returns:
        The standard error, or None when there are fewer than two values.
    """
    if values is None:
        return None
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        return None
    return float(array.std(ddof=1) / np.sqrt(array.size))


def rocch_eer_standard_error(
    scores: np.ndarray,
    true_rows: np.ndarray,
    n_reps: int = 20,
    seed: int = 0,
) -> float | None:
    """Standard error of the ROCCH-EER, by resampling trials.

    The EER is almost always quoted as a single number, which makes it the one
    metric in this package that cannot be compared against its own noise. Two
    systems whose EERs differ by less than this have not been shown to differ.

    **Trials are the unit of resampling.** Each replicate draws trial columns
    with replacement, keeping a trial's mated score together with its nonmated
    ones rather than breaking them apart. Resampling the candidate draw alone
    is what produced a standard error of exactly zero and a false finding
    earlier in this project.

    Cost scales as the square of the population, because every replicate
    re-sorts the whole score matrix. See `docs/protocol.md`.

    Args:
        scores: ``(n_enroll, n_trials)`` score matrix.
        true_rows: ``(n_trials,)`` enrollment row holding each trial's true
            speaker.
        n_reps: Replicates. Below two the standard error is undefined.
        seed: RNG seed, so a reported figure is reproducible.

    Returns:
        The standard error, or None when ``n_reps`` is below 2.

    Raises:
        ValueError: If the shapes disagree, or a row index is out of range.
    """
    scores = np.asarray(scores, dtype=np.float64)
    true_rows = np.asarray(true_rows)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")
    n_enroll, n_trials = scores.shape
    if true_rows.shape != (n_trials,):
        raise ValueError(
            f"true_rows must have one entry per trial: expected {(n_trials,)}, "
            f"got {true_rows.shape}"
        )
    if n_trials and (true_rows.min() < 0 or true_rows.max() >= n_enroll):
        raise ValueError(
            f"true_rows holds an index outside the {n_enroll} enrollment rows"
        )
    if n_reps < 2:
        return None

    rng = np.random.default_rng(seed)
    columns = np.arange(n_trials)
    values = []
    for _ in range(n_reps):
        pick = rng.integers(0, n_trials, size=n_trials)
        sub = scores[:, pick]
        sub_rows = true_rows[pick]
        mated = sub[sub_rows, columns]
        mask = np.ones_like(sub, dtype=bool)
        mask[sub_rows, columns] = False
        values.append(rocch_eer(mated, sub[mask]))
    return standard_error_of_mean(values)
