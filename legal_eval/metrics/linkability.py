r"""The Linkability metric :math:`\pi^\text{link}`.

Linkability is the probability that an attacker matches a test embedding of
speaker :math:`i` to that speaker's own enrollment embedding, rather than to any
of the other :math:`N'-1` enrollment speakers:

.. math::

    \pi^\text{link} = \Pr_{x_i^\text{test}}\Bigl\{
        s(x_i^\text{test}, x_i^\text{enroll}) >
        \max_{j \neq i} s(x_i^\text{test}, x_j^\text{enroll})
    \Bigr\}

The attacker's candidate set is a random draw: for each test speaker, the true
enrollment speaker plus :math:`N'-1` others sampled without replacement from the
enrollment population. Sweeping :math:`N'` shows how the risk falls as the
attacker has to search a larger population.

Implementation note
-------------------
The obvious implementation draws an explicit candidate set per test speaker, per
value of :math:`N'`, per run. That is what the original experiment code did, and
it dominates the runtime: the paper's grid is 220 values of :math:`N'` x 5 runs x
4,949 test speakers, each drawing up to 22,023 indices.

It is unnecessary. Linkage succeeds exactly when none of the sampled candidates
beats the true speaker, so the only thing that matters about test speaker
:math:`i` is

.. math:: m_i = \#\{j \neq i : s(x_i^\text{test}, x_j^\text{enroll}) >
                              s(x_i^\text{test}, x_i^\text{enroll})\},

the number of enrollment speakers that outscore the true one. Drawing
:math:`N'-1` of the :math:`n-1` other speakers without replacement and asking
whether any of the :math:`m_i` beaters came along is, by definition, a
hypergeometric experiment:

.. math:: \Pr\{\text{success}\} = \frac{\binom{n-1-m_i}{N'-1}}{\binom{n-1}{N'-1}}

So one ``Hypergeometric(m_i, n-1-m_i, N'-1)`` draw per test speaker replaces the
subset construction, with *identical* sampling distribution. ``m_i`` is computed
once from the score matrix and reused for the whole sweep. The equivalence is
verified against a literal subset-sampling implementation in the test suite.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from legal_eval.io import ScoreMatrix


def count_beaters(
    score_matrix: ScoreMatrix,
    test_speakers: Sequence[str] | None = None,
) -> np.ndarray:
    """Count, per test column, the enrollment speakers that outscore the true one.

    Ties count as a success for the attacker's target, matching the ``argmax``
    convention of the original code: with the true speaker placed first, a tie
    resolves in its favour. Only *strictly* higher scores count as beaters.

    Args:
        score_matrix: Enrollment-by-test cosine scores.
        test_speakers: Column labels, overriding
            ``score_matrix.test_speakers``. Must label every column, in column
            order; a speaker may label several columns. Every label must also be
            an enrollment speaker, since the true speaker has to be in the
            population the attacker searches.

    Returns:
        Integer array ``m``, one entry per column.

    Raises:
        KeyError: If a test speaker is missing from the enrollment set.
        ValueError: If the labels do not cover every column.
    """
    labels = list(score_matrix.test_speakers if test_speakers is None else test_speakers)
    n_columns = score_matrix.scores.shape[1]
    if len(labels) != n_columns:
        raise ValueError(
            f"got {len(labels)} test speaker labels for {n_columns} columns; "
            "labels must cover every column, in column order"
        )
    enroll_index = score_matrix.enroll_index

    missing = sorted({s for s in labels if s not in enroll_index})
    if missing:
        raise KeyError(
            f"{len(missing)} test speaker(s) are absent from the enrollment set, "
            f"so linkability is undefined for them, e.g. {missing[:5]}"
        )

    scores = score_matrix.scores
    true_rows = np.fromiter((enroll_index[s] for s in labels), dtype=np.intp, count=len(labels))
    true_scores = scores[true_rows, np.arange(n_columns, dtype=np.intp)]
    # The true speaker's own entry is never strictly greater than itself, so it
    # drops out of the count without needing to be masked.
    return np.count_nonzero(scores > true_scores[np.newaxis, :], axis=0)


def linkability(
    score_matrix: ScoreMatrix,
    n_enroll_speakers: int,
    n_runs: int = 5,
    seed: int = 0,
    test_speakers: Sequence[str] | None = None,
    estimator: str = "sampling",
    beaters: np.ndarray | None = None,
) -> list[float]:
    r"""Compute :math:`\pi^\text{link}` for a given candidate-set size.

    Args:
        score_matrix: Enrollment-by-test cosine scores.
        n_enroll_speakers: Candidate-set size :math:`N'`, counting the true
            speaker. Must be between 1 and the number of enrollment speakers.
        n_runs: Number of independent random draws of the candidate sets.
        seed: Base seed. Run ``r`` uses the stream ``(seed, n_enroll_speakers, r)``,
            so results do not depend on evaluation order or parallelism.
        test_speakers: Column labels to evaluate. Defaults to all columns.
        estimator: ``"sampling"`` draws candidate sets, reproducing the original
            experiment's sampling distribution. ``"exact"`` instead returns the
            closed-form probability per test speaker, which is the same quantity
            in expectation with no sampling noise; it yields one value, as
            repeated runs would be identical.
        beaters: Precomputed output of :func:`count_beaters`, to avoid
            recomputing it for every :math:`N'` in a sweep.

    Returns:
        One linkability probability per run. Length ``n_runs`` for
        ``"sampling"``, length 1 for ``"exact"``.

    Raises:
        ValueError: If ``n_enroll_speakers`` or ``estimator`` is out of range.
    """
    n_enroll_total = score_matrix.scores.shape[0]
    if not 1 <= n_enroll_speakers <= n_enroll_total:
        raise ValueError(
            f"n_enroll_speakers must be in [1, {n_enroll_total}], got {n_enroll_speakers}"
        )
    if estimator not in ("sampling", "exact"):
        raise ValueError(f"estimator must be 'sampling' or 'exact', got {estimator!r}")

    if beaters is None:
        beaters = count_beaters(score_matrix, test_speakers)
    beaters = np.asarray(beaters, dtype=np.int64)

    n_others = n_enroll_total - 1
    n_sampled = n_enroll_speakers - 1

    if estimator == "exact":
        return [float(np.mean(_success_probability(beaters, n_others, n_sampled)))]

    results: list[float] = []
    for run in range(n_runs):
        rng = np.random.default_rng((seed, n_enroll_speakers, run))
        # Hypergeometric(ngood=m, nbad=n_others-m, nsample=n_sampled) counts how
        # many beaters are drawn; zero beaters means the attacker links correctly.
        drawn_beaters = rng.hypergeometric(
            ngood=beaters,
            nbad=n_others - beaters,
            nsample=np.minimum(n_sampled, n_others),
        )
        results.append(float(np.mean(drawn_beaters == 0)))
    return results


def _success_probability(
    beaters: np.ndarray, n_others: int, n_sampled: int
) -> np.ndarray:
    """Probability that none of ``beaters`` lands in a draw of ``n_sampled``.

    Computed as ``prod_{t=0}^{n_sampled-1} (n_others - m - t) / (n_others - t)``
    in log space, which is stable for the large populations involved.
    """
    if n_sampled == 0:
        return np.ones_like(beaters, dtype=np.float64)
    from scipy.special import gammaln

    good = n_others - beaters.astype(np.float64)
    # C(good, k) / C(n_others, k) with k = n_sampled.
    log_p = (
        gammaln(good + 1.0)
        - gammaln(good - n_sampled + 1.0)
        - gammaln(n_others + 1.0)
        + gammaln(n_others - n_sampled + 1.0)
    )
    # A speaker with more beaters than the population leaves room for is
    # impossible to link, giving probability zero rather than NaN.
    return np.where(good >= n_sampled, np.exp(log_p), 0.0)


def linkability_naive(
    score_matrix: ScoreMatrix,
    n_enroll_speakers: int,
    n_runs: int = 5,
    seed: int = 0,
    test_speakers: Sequence[str] | None = None,
) -> list[float]:
    """Literal implementation of :func:`linkability`, for testing.

    Draws the candidate set explicitly, exactly as the original experiment code
    did. Correct but slow; :func:`linkability` is the one to use. Kept so the
    fast path can be checked against it.
    """
    labels = list(score_matrix.test_speakers if test_speakers is None else test_speakers)
    enroll_index = score_matrix.enroll_index
    scores = score_matrix.scores
    n_enroll_total = scores.shape[0]
    n_sampled = n_enroll_speakers - 1

    results: list[float] = []
    for run in range(n_runs):
        rng = np.random.default_rng((seed, n_enroll_speakers, run))
        successes = 0
        for column, speaker in enumerate(labels):
            true_row = enroll_index[speaker]
            others = rng.choice(n_enroll_total - 1, size=n_sampled, replace=False)
            others = np.where(others >= true_row, others + 1, others)
            true_score = scores[true_row, column]
            if n_sampled == 0 or true_score >= scores[others, column].max():
                successes += 1
        results.append(successes / len(labels))
    return results
