"""Sweeps over population size and conversation length.

A single metric value says little. What the paper reports, and what reveals the
gap between the legal metrics and the EER, is how each metric moves as

* ``N`` / ``N'``: the number of speakers the attacker must search, and
* ``L``: the conversation length, i.e. how many utterances per speaker the
  attacker gets to average.

These functions produce those curves, with one value per run so that the spread
across runs can be shown as an error bar.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from legal_eval.embeddings import SpeakerConversations, build_conversations
from legal_eval.io import ScoreMatrix
from legal_eval.metrics.eer import rocch_eer
from legal_eval.metrics.linkability import count_beaters, linkability
from legal_eval.metrics.singling_out import score_conversations, singling_out


@dataclass
class SweepResult:
    """A metric's value as a function of the number of speakers.

    Attributes:
        metric: ``"linkability"``, ``"singling_out"`` or ``"eer"``.
        conversation_length: The ``L`` this curve was computed at.
        values: Mapping from speaker count to the per-run values.
        metadata: Provenance, e.g. attacker model, seed, estimator.
    """

    metric: str
    conversation_length: int
    values: dict[int, list[float]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def speaker_counts(self) -> list[int]:
        """The sorted speaker counts this curve covers."""
        return sorted(self.values)

    def mean(self) -> dict[int, float]:
        """Mean over runs, per speaker count."""
        return {n: float(np.mean(self.values[n])) for n in self.speaker_counts}

    def std(self) -> dict[int, float]:
        """Standard deviation over runs, per speaker count."""
        return {n: float(np.std(self.values[n])) for n in self.speaker_counts}

    def to_dict(self) -> dict[str, object]:
        """Serialise to plain types, with string keys, for JSON."""
        return {
            "metric": self.metric,
            "conversation_length": self.conversation_length,
            "metadata": self.metadata,
            "values": {str(n): self.values[n] for n in self.speaker_counts},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SweepResult:
        """Inverse of :meth:`to_dict`."""
        raw = payload["values"]
        assert isinstance(raw, Mapping)
        return cls(
            metric=str(payload["metric"]),
            conversation_length=int(payload["conversation_length"]),
            values={int(n): [float(v) for v in vals] for n, vals in raw.items()},
            metadata=dict(payload.get("metadata", {})),  # type: ignore[arg-type]
        )


def default_speaker_counts(
    maximum: int, start: int = 20, step: int = 100
) -> list[int]:
    """The paper's Linkability grid: ``start`` then every ``step`` up to ``maximum``.

    Args:
        maximum: Largest speaker count available.
        start: First point on the grid.
        step: Spacing.

    Returns:
        Increasing speaker counts, all at most ``maximum``.
    """
    counts = list(range(start, maximum + 1, step))
    if counts and counts[-1] != maximum:
        counts.append(maximum)
    return counts


def geometric_speaker_counts(maximum: int, start: int = 30, base: int = 100) -> list[int]:
    """The paper's Singling Out grid: ``start``, then ``base`` doubling to ``maximum``.

    Singling Out costs far more per point than Linkability, so it is evaluated on
    a geometric rather than a linear grid.
    """
    counts = [start]
    value = base
    while value < maximum:
        counts.append(value)
        value *= 2
    counts.append(maximum)
    return [n for n in dict.fromkeys(counts) if n <= maximum]


def linkability_sweep(
    score_matrix: ScoreMatrix,
    speaker_counts: Sequence[int] | None = None,
    n_runs: int = 5,
    seed: int = 0,
    conversation_length: int = 1,
    estimator: str = "sampling",
    test_speakers: Sequence[str] | None = None,
    metadata: dict[str, object] | None = None,
    progress: bool = False,
) -> SweepResult:
    r"""Compute :math:`\pi^\text{link}` across candidate-set sizes.

    Args:
        score_matrix: Enrollment-by-test cosine scores for one conversation
            length and one attacker.
        speaker_counts: Values of :math:`N'`. Defaults to the paper's grid.
        n_runs: Random draws of the candidate sets per point.
        seed: Base seed.
        conversation_length: Recorded on the result; the matrix already encodes it.
        estimator: See :func:`~legal_eval.metrics.linkability.linkability`.
        test_speakers: Column labels, if overriding the matrix's own.
        metadata: Extra provenance to record.
        progress: Show a progress bar if ``tqdm`` is installed.

    Returns:
        The :class:`SweepResult`.
    """
    n_enroll = score_matrix.scores.shape[0]
    counts = list(speaker_counts) if speaker_counts is not None else default_speaker_counts(n_enroll)
    # Computed once and reused: this is what makes the sweep cheap.
    beaters = count_beaters(score_matrix, test_speakers)

    result = SweepResult(
        metric="linkability",
        conversation_length=conversation_length,
        metadata={
            "n_enroll_total": n_enroll,
            "n_test": int(beaters.size),
            "n_runs": n_runs,
            "seed": seed,
            "estimator": estimator,
            **dict(metadata or {}),
        },
    )
    for count in _maybe_progress(counts, progress, "linkability"):
        if count > n_enroll:
            continue
        result.values[int(count)] = linkability(
            score_matrix,
            n_enroll_speakers=int(count),
            n_runs=n_runs,
            seed=seed,
            test_speakers=test_speakers,
            estimator=estimator,
            beaters=beaters,
        )
    return result


def singling_out_sweep(
    enroll_embeddings: Mapping[str, np.ndarray],
    dataset_spk2utt: Mapping[str, Sequence[str]],
    utt2embedding: Mapping[str, np.ndarray],
    conversation_length: int,
    speaker_counts: Sequence[int] | None = None,
    n_runs: int = 5,
    n_folds: int = 10,
    max_calibration: int = 9,
    seed: int = 0,
    vary_folds: bool = True,
    metadata: dict[str, object] | None = None,
    progress: bool = False,
) -> SweepResult:
    r"""Compute :math:`\pi^\text{sing}` across test-set sizes.

    Each fold rebuilds the test and calibration conversations, rescoring them
    against the enrollment speakers; within a fold, every ``N`` and every
    enrollment speaker is evaluated over ``n_runs`` draws of the test-speaker
    set. A point's value is the mean outcome per ``(enrollment speaker, fold)``
    pair, so the reported spread is the spread across attackers and splits.

    Args:
        enroll_embeddings: Averaged embedding per enrollment speaker, each one
            defining a predicate.
        dataset_spk2utt: Test speakers and their utterance ids. Must be disjoint
            from the utterances used to build ``enroll_embeddings``.
        utt2embedding: Embedding per utterance id for the test speakers.
        conversation_length: Conversation length ``L``.
        speaker_counts: Values of ``N``. Defaults to the paper's geometric grid.
        n_runs: Draws of the test-speaker set per enrollment speaker and fold.
        n_folds: Cross-validation folds over the conversation split.
        max_calibration: Calibration conversations per speaker, ``M`` in the paper.
        seed: Base seed.
        vary_folds: Whether folds get different conversation splits. ``False``
            reproduces the original code's behaviour; see ``docs/differences.md``.
        metadata: Extra provenance to record.
        progress: Show a progress bar if ``tqdm`` is installed.

    Returns:
        The :class:`SweepResult`. Each point holds one value per
        ``(enrollment speaker, fold)`` pair.
    """
    per_point: dict[int, list[float]] = {}
    n_eligible = 0

    for fold in _maybe_progress(range(n_folds), progress, f"singling out L={conversation_length}"):
        conversations: dict[str, SpeakerConversations] = build_conversations(
            dataset_spk2utt,
            utt2embedding,
            conversation_length=conversation_length,
            max_calibration=max_calibration,
            seed=seed,
            fold=fold,
            vary_folds=vary_folds,
        )
        if not conversations:
            raise ValueError(
                f"no test speaker has the {2 * conversation_length} utterances "
                f"needed for a conversation length of {conversation_length}"
            )
        scores = score_conversations(enroll_embeddings, conversations)
        n_eligible = len(conversations)

        counts = (
            list(speaker_counts)
            if speaker_counts is not None
            else geometric_speaker_counts(n_eligible)
        )
        for count in counts:
            if not 2 <= count <= n_eligible:
                continue
            outcomes = singling_out(
                scores,
                n_test_speakers=int(count),
                n_runs=n_runs,
                seed=seed,
                fold=fold,
            )
            per_point.setdefault(int(count), []).extend(
                float(np.mean(values)) for values in outcomes.values()
            )

    return SweepResult(
        metric="singling_out",
        conversation_length=conversation_length,
        values=per_point,
        metadata={
            "n_enroll": len(enroll_embeddings),
            "n_dataset_eligible": n_eligible,
            "n_runs": n_runs,
            "n_folds": n_folds,
            "max_calibration": max_calibration,
            "seed": seed,
            "vary_folds": vary_folds,
            **dict(metadata or {}),
        },
    )


def eer_sweep(
    score_matrix: ScoreMatrix,
    speaker_counts: Sequence[int] | None = None,
    n_runs: int = 5,
    seed: int = 0,
    conversation_length: int = 1,
    max_nontarget_per_speaker: int = 500,
    test_speakers: Sequence[str] | None = None,
    metadata: dict[str, object] | None = None,
    progress: bool = False,
) -> SweepResult:
    """Compute the ROCCH-EER across candidate-set sizes.

    Target trials are each test item against its own enrollment speaker;
    nontarget trials are against speakers drawn from the rest of the candidate
    set. The number of nontarget trials per test item is capped, because the EER
    depends on the *distribution* of nontarget scores rather than on how many
    were drawn, and that distribution does not change with the candidate-set
    size. The cap keeps the sweep affordable at large ``N'`` without biasing the
    result; the paper's finding that the EER is nearly flat in ``N'`` is exactly
    this observation.

    Args:
        score_matrix: Enrollment-by-test cosine scores.
        speaker_counts: Values of ``N'``. Defaults to the geometric grid, since a
            flat curve does not need a fine one.
        n_runs: Random draws per point.
        seed: Base seed.
        conversation_length: Recorded on the result.
        max_nontarget_per_speaker: Cap on nontarget trials per test item.
        test_speakers: Column labels, if overriding the matrix's own.
        metadata: Extra provenance to record.
        progress: Show a progress bar if ``tqdm`` is installed.

    Returns:
        The :class:`SweepResult`, holding ROCCH-EER values. Plot ``1 - value``.
    """
    scores = score_matrix.scores
    n_enroll, n_columns = scores.shape
    labels = list(score_matrix.test_speakers if test_speakers is None else test_speakers)
    if len(labels) != n_columns:
        raise ValueError(
            f"got {len(labels)} test speaker labels for {n_columns} columns"
        )
    enroll_index = score_matrix.enroll_index
    true_rows = np.fromiter(
        (enroll_index[s] for s in labels), dtype=np.intp, count=n_columns
    )
    target_scores = scores[true_rows, np.arange(n_columns, dtype=np.intp)]

    counts = (
        list(speaker_counts)
        if speaker_counts is not None
        else geometric_speaker_counts(n_enroll)
    )

    result = SweepResult(
        metric="eer",
        conversation_length=conversation_length,
        metadata={
            "n_enroll_total": n_enroll,
            "n_test": n_columns,
            "n_runs": n_runs,
            "seed": seed,
            "max_nontarget_per_speaker": max_nontarget_per_speaker,
            **dict(metadata or {}),
        },
    )

    for count in _maybe_progress(counts, progress, "eer"):
        if not 2 <= count <= n_enroll:
            continue
        n_sampled = min(count - 1, max_nontarget_per_speaker)
        per_run: list[float] = []
        for run in range(n_runs):
            rng = np.random.default_rng((seed, int(count), run))
            nontarget = np.empty(n_columns * n_sampled, dtype=np.float64)
            for column in range(n_columns):
                others = _sample_excluding(rng, n_enroll, int(true_rows[column]), n_sampled)
                nontarget[column * n_sampled : (column + 1) * n_sampled] = scores[
                    others, column
                ]
            per_run.append(rocch_eer(target_scores, nontarget))
        result.values[int(count)] = per_run
    return result


def _sample_excluding(
    rng: np.random.Generator, population: int, excluded: int, size: int
) -> np.ndarray:
    """Draw ``size`` distinct indices from ``range(population)`` minus ``excluded``."""
    if size > population - 1:
        raise ValueError(
            f"cannot draw {size} distinct indices from a population of {population - 1}"
        )
    drawn = rng.choice(population - 1, size=size, replace=False)
    return np.where(drawn >= excluded, drawn + 1, drawn)


def _maybe_progress(iterable: Iterable, enabled: bool, description: str) -> Iterable:
    """Wrap ``iterable`` in a tqdm bar when asked for and available."""
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description)
