r"""The Singling Out metric :math:`\pi^\text{sing}`.

Singling Out follows the predicate singling out (PSO) framework of Cohen and
Nissim. The attacker holds one enrollment speaker embedding
:math:`x^\text{enroll}` and a threshold, which together define the predicate

.. math:: p(x^\text{test}) = \mathds{1}\{s(x^\text{test}, x^\text{enroll}) > s^\text{thresh}\}

Given a set :math:`X` of :math:`N` test embeddings, one per speaker, isolation
succeeds when the predicate fires on exactly one of them, whether or not it is
the attacker's own speaker. The metric is the probability of isolation over sets
and enrollment speakers:

.. math:: \pi^\text{sing} = \Pr_{X, x^\text{enroll}}\{\exists! i :\ p(x^\text{test}_i) = 1\}

The threshold is calibrated, not chosen: it is set so that the predicate fires on
:math:`1/N` of a held-out calibration set, which is what makes the measured
isolation rate comparable to the :math:`\exp(-1) \approx 37\%` a random
predicate of the same expectation achieves. With :math:`M` calibration
conversations per speaker and :math:`N` speakers there are :math:`MN`
calibration scores, of which :math:`M` must pass; the threshold is therefore
placed midway between the :math:`M`-th and :math:`(M+1)`-th highest scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from legal_eval.embeddings import SpeakerConversations
from legal_eval.scoring import cosine_similarity_matrix

#: Isolation rate achieved by a random predicate of expectation 1/N.
TRIVIAL_SINGLING_OUT = float(np.exp(-1))


class ConversationScores:
    """Cosine scores from each enrollment speaker to every conversation.

    Holds, for each enrollment ("predicate") speaker, the score against every
    test speaker's test conversation and against every calibration conversation.

    Attributes:
        enroll_speakers: Enrollment speaker ids, in row order.
        dataset_speakers: Test speaker ids, in column order.
        test_scores: Array ``(n_enroll, n_dataset)`` of test-conversation scores.
        calibration_scores: Array ``(n_enroll, total_calibration)`` of
            calibration-conversation scores.
        calibration_slices: For each test speaker, the half-open column range of
            ``calibration_scores`` belonging to it.
    """

    __slots__ = (
        "enroll_speakers",
        "dataset_speakers",
        "test_scores",
        "calibration_scores",
        "calibration_slices",
    )

    def __init__(
        self,
        enroll_speakers: list[str],
        dataset_speakers: list[str],
        test_scores: np.ndarray,
        calibration_scores: np.ndarray,
        calibration_slices: list[tuple[int, int]],
    ) -> None:
        self.enroll_speakers = enroll_speakers
        self.dataset_speakers = dataset_speakers
        self.test_scores = test_scores
        self.calibration_scores = calibration_scores
        self.calibration_slices = calibration_slices

    @property
    def dataset_index(self) -> dict[str, int]:
        """Mapping from test speaker id to column index."""
        return {spk: i for i, spk in enumerate(self.dataset_speakers)}


def score_conversations(
    enroll_embeddings: Mapping[str, np.ndarray],
    conversations: Mapping[str, SpeakerConversations],
) -> ConversationScores:
    """Score every enrollment speaker against every conversation.

    Args:
        enroll_embeddings: Mapping from enrollment speaker id to the averaged
            embedding that defines its predicate.
        conversations: Output of
            :func:`~legal_eval.embeddings.build_conversations`.

    Returns:
        The :class:`ConversationScores`.
    """
    enroll_speakers = list(enroll_embeddings.keys())
    dataset_speakers = list(conversations.keys())
    if not enroll_speakers:
        raise ValueError("no enrollment embeddings given")
    if not dataset_speakers:
        raise ValueError("no conversations given")

    enroll = np.vstack(
        [np.asarray(enroll_embeddings[s], dtype=np.float64) for s in enroll_speakers]
    )

    test = np.vstack([conversations[s].test for s in dataset_speakers])
    test_scores = cosine_similarity_matrix(enroll, test)

    calibration_rows: list[np.ndarray] = []
    calibration_slices: list[tuple[int, int]] = []
    offset = 0
    for speaker in dataset_speakers:
        block = conversations[speaker].calibration
        calibration_rows.extend(block)
        calibration_slices.append((offset, offset + len(block)))
        offset += len(block)
    calibration_scores = cosine_similarity_matrix(enroll, np.vstack(calibration_rows))

    return ConversationScores(
        enroll_speakers=enroll_speakers,
        dataset_speakers=dataset_speakers,
        test_scores=test_scores,
        calibration_scores=calibration_scores,
        calibration_slices=calibration_slices,
    )


def isolation_threshold(
    calibration_scores: np.ndarray, n_expected_above: int
) -> float:
    """Place the threshold so that exactly ``n_expected_above`` scores pass.

    Args:
        calibration_scores: Calibration scores of one enrollment speaker against
            the calibration conversations of the ``N`` selected test speakers.
        n_expected_above: How many scores should fall above the threshold, i.e.
            the number of calibration conversations the enrollment speaker's own
            test speaker contributes. This is what enforces
            ``E{p(x_calib)} = 1/N``.

    Returns:
        The midpoint between the ``n``-th and ``(n+1)``-th highest scores.

    Raises:
        ValueError: If there are not enough calibration scores to straddle.
    """
    n = int(n_expected_above)
    if n < 1:
        raise ValueError("n_expected_above must be at least 1")
    if calibration_scores.size < n + 1:
        raise ValueError(
            f"need at least {n + 1} calibration scores to place the threshold, "
            f"got {calibration_scores.size}"
        )
    # Only the top n+1 order statistics matter, so partition rather than sort.
    # Sorted ascending, top[0] is the (n+1)-th highest score and top[1] the n-th.
    top = np.partition(calibration_scores, -(n + 1))[-(n + 1) :]
    top.sort()
    return float((top[0] + top[1]) / 2.0)


def singling_out(
    conversation_scores: ConversationScores,
    n_test_speakers: int,
    n_runs: int = 5,
    seed: int = 0,
    fold: int = 0,
    enroll_speakers: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    r"""Compute per-enrollment-speaker isolation outcomes for one ``N`` and fold.

    For each enrollment speaker and each run, a set of ``N`` test speakers is
    drawn: the enrollment speaker's own test speaker plus ``N-1`` others. The
    threshold is calibrated on that set's calibration conversations, and the
    predicate is applied to the ``N`` test conversations. The outcome is 1 when
    exactly one test conversation fires.

    Only enrollment speakers that also appear as test speakers are evaluated,
    because the calibration rule needs the speaker's own calibration
    conversations to be present.

    Args:
        conversation_scores: Output of :func:`score_conversations`.
        n_test_speakers: Set size :math:`N`, counting the enrollment speaker's
            own test speaker.
        n_runs: Number of random draws of the test-speaker set.
        seed: Base seed. Each ``(enrollment speaker, N, fold, run)`` gets its own
            stream, so results are independent of evaluation order.
        fold: Fold index, mixed into the seed.
        enroll_speakers: Restrict to these enrollment speakers.

    Returns:
        Mapping from enrollment speaker id to its ``n_runs`` isolation outcomes,
        each 0 or 1.

    Raises:
        ValueError: If ``n_test_speakers`` exceeds the number of test speakers.
    """
    dataset_speakers = conversation_scores.dataset_speakers
    dataset_index = conversation_scores.dataset_index
    n_dataset = len(dataset_speakers)
    if not 2 <= n_test_speakers <= n_dataset:
        raise ValueError(
            f"n_test_speakers must be in [2, {n_dataset}], got {n_test_speakers}"
        )

    candidates = list(
        conversation_scores.enroll_speakers if enroll_speakers is None else enroll_speakers
    )
    evaluated = [s for s in candidates if s in dataset_index]

    enroll_index = {s: i for i, s in enumerate(conversation_scores.enroll_speakers)}
    slices = conversation_scores.calibration_slices
    test_scores = conversation_scores.test_scores
    calibration_scores = conversation_scores.calibration_scores

    outcomes: dict[str, list[int]] = {}
    for speaker in evaluated:
        row = enroll_index[speaker]
        own_column = dataset_index[speaker]
        own_start, own_stop = slices[own_column]
        n_own_calibration = own_stop - own_start

        row_test = test_scores[row]
        row_calibration = calibration_scores[row]

        speaker_outcomes: list[int] = []
        for run in range(n_runs):
            rng = np.random.default_rng((seed, n_test_speakers, fold, row, run))
            others = rng.choice(n_dataset - 1, size=n_test_speakers - 1, replace=False)
            others = np.where(others >= own_column, others + 1, others)
            selected = np.concatenate(([own_column], others))

            calibration_indices = np.concatenate(
                [np.arange(slices[c][0], slices[c][1]) for c in selected]
            )
            threshold = isolation_threshold(
                row_calibration[calibration_indices], n_own_calibration
            )
            n_firing = int(np.count_nonzero(row_test[selected] > threshold))
            speaker_outcomes.append(1 if n_firing == 1 else 0)
        outcomes[speaker] = speaker_outcomes
    return outcomes
