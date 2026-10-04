"""Turning utterance-level embeddings into the speaker and conversation
embeddings the metrics consume.

Two conventions from the original experiments are preserved exactly, because
both affect the reported numbers:

1.  Averaging re-normalises. The x-vector extractor already L2-normalises each
    utterance embedding, so a single-utterance "average" is returned untouched;
    an average over several utterances is L2-normalised once, after averaging.
2.  Which utterances form a conversation depends on the conversation length.
    See :func:`build_test_embeddings`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np

#: How to pick the ``L`` utterances that make up a test conversation.
SELECTION_MODES = ("reference", "first", "random")


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` scaled to unit L2 norm.

    A zero vector is returned unchanged rather than producing NaNs.
    """
    norm = np.linalg.norm(vector, ord=2)
    if norm == 0:
        return vector
    return vector / norm


def average_embeddings(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Average embeddings and L2-normalise the result.

    A single embedding is returned as-is: the extractor already normalised it,
    and normalising twice would be a no-op at best and a rounding difference at
    worst. This mirrors ``average_x_vectors`` in the original experiment code.

    Args:
        embeddings: One or more 1-D embeddings of equal dimension.

    Returns:
        The L2-normalised mean embedding.

    Raises:
        ValueError: If ``embeddings`` is empty.
    """
    if len(embeddings) == 0:
        raise ValueError("cannot average an empty sequence of embeddings")
    if len(embeddings) == 1:
        return np.asarray(embeddings[0], dtype=np.float64)
    mean = np.mean(np.asarray(embeddings, dtype=np.float64), axis=0)
    return l2_normalize(mean)


def build_speaker_embeddings(
    spk2utt: Mapping[str, Sequence[str]],
    utt2embedding: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Average *all* of each speaker's utterances into one embedding.

    This is how the enrollment side of the Linkability experiment is built: the
    attacker is assumed to hold every enrollment utterance of a speaker.

    Args:
        spk2utt: Mapping from speaker id to utterance ids.
        utt2embedding: Mapping from utterance id to embedding.

    Returns:
        Mapping from speaker id to that speaker's averaged embedding.
    """
    return {
        speaker: average_embeddings([utt2embedding[utt] for utt in utterances])
        for speaker, utterances in spk2utt.items()
    }


def build_test_embeddings(
    spk2utt: Mapping[str, Sequence[str]],
    utt2embedding: Mapping[str, np.ndarray],
    conversation_length: int,
    selection: str = "reference",
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Build one embedding per test speaker from ``L`` of its utterances.

    ``selection`` controls which utterances are used:

    ``"reference"``
        Reproduces the original experiments exactly: for ``L == 1`` a single
        utterance is drawn at random, and for ``L > 1`` the *first* ``L``
        utterances in ``spk2utt`` order are used. The asymmetry is an artefact
        of the two scripts the paper's rows came from; it is kept so that the
        published curves can be reproduced.
    ``"first"``
        Always the first ``L`` utterances. Deterministic, no RNG needed.
    ``"random"``
        A random sample of ``L`` utterances, for every ``L``. The most
        defensible choice for new experiments.

    Args:
        spk2utt: Mapping from speaker id to utterance ids.
        utt2embedding: Mapping from utterance id to embedding.
        conversation_length: Number of utterances ``L`` per speaker.
        selection: One of :data:`SELECTION_MODES`.
        rng: Generator used by the random modes. Required for ``"random"``, and
            for ``"reference"`` when ``L == 1``.

    Returns:
        A pair ``(embeddings, chosen)``: the averaged embedding per speaker, and
        the utterance ids that went into it.

    Raises:
        ValueError: If ``selection`` is unknown, if ``conversation_length`` is
            not positive, if a random mode is requested without an ``rng``, or
            if a speaker has fewer than ``L`` utterances.
    """
    if selection not in SELECTION_MODES:
        raise ValueError(f"selection must be one of {SELECTION_MODES}, got {selection!r}")
    if conversation_length < 1:
        raise ValueError("conversation_length must be at least 1")

    needs_rng = selection == "random" or (selection == "reference" and conversation_length == 1)
    if needs_rng and rng is None:
        raise ValueError(f"selection={selection!r} with L={conversation_length} requires an rng")

    embeddings: dict[str, np.ndarray] = {}
    chosen: dict[str, list[str]] = {}
    for speaker, utterances in spk2utt.items():
        if len(utterances) < conversation_length:
            raise ValueError(
                f"speaker {speaker!r} has {len(utterances)} utterances, "
                f"fewer than the conversation length {conversation_length}"
            )
        if selection == "first" or (selection == "reference" and conversation_length > 1):
            picked = list(utterances[:conversation_length])
        elif selection == "reference":  # L == 1, single random utterance
            picked = [utterances[int(rng.integers(len(utterances)))]]
        else:  # "random"
            indices = rng.choice(len(utterances), size=conversation_length, replace=False)
            picked = [utterances[int(i)] for i in indices]
        chosen[speaker] = picked
        embeddings[speaker] = average_embeddings([utt2embedding[utt] for utt in picked])
    return embeddings, chosen


class SpeakerConversations:
    """The test and calibration conversations of one speaker, as embeddings.

    The Singling Out metric needs, per speaker, one *test* conversation and up
    to ``M`` *calibration* conversations, all of length ``L`` and all built from
    disjoint utterances of that speaker.

    Attributes:
        speaker: Speaker id.
        test: Embedding of the test conversation.
        calibration: Embeddings of the calibration conversations.
    """

    __slots__ = ("speaker", "test", "calibration")

    def __init__(self, speaker: str, test: np.ndarray, calibration: list[np.ndarray]) -> None:
        self.speaker = speaker
        self.test = test
        self.calibration = calibration

    @property
    def n_calibration(self) -> int:
        """Number of calibration conversations, written ``C`` in the paper."""
        return len(self.calibration)


def build_conversations(
    spk2utt: Mapping[str, Sequence[str]],
    utt2embedding: Mapping[str, np.ndarray],
    conversation_length: int,
    max_calibration: int = 9,
    seed: int = 0,
    fold: int = 0,
    vary_folds: bool = True,
) -> dict[str, SpeakerConversations]:
    """Split each speaker's utterances into one test and ``C`` calibration
    conversations of length ``L``.

    For a speaker with ``K`` utterances the utterances are shuffled, the first
    ``L`` become the test conversation, and
    ``C = min(max_calibration, comb(K - L, L))`` calibration conversations are
    taken as the first ``C`` combinations of the remaining ``K - L``
    utterances. Speakers with ``K < 2L`` cannot supply both and are skipped.

    Args:
        spk2utt: Mapping from speaker id to utterance ids.
        utt2embedding: Mapping from utterance id to embedding.
        conversation_length: Conversation length ``L``.
        max_calibration: Upper bound ``M`` on the number of calibration
            conversations per speaker. The paper uses 9, which together with the
            single test conversation gives the 10 utterances-per-speaker budget.
        seed: Base seed for the per-speaker shuffles.
        fold: Cross-validation fold index.
        vary_folds: If ``True`` the shuffle depends on ``fold``, so that each
            fold is a genuinely different split. The original code seeded only
            on the speaker's position, which froze the split across folds; pass
            ``False`` to reproduce that. See ``docs/differences.md``.

    Returns:
        Mapping from speaker id to its :class:`SpeakerConversations`. Speakers
        with too few utterances are absent.
    """
    if conversation_length < 1:
        raise ValueError("conversation_length must be at least 1")

    conversations: dict[str, SpeakerConversations] = {}
    for position, (speaker, utterances) in enumerate(spk2utt.items()):
        if len(utterances) < 2 * conversation_length:
            continue
        # One independent stream per (speaker, fold) so that results do not
        # depend on how the work is distributed across processes.
        stream = np.random.default_rng(
            (seed, position, fold) if vary_folds else (seed, position)
        )
        shuffled = list(utterances)
        stream.shuffle(shuffled)

        test_utterances = shuffled[:conversation_length]
        test = average_embeddings([utt2embedding[utt] for utt in test_utterances])

        remaining = shuffled[conversation_length:]
        n_calibration = min(
            max_calibration,
            math.comb(len(remaining), conversation_length),
        )
        calibration = [
            average_embeddings([utt2embedding[utt] for utt in combination])
            for combination in itertools.islice(
                itertools.combinations(remaining, conversation_length), n_calibration
            )
        ]
        if not calibration:
            continue
        conversations[speaker] = SpeakerConversations(speaker, test, calibration)
    return conversations
