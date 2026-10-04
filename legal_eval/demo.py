"""Synthetic embeddings, so the framework can be run and tested without data.

The real evaluation needs 1,700 hours of anonymized Common Voice and three
trained x-vector extractors. To let anyone exercise the pipeline in seconds,
this module fabricates embeddings with the structure the metrics care about:
speakers sit at random points on the unit sphere, and their utterances scatter
around them.

What is simulated here is *embedding geometry*, not speech and not
anonymization. :func:`simulate_anonymization` degrades speaker separability in a
controlled way so that the metrics visibly move; it is not an anonymization
system and tells you nothing about any real one. To evaluate a real system,
extract embeddings from its output and feed those in instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from legal_eval.embeddings import l2_normalize


def synthetic_corpus(
    n_speakers: int = 60,
    n_utterances: int = 12,
    dim: int = 32,
    within_speaker_std: float = 2.0,
    seed: int = 0,
    prefix: str = "spk",
) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    """Generate a corpus of synthetic speaker embeddings.

    Args:
        n_speakers: Number of speakers.
        n_utterances: Utterances per speaker.
        dim: Embedding dimension.
        within_speaker_std: Scatter of utterances around their speaker's centre,
            as a fraction of the unit radius. It is scaled by ``1/sqrt(dim)``
            internally, so the same value means the same separability at any
            embedding dimension. Larger means less separable speakers and so
            lower measured re-identification risk. The default is chosen so
            that Linkability on the unmodified corpus falls with population size
            at roughly the rate the paper reports for original speech.
        seed: Seed for reproducibility.
        prefix: Speaker id prefix.

    Returns:
        A pair ``(spk2utt, utt2embedding)`` in the same shape the real loaders
        return.
    """
    if n_speakers < 2:
        raise ValueError("need at least 2 speakers")
    if n_utterances < 1:
        raise ValueError("need at least 1 utterance per speaker")

    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_speakers, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    # Scale per-dimension noise so the perturbation's norm, not its per-axis
    # standard deviation, is what within_speaker_std controls.
    scatter = within_speaker_std / np.sqrt(dim)

    spk2utt: dict[str, list[str]] = {}
    utt2embedding: dict[str, np.ndarray] = {}
    for index in range(n_speakers):
        speaker = f"{prefix}{index:04d}"
        utterances = []
        for utt_index in range(n_utterances):
            utt = f"{speaker}-{utt_index:04d}"
            noisy = centres[index] + scatter * rng.normal(size=dim)
            utt2embedding[utt] = l2_normalize(noisy)
            utterances.append(utt)
        spk2utt[speaker] = utterances
    return spk2utt, utt2embedding


def split_corpus(
    spk2utt: Mapping[str, Sequence[str]], n_first: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split each speaker's utterances into two disjoint sets.

    The evaluation needs disjoint utterance sets for enrollment, calibration and
    test, so that a measured similarity never comes from a shared recording.

    Args:
        spk2utt: Mapping from speaker id to utterance ids.
        n_first: How many utterances go to the first part.

    Returns:
        A pair of ``spk2utt`` mappings. Speakers left with nothing in either part
        are dropped from that part.
    """
    first: dict[str, list[str]] = {}
    second: dict[str, list[str]] = {}
    for speaker, utterances in spk2utt.items():
        head, tail = list(utterances[:n_first]), list(utterances[n_first:])
        if head:
            first[speaker] = head
        if tail:
            second[speaker] = tail
    return first, second


def simulate_anonymization(
    utt2embedding: Mapping[str, np.ndarray],
    strength: float = 0.8,
    n_pseudo_speakers: int = 8,
    seed: int = 1,
) -> dict[str, np.ndarray]:
    """Degrade speaker separability, to show the metrics responding.

    Each utterance is pulled toward one of a few randomly placed pseudo-speakers,
    loosely echoing what an x-vector replacement system does to the embedding
    space. This is a demonstration aid, not an anonymization system, and its
    output says nothing about the privacy of any real system.

    Args:
        utt2embedding: Original embeddings.
        strength: Fraction of the pseudo-speaker direction mixed in, in ``[0, 1]``.
            ``0`` leaves embeddings untouched; ``1`` erases the original speaker.
        n_pseudo_speakers: Size of the pseudo-speaker pool.
        seed: Seed for reproducibility.

    Returns:
        Mapping from utterance id to the degraded embedding.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    rng = np.random.default_rng(seed)
    any_embedding = next(iter(utt2embedding.values()))
    pseudo = rng.normal(size=(n_pseudo_speakers, len(any_embedding)))
    pseudo /= np.linalg.norm(pseudo, axis=1, keepdims=True)

    out: dict[str, np.ndarray] = {}
    for utt, embedding in utt2embedding.items():
        target = pseudo[rng.integers(n_pseudo_speakers)]
        out[utt] = l2_normalize((1.0 - strength) * embedding + strength * target)
    return out
