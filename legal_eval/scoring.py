"""Cosine score matrices.

Every metric in this package is a function of a matrix of cosine similarities
between enrollment speaker embeddings and test items. Computing that matrix once
and reusing it across the sweeps over the number of speakers and the attacker
models is what makes the evaluation tractable: the paper's Linkability curves
come from a single 22,024 x 4,949 matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from legal_eval.io import ScoreMatrix


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of ``left`` and every row of ``right``.

    Args:
        left: Array of shape ``(n, d)``.
        right: Array of shape ``(m, d)``.

    Returns:
        Array of shape ``(n, m)``.
    """
    left = np.atleast_2d(np.asarray(left, dtype=np.float64))
    right = np.atleast_2d(np.asarray(right, dtype=np.float64))
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"embedding dimensions differ: {left.shape[1]} and {right.shape[1]}"
        )
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


def cosine_score_matrix(
    enroll_embeddings: Mapping[str, np.ndarray],
    test_embeddings: Mapping[str, np.ndarray],
    test_order: Sequence[str] | None = None,
    metadata: dict[str, object] | None = None,
    block_size: int = 4096,
) -> ScoreMatrix:
    """Score every enrollment speaker against every test item.

    Args:
        enroll_embeddings: Mapping from enrollment speaker id to embedding.
        test_embeddings: Mapping from test item id to embedding.
        test_order: Column order. Defaults to ``test_embeddings`` insertion
            order. Pass an explicit order when a speaker contributes several
            columns.
        metadata: Provenance stored in the sidecar, for example the conversation
            length and the attacker model.
        block_size: Number of columns scored at a time, to bound peak memory.

    Returns:
        The :class:`~legal_eval.io.ScoreMatrix`.
    """
    enroll_speakers: list[str] = list(enroll_embeddings.keys())
    test_speakers: list[str] = (
        list(test_embeddings.keys()) if test_order is None else list(test_order)
    )
    if not enroll_speakers:
        raise ValueError("no enrollment embeddings given")
    if not test_speakers:
        raise ValueError("no test embeddings given")

    enroll = np.vstack([np.asarray(enroll_embeddings[s], dtype=np.float64) for s in enroll_speakers])
    scores = np.empty((len(enroll_speakers), len(test_speakers)), dtype=np.float64)

    for start in range(0, len(test_speakers), block_size):
        block = test_speakers[start : start + block_size]
        test = np.vstack([np.asarray(test_embeddings[s], dtype=np.float64) for s in block])
        scores[:, start : start + len(block)] = cosine_similarity_matrix(enroll, test)

    return ScoreMatrix(
        scores=scores,
        enroll_speakers=enroll_speakers,
        test_speakers=test_speakers,
        metadata=dict(metadata or {}),
    )
