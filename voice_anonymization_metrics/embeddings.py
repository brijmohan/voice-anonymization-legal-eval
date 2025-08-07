"""Helpers for manipulating speaker embeddings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def average_embeddings(embeddings: ArrayLike) -> np.ndarray:
    """Return the average embedding.

    Parameters
    ----------
    embeddings:
        Sequence or array of shape ``(n_samples, n_features)``.

    Returns
    -------
    numpy.ndarray
        Vector of shape ``(n_features,)`` representing the arithmetic mean of
        the input embeddings.
    """

    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim == 1:
        return emb.copy()
    return emb.mean(axis=0)
