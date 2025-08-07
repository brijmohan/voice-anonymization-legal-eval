"""Utility helpers for metric computation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Return the cosine similarity matrix between ``a`` and ``b``.

    Parameters
    ----------
    a, b:
        Arrays of shape ``(n_samples, n_features)`` or ``(n_features,)``.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n_a, n_b)`` where ``n_a`` and ``n_b`` are the
        number of rows in ``a`` and ``b`` respectively.
    """

    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T
