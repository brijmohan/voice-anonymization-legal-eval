"""Calibration utilities for Singling Out metric."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .utils import cosine_similarity


def calibrate_threshold(
    enroll_vec: ArrayLike,
    calib_vecs: ArrayLike,
    target_prob: float,
) -> float:
    """Return similarity threshold yielding the desired predicate expectation.

    The function computes cosine similarities between ``enroll_vec`` and
    ``calib_vecs`` then finds a threshold such that a random calibration score
    exceeds it with probability close to ``target_prob``.
    """

    scores = cosine_similarity(calib_vecs, enroll_vec).ravel()
    scores.sort()
    scores = scores[::-1]
    k = int(np.ceil(len(scores) * target_prob))
    if k <= 0:
        return 1.0
    if k >= len(scores):
        return scores[-1]
    upper = scores[k - 1]
    lower = scores[k]
    return 0.5 * (upper + lower)
