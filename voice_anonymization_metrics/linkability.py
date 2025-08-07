"""Implementation of the Linkability metric."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .utils import cosine_similarity


def linkability_probability(test_vecs: ArrayLike, enroll_vecs: ArrayLike) -> float:
    """Return the probability that test vectors link to the correct speaker.

    Parameters
    ----------
    test_vecs:
        Array of shape ``(n_speakers, n_features)`` containing embeddings for
        each test speaker.
    enroll_vecs:
        Array of shape ``(n_speakers, n_features)`` containing enrollment
        embeddings for the same speakers in the same order.
    """

    test = np.asarray(test_vecs)
    enroll = np.asarray(enroll_vecs)
    scores = cosine_similarity(test, enroll)
    best = scores.argmax(axis=1)
    successes = np.sum(best == np.arange(test.shape[0]))
    return successes / test.shape[0]
