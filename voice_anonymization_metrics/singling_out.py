"""Implementation of the Singling Out metric."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .calibration import calibrate_threshold
from .utils import cosine_similarity


def singling_out_probability(
    test_vecs: ArrayLike,
    enroll_vecs: ArrayLike,
    calib_vecs: ArrayLike,
    target_prob: float,
) -> float:
    """Compute isolation probability for a set of enrollment speakers.

    Parameters
    ----------
    test_vecs:
        Array of shape ``(n_speakers, n_features)`` with one embedding per
        speaker in the test subset.
    enroll_vecs:
        Array of enrollment embeddings, one per speaker used to build
        predicates.
    calib_vecs:
        Calibration embeddings used to derive thresholds.  They may include
        multiple embeddings per speaker.
    target_prob:
        Desired probability of predicate being true for a random calibration
        vector.  In the paper this is typically ``1/N`` where ``N`` is the
        number of speakers under evaluation.
    """

    test = np.asarray(test_vecs)
    enroll = np.asarray(enroll_vecs)
    calib = np.asarray(calib_vecs)
    successes = 0
    for e in enroll:
        thresh = calibrate_threshold(e, calib, target_prob)
        scores = cosine_similarity(test, e).ravel()
        if np.sum(scores > thresh) == 1:
            successes += 1
    return successes / len(enroll)
