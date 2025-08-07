import numpy as np

from voice_anonymization_metrics.linkability import linkability_probability


def test_linkability_perfect():
    enroll = np.array([[1, 0], [0, 1], [-1, 0]])
    test = enroll.copy()
    prob = linkability_probability(test, enroll)
    assert np.isclose(prob, 1.0)


def test_linkability_mismatch():
    enroll = np.array([[1, 0], [0, 1], [-1, 0]])
    test = np.array([[0, 1], [1, 0], [-1, 0]])
    prob = linkability_probability(test, enroll)
    assert np.isclose(prob, 1 / 3)
