import numpy as np

from voice_anonymization_metrics.utils import cosine_similarity


def test_cosine_similarity_basic():
    a = np.array([1, 0])
    b = np.array([0, 1])
    sim = cosine_similarity(a, b)
    assert sim.shape == (1, 1)
    assert np.isclose(sim[0, 0], 0.0)

    a2 = np.array([[1, 0], [0, 1]])
    b2 = np.array([[1, 0], [0, 1]])
    sim2 = cosine_similarity(a2, b2)
    assert np.allclose(sim2, np.eye(2))
