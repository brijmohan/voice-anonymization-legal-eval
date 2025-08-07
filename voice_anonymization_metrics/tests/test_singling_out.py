import numpy as np

from voice_anonymization_metrics.singling_out import singling_out_probability


def unit_vector_for_similarity(sim):
    return np.array([sim, np.sqrt(1 - sim ** 2)])


def test_singling_out_success():
    enroll = unit_vector_for_similarity(1.0)
    calib = np.array([unit_vector_for_similarity(s) for s in [0.95, 0.85, 0.75, 0.65, 0.55]])
    test = np.array([unit_vector_for_similarity(s) for s in [0.92, 0.8, 0.7, 0.6, 0.5]])
    prob = singling_out_probability(test, enroll, calib, target_prob=0.2)
    assert np.isclose(prob, 1.0)


def test_singling_out_failure():
    enroll = unit_vector_for_similarity(1.0)
    calib = np.array([unit_vector_for_similarity(s) for s in [0.95, 0.85, 0.75, 0.65, 0.55]])
    # two test vectors exceed the threshold
    test = np.array([unit_vector_for_similarity(s) for s in [0.92, 0.91, 0.7, 0.6, 0.5]])
    prob = singling_out_probability(test, enroll, calib, target_prob=0.2)
    assert np.isclose(prob, 0.0)
