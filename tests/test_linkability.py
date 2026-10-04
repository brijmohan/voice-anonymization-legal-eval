"""Linkability correctness.

The fast path replaces explicit candidate-set sampling with one hypergeometric
draw per test speaker. These tests pin that the two agree, both exactly in
distribution on small cases and statistically on larger ones, and that the metric
behaves as the definition requires.
"""

import numpy as np
import pytest

from legal_eval.demo import simulate_anonymization, split_corpus, synthetic_corpus
from legal_eval.embeddings import build_speaker_embeddings, build_test_embeddings
from legal_eval.io import ScoreMatrix
from legal_eval.metrics.linkability import (
    count_beaters,
    linkability,
    linkability_naive,
)
from legal_eval.scoring import cosine_score_matrix


def make_score_matrix(n_enroll=40, n_test=25, seed=0):
    """A score matrix whose first ``n_test`` enrollment speakers are the test ones."""
    rng = np.random.default_rng(seed)
    enroll = [f"spk{i:03d}" for i in range(n_enroll)]
    test = enroll[:n_test]
    scores = rng.normal(size=(n_enroll, n_test))
    # Give the true speaker a genuine advantage, as a real system would.
    scores[np.arange(n_test), np.arange(n_test)] += 1.5
    return ScoreMatrix(scores, enroll, test, {})


def test_count_beaters_matches_direct_count():
    matrix = make_score_matrix()
    beaters = count_beaters(matrix)
    for column, speaker in enumerate(matrix.test_speakers):
        true_row = matrix.enroll_index[speaker]
        true_score = matrix.scores[true_row, column]
        expected = sum(
            1
            for row in range(matrix.scores.shape[0])
            if row != true_row and matrix.scores[row, column] > true_score
        )
        assert beaters[column] == expected


def test_exact_estimator_matches_mean_of_many_naive_runs():
    """The closed form is the expectation the sampling estimator converges to."""
    matrix = make_score_matrix(n_enroll=30, n_test=20, seed=3)
    exact = linkability(matrix, n_enroll_speakers=8, estimator="exact")[0]
    naive = linkability_naive(matrix, n_enroll_speakers=8, n_runs=400, seed=11)
    assert np.mean(naive) == pytest.approx(exact, abs=0.02)


def test_sampling_estimator_matches_naive_sampling():
    """Both draw from the same distribution, so their means agree."""
    matrix = make_score_matrix(n_enroll=50, n_test=40, seed=4)
    for count in (2, 5, 20, 50):
        fast = linkability(matrix, n_enroll_speakers=count, n_runs=200, seed=7)
        naive = linkability_naive(matrix, n_enroll_speakers=count, n_runs=200, seed=7)
        assert np.mean(fast) == pytest.approx(np.mean(naive), abs=0.02)
        assert np.std(fast) == pytest.approx(np.std(naive), abs=0.02)


def test_full_population_is_deterministic_and_exact():
    """With every enrollment speaker in the candidate set there is nothing to draw."""
    matrix = make_score_matrix(n_enroll=30, n_test=20, seed=5)
    beaters = count_beaters(matrix)
    expected = float(np.mean(beaters == 0))

    sampled = linkability(matrix, n_enroll_speakers=30, n_runs=4, seed=0)
    assert sampled == pytest.approx([expected] * 4)
    assert linkability(matrix, n_enroll_speakers=30, estimator="exact")[0] == pytest.approx(
        expected
    )
    assert linkability_naive(matrix, n_enroll_speakers=30, n_runs=2) == pytest.approx(
        [expected] * 2
    )


def test_single_candidate_always_links():
    matrix = make_score_matrix()
    assert linkability(matrix, n_enroll_speakers=1, n_runs=3) == pytest.approx([1.0] * 3)


def test_linkability_decreases_with_population_size():
    matrix = make_score_matrix(n_enroll=200, n_test=100, seed=6)
    curve = [
        linkability(matrix, n_enroll_speakers=n, estimator="exact")[0]
        for n in (2, 10, 50, 200)
    ]
    assert all(earlier >= later - 1e-12 for earlier, later in zip(curve, curve[1:]))


def test_chance_level_for_uninformative_scores():
    """Random scores must give the 1/N' chance level the paper plots."""
    rng = np.random.default_rng(8)
    enroll = [f"spk{i:03d}" for i in range(400)]
    matrix = ScoreMatrix(rng.normal(size=(400, 400)), enroll, list(enroll), {})
    for count in (4, 20, 100):
        value = linkability(matrix, n_enroll_speakers=count, estimator="exact")[0]
        assert value == pytest.approx(1.0 / count, rel=0.25)


def test_anonymization_reduces_linkability_end_to_end():
    spk2utt, utt2embedding = synthetic_corpus(n_speakers=80, n_utterances=10, seed=0)
    enroll_utts, test_utts = split_corpus(spk2utt, n_first=6)

    def curve(embeddings):
        enroll = build_speaker_embeddings(enroll_utts, embeddings)
        test, _ = build_test_embeddings(test_utts, embeddings, 1, "first")
        matrix = cosine_score_matrix(enroll, test)
        return linkability(matrix, n_enroll_speakers=80, estimator="exact")[0]

    original = curve(utt2embedding)
    anonymized = curve(simulate_anonymization(utt2embedding, strength=0.9))
    assert original > 0.3
    assert anonymized < original / 2


def test_rejects_test_speaker_absent_from_enrollment():
    matrix = make_score_matrix()
    broken = ScoreMatrix(
        matrix.scores, matrix.enroll_speakers, ["ghost"] * len(matrix.test_speakers), {}
    )
    with pytest.raises(KeyError, match="absent from the enrollment set"):
        count_beaters(broken)


def test_rejects_out_of_range_population():
    matrix = make_score_matrix(n_enroll=10, n_test=5)
    with pytest.raises(ValueError):
        linkability(matrix, n_enroll_speakers=11)
    with pytest.raises(ValueError):
        linkability(matrix, n_enroll_speakers=0)


def test_rejects_label_count_mismatch():
    matrix = make_score_matrix(n_enroll=10, n_test=5)
    with pytest.raises(ValueError, match="cover every column"):
        count_beaters(matrix, test_speakers=["spk000"])


def test_results_are_reproducible_from_the_seed():
    matrix = make_score_matrix()
    first = linkability(matrix, n_enroll_speakers=10, n_runs=5, seed=42)
    second = linkability(matrix, n_enroll_speakers=10, n_runs=5, seed=42)
    assert first == second
    assert linkability(matrix, n_enroll_speakers=10, n_runs=5, seed=43) != first
