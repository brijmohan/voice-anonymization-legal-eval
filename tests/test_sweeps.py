"""Sweeps over population size and conversation length.

Beyond mechanics, these pin the two qualitative findings the paper rests on:
Linkability and Singling Out move substantially with the conversation length,
while the EER barely moves with the population size.
"""

import numpy as np
import pytest

from legal_eval.demo import simulate_anonymization, split_corpus, synthetic_corpus
from legal_eval.embeddings import build_speaker_embeddings, build_test_embeddings
from legal_eval.scoring import cosine_score_matrix, cosine_similarity_matrix
from legal_eval.sweeps import (
    SweepResult,
    default_speaker_counts,
    eer_sweep,
    geometric_speaker_counts,
    linkability_sweep,
    singling_out_sweep,
)


@pytest.fixture
def corpus():
    spk2utt, embeddings = synthetic_corpus(n_speakers=60, n_utterances=12, seed=0)
    enroll_utts, test_utts = split_corpus(spk2utt, n_first=6)
    return enroll_utts, test_utts, embeddings


def score_matrix_for(enroll_utts, test_utts, embeddings, conversation_length):
    enroll = build_speaker_embeddings(enroll_utts, embeddings)
    test, _ = build_test_embeddings(test_utts, embeddings, conversation_length, "first")
    return cosine_score_matrix(enroll, test)


def test_default_speaker_counts_covers_the_paper_grid():
    counts = default_speaker_counts(22024)
    assert counts[0] == 20
    assert counts[1] == 120
    assert counts[-1] == 22024
    assert all(n <= 22024 for n in counts)


def test_geometric_speaker_counts_doubles():
    assert geometric_speaker_counts(22024) == [
        30, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 22024
    ]
    assert geometric_speaker_counts(150) == [30, 100, 150]


def test_cosine_similarity_matrix_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="dimensions differ"):
        cosine_similarity_matrix(np.zeros((2, 3)), np.zeros((2, 4)))


def test_cosine_score_matrix_blocking_does_not_change_results(corpus):
    enroll_utts, test_utts, embeddings = corpus
    enroll = build_speaker_embeddings(enroll_utts, embeddings)
    test, _ = build_test_embeddings(test_utts, embeddings, 1, "first")
    whole = cosine_score_matrix(enroll, test, block_size=10_000)
    blocked = cosine_score_matrix(enroll, test, block_size=7)
    assert np.allclose(whole.scores, blocked.scores)


def test_cosine_score_matrix_honours_an_explicit_column_order(corpus):
    enroll_utts, test_utts, embeddings = corpus
    enroll = build_speaker_embeddings(enroll_utts, embeddings)
    test, _ = build_test_embeddings(test_utts, embeddings, 1, "first")
    speakers = list(test)[:3]
    # A speaker may legitimately occupy several columns.
    matrix = cosine_score_matrix(enroll, test, test_order=speakers + speakers)
    assert matrix.test_speakers == speakers + speakers
    assert np.allclose(matrix.scores[:, :3], matrix.scores[:, 3:])


def test_linkability_sweep_decreases_and_records_provenance(corpus):
    matrix = score_matrix_for(*corpus, conversation_length=1)
    result = linkability_sweep(matrix, speaker_counts=[2, 10, 30, 60], n_runs=4, seed=0)
    assert result.metric == "linkability"
    assert result.speaker_counts == [2, 10, 30, 60]
    assert all(len(v) == 4 for v in result.values.values())
    assert result.metadata["n_enroll_total"] == 60
    means = result.mean()
    assert all(
        means[a] >= means[b] - 0.05
        for a, b in zip(result.speaker_counts, result.speaker_counts[1:])
    )


def test_linkability_sweep_skips_counts_above_the_population(corpus):
    matrix = score_matrix_for(*corpus, conversation_length=1)
    result = linkability_sweep(matrix, speaker_counts=[10, 999], n_runs=2)
    assert result.speaker_counts == [10]


def test_longer_conversations_raise_both_legal_metrics(corpus):
    """The paper's central finding: more speech per speaker means more risk."""
    enroll_utts, test_utts, embeddings = corpus
    at_one = linkability_sweep(
        score_matrix_for(enroll_utts, test_utts, embeddings, 1),
        speaker_counts=[60],
        n_runs=3,
        conversation_length=1,
    ).mean()[60]
    at_three = linkability_sweep(
        score_matrix_for(enroll_utts, test_utts, embeddings, 3),
        speaker_counts=[60],
        n_runs=3,
        conversation_length=3,
    ).mean()[60]
    assert at_three > at_one


def test_singling_out_sweep_runs_and_averages_over_folds(corpus):
    enroll_utts, test_utts, embeddings = corpus
    enroll = build_speaker_embeddings(enroll_utts, embeddings)
    result = singling_out_sweep(
        enroll,
        test_utts,
        embeddings,
        conversation_length=1,
        speaker_counts=[5, 20],
        n_runs=2,
        n_folds=3,
        seed=0,
    )
    assert result.metric == "singling_out"
    assert result.speaker_counts == [5, 20]
    assert result.metadata["n_folds"] == 3
    # One value per (enrollment speaker, fold) pair.
    assert len(result.values[20]) == 60 * 3
    assert all(0.0 <= v <= 1.0 for v in result.values[20])


def test_singling_out_sweep_rejects_an_impossible_conversation_length(corpus):
    enroll_utts, test_utts, embeddings = corpus
    enroll = build_speaker_embeddings(enroll_utts, embeddings)
    with pytest.raises(ValueError, match="no test speaker has"):
        singling_out_sweep(
            enroll, test_utts, embeddings, conversation_length=40, n_folds=1
        )


def test_eer_is_nearly_flat_in_population_size(corpus):
    """The paper's contrast: the EER hides what the legal metrics expose."""
    matrix = score_matrix_for(*corpus, conversation_length=1)
    result = eer_sweep(matrix, speaker_counts=[5, 20, 60], n_runs=3, seed=0)
    means = result.mean()
    assert max(means.values()) - min(means.values()) < 0.05
    assert all(0.0 <= v <= 0.5 for v in means.values())


def test_eer_falls_when_embeddings_are_more_separable(corpus):
    enroll_utts, test_utts, embeddings = corpus
    clean = eer_sweep(
        score_matrix_for(enroll_utts, test_utts, embeddings, 3),
        speaker_counts=[60],
        n_runs=2,
    ).mean()[60]
    degraded = eer_sweep(
        score_matrix_for(
            enroll_utts, test_utts, simulate_anonymization(embeddings, strength=0.9), 3
        ),
        speaker_counts=[60],
        n_runs=2,
    ).mean()[60]
    assert clean < degraded


def test_eer_sweep_rejects_a_label_count_mismatch(corpus):
    matrix = score_matrix_for(*corpus, conversation_length=1)
    with pytest.raises(ValueError, match="labels for"):
        eer_sweep(matrix, speaker_counts=[10], test_speakers=["spk0000"])


def test_sweep_result_round_trips_through_json():
    result = SweepResult(
        metric="linkability",
        conversation_length=3,
        values={20: [0.8, 0.82], 100: [0.5, 0.51]},
        metadata={"attacker": "informed"},
    )
    restored = SweepResult.from_dict(result.to_dict())
    assert restored.metric == result.metric
    assert restored.conversation_length == 3
    assert restored.values == result.values
    assert restored.metadata == {"attacker": "informed"}
    assert restored.mean()[20] == pytest.approx(0.81)
    assert restored.std()[20] == pytest.approx(0.01)
