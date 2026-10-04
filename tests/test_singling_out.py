"""Singling Out correctness.

The headline property is the one the PSO framework is built on: a predicate
calibrated to fire on 1/N of a calibration set isolates exactly one of N test
items about ``exp(-1)`` of the time when it carries no speaker information. That
is the yardstick the paper's curves are read against, so it is tested directly.

A deliberately naive oracle re-derives the threshold and the isolation decision
from the definitions, independently of the vectorised implementation.
"""

import math

import numpy as np
import pytest

from legal_eval.demo import simulate_anonymization, split_corpus, synthetic_corpus
from legal_eval.embeddings import build_conversations, build_speaker_embeddings
from legal_eval.metrics.singling_out import (
    TRIVIAL_SINGLING_OUT,
    isolation_threshold,
    score_conversations,
    singling_out,
)


def naive_threshold(calibration_scores, n_expected_above):
    """Threshold from the definition: midpoint of the n-th and (n+1)-th highest."""
    ordered = sorted(calibration_scores, reverse=True)
    n = n_expected_above
    return (ordered[n - 1] + ordered[n]) / 2.0


def naive_isolation(enroll_embedding, selected, conversations):
    """Isolation outcome from the definition, using plain Python loops."""
    import numpy as np

    def similarity(a, b):
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        )

    calibration = [
        similarity(enroll_embedding, conv)
        for speaker in selected
        for conv in conversations[speaker].calibration
    ]
    own = selected[0]
    threshold = naive_threshold(calibration, conversations[own].n_calibration)
    firing = sum(
        1
        for speaker in selected
        if similarity(enroll_embedding, conversations[speaker].test) > threshold
    )
    return 1 if firing == 1 else 0


def test_isolation_threshold_matches_definition():
    rng = np.random.default_rng(0)
    for n in (1, 3, 9):
        scores = rng.normal(size=50)
        assert isolation_threshold(scores, n) == pytest.approx(naive_threshold(scores, n))


def test_isolation_threshold_lets_exactly_n_scores_through():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=80)
    for n in (1, 5, 9):
        threshold = isolation_threshold(scores, n)
        assert np.count_nonzero(scores > threshold) == n


def test_isolation_threshold_rejects_too_few_scores():
    with pytest.raises(ValueError, match="at least"):
        isolation_threshold(np.array([0.1, 0.2]), n_expected_above=5)
    with pytest.raises(ValueError, match="at least 1"):
        isolation_threshold(np.array([0.1, 0.2]), n_expected_above=0)


def test_singling_out_matches_naive_oracle():
    """Vectorised outcomes must equal the definition, speaker by speaker."""
    spk2utt, embeddings = synthetic_corpus(n_speakers=24, n_utterances=8, seed=2)
    enroll_utts, test_utts = split_corpus(spk2utt, n_first=4)
    enroll = build_speaker_embeddings(enroll_utts, embeddings)
    conversations = build_conversations(test_utts, embeddings, conversation_length=1, seed=3)
    scores = score_conversations(enroll, conversations)

    n_test_speakers = 6
    outcomes = singling_out(scores, n_test_speakers=n_test_speakers, n_runs=3, seed=5, fold=0)

    dataset_speakers = scores.dataset_speakers
    enroll_rows = {s: i for i, s in enumerate(scores.enroll_speakers)}
    dataset_index = scores.dataset_index

    for speaker, values in outcomes.items():
        own_column = dataset_index[speaker]
        row = enroll_rows[speaker]
        for run, value in enumerate(values):
            # Reproduce the same draw the implementation makes.
            rng = np.random.default_rng((5, n_test_speakers, 0, row, run))
            others = rng.choice(len(dataset_speakers) - 1, size=n_test_speakers - 1, replace=False)
            others = np.where(others >= own_column, others + 1, others)
            selected = [dataset_speakers[c] for c in [own_column, *others]]
            assert value == naive_isolation(enroll[speaker], selected, conversations)


def test_trivial_attacker_isolates_at_exp_minus_one():
    """A predicate carrying no speaker information must land near exp(-1).

    This is the PSO baseline the paper compares against. Enrollment embeddings are
    drawn independently of the test corpus, so the calibrated predicate is
    effectively random, and isolation should occur about 37% of the time.
    """
    rng = np.random.default_rng(7)
    spk2utt, embeddings = synthetic_corpus(n_speakers=120, n_utterances=12, seed=8)
    conversations = build_conversations(spk2utt, embeddings, conversation_length=1, seed=9)

    dim = len(next(iter(embeddings.values())))
    unrelated = rng.normal(size=(60, dim))
    unrelated /= np.linalg.norm(unrelated, axis=1, keepdims=True)
    # Name them after real test speakers so each has its own calibration block,
    # while the embedding itself carries no information about that speaker.
    names = list(conversations.keys())[:60]
    enroll = {name: unrelated[i] for i, name in enumerate(names)}

    scores = score_conversations(enroll, conversations)
    outcomes = singling_out(scores, n_test_speakers=20, n_runs=20, seed=10)
    observed = float(np.mean([v for values in outcomes.values() for v in values]))
    assert observed == pytest.approx(TRIVIAL_SINGLING_OUT, abs=0.05)


def test_informative_attacker_beats_the_trivial_baseline():
    """A predicate built from the speaker's own data must isolate more often."""
    spk2utt, embeddings = synthetic_corpus(
        n_speakers=100, n_utterances=12, within_speaker_std=1.0, seed=11
    )
    enroll_utts, test_utts = split_corpus(spk2utt, n_first=6)
    enroll = build_speaker_embeddings(enroll_utts, embeddings)
    conversations = build_conversations(test_utts, embeddings, conversation_length=1, seed=12)
    scores = score_conversations(enroll, conversations)

    outcomes = singling_out(scores, n_test_speakers=20, n_runs=10, seed=13)
    observed = float(np.mean([v for values in outcomes.values() for v in values]))
    assert observed > TRIVIAL_SINGLING_OUT + 0.15


def test_degrading_embeddings_lowers_singling_out():
    spk2utt, embeddings = synthetic_corpus(
        n_speakers=100, n_utterances=12, within_speaker_std=1.0, seed=14
    )
    enroll_utts, test_utts = split_corpus(spk2utt, n_first=6)

    def measure(source):
        enroll = build_speaker_embeddings(enroll_utts, source)
        conversations = build_conversations(test_utts, source, conversation_length=1, seed=15)
        scores = score_conversations(enroll, conversations)
        outcomes = singling_out(scores, n_test_speakers=20, n_runs=10, seed=16)
        return float(np.mean([v for values in outcomes.values() for v in values]))

    original = measure(embeddings)
    degraded = measure(simulate_anonymization(embeddings, strength=0.95, seed=17))
    assert degraded < original


def test_build_conversations_uses_disjoint_utterances():
    spk2utt, embeddings = synthetic_corpus(n_speakers=5, n_utterances=9, seed=18)
    conversations = build_conversations(
        spk2utt, embeddings, conversation_length=2, max_calibration=9, seed=19
    )
    for _speaker, conv in conversations.items():
        assert conv.n_calibration == min(9, math.comb(9 - 2, 2))
        assert conv.test.shape == (32,)
        assert np.isclose(np.linalg.norm(conv.test), 1.0)


def test_build_conversations_skips_speakers_with_too_few_utterances():
    spk2utt, embeddings = synthetic_corpus(n_speakers=4, n_utterances=3, seed=20)
    assert build_conversations(spk2utt, embeddings, conversation_length=1) != {}
    # A length-2 conversation needs 4 utterances to leave a calibration set.
    assert build_conversations(spk2utt, embeddings, conversation_length=2) == {}


def test_folds_differ_by_default_and_can_be_frozen():
    spk2utt, embeddings = synthetic_corpus(n_speakers=6, n_utterances=10, seed=21)
    first = build_conversations(spk2utt, embeddings, 2, seed=0, fold=0)
    second = build_conversations(spk2utt, embeddings, 2, seed=0, fold=1)
    assert not np.allclose(first["spk0000"].test, second["spk0000"].test)

    frozen_a = build_conversations(spk2utt, embeddings, 2, seed=0, fold=0, vary_folds=False)
    frozen_b = build_conversations(spk2utt, embeddings, 2, seed=0, fold=1, vary_folds=False)
    assert np.allclose(frozen_a["spk0000"].test, frozen_b["spk0000"].test)


def test_rejects_out_of_range_set_size():
    spk2utt, embeddings = synthetic_corpus(n_speakers=8, n_utterances=8, seed=22)
    enroll = build_speaker_embeddings(spk2utt, embeddings)
    conversations = build_conversations(spk2utt, embeddings, 1, seed=23)
    scores = score_conversations(enroll, conversations)
    with pytest.raises(ValueError):
        singling_out(scores, n_test_speakers=1)
    with pytest.raises(ValueError):
        singling_out(scores, n_test_speakers=99)
