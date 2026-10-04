"""Averaging conventions, conversation construction and the on-disk formats."""

import json

import numpy as np
import pytest

from legal_eval.embeddings import (
    average_embeddings,
    build_speaker_embeddings,
    build_test_embeddings,
    l2_normalize,
)
from legal_eval.io import (
    ScoreMatrix,
    load_score_matrix,
    load_spk2utt,
    load_utt2spk,
    load_xvectors,
    read_results,
    save_score_matrix,
    utterances_of,
    write_results,
)


def test_l2_normalize_handles_the_zero_vector():
    assert np.allclose(l2_normalize(np.array([3.0, 4.0])), [0.6, 0.8])
    assert np.allclose(l2_normalize(np.zeros(3)), np.zeros(3))


def test_single_embedding_is_returned_unchanged():
    """The extractor already normalised it; averaging must not touch it again."""
    vector = np.array([0.6, 0.8])
    assert np.allclose(average_embeddings([vector]), vector)
    # Even an unnormalised lone vector passes through, matching the original code.
    unnormalised = np.array([3.0, 4.0])
    assert np.allclose(average_embeddings([unnormalised]), unnormalised)


def test_averaging_several_embeddings_normalises_once():
    embeddings = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    averaged = average_embeddings(embeddings)
    assert np.isclose(np.linalg.norm(averaged), 1.0)
    assert np.allclose(averaged, [np.sqrt(0.5), np.sqrt(0.5)])


def test_averaging_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="empty"):
        average_embeddings([])


def test_build_speaker_embeddings_averages_every_utterance():
    spk2utt = {"a": ["a1", "a2"], "b": ["b1"]}
    utt2embedding = {
        "a1": np.array([1.0, 0.0]),
        "a2": np.array([0.0, 1.0]),
        "b1": np.array([0.0, 1.0]),
    }
    result = build_speaker_embeddings(spk2utt, utt2embedding)
    assert np.allclose(result["a"], [np.sqrt(0.5), np.sqrt(0.5)])
    assert np.allclose(result["b"], [0.0, 1.0])


def test_reference_selection_reproduces_the_original_asymmetry():
    """L=1 draws at random; L>1 takes the first L. Both are kept deliberately."""
    spk2utt = {"a": [f"a{i}" for i in range(5)]}
    utt2embedding = {f"a{i}": np.eye(5)[i] for i in range(5)}

    _, chosen = build_test_embeddings(
        spk2utt, utt2embedding, 3, "reference", np.random.default_rng(0)
    )
    assert chosen["a"] == ["a0", "a1", "a2"]

    _, chosen = build_test_embeddings(
        spk2utt, utt2embedding, 1, "reference", np.random.default_rng(0)
    )
    assert len(chosen["a"]) == 1

    picks = {
        tuple(
            build_test_embeddings(
                spk2utt, utt2embedding, 1, "reference", np.random.default_rng(seed)
            )[1]["a"]
        )
        for seed in range(30)
    }
    assert len(picks) > 1, "L=1 must vary with the seed"


def test_first_selection_needs_no_rng():
    spk2utt = {"a": ["a0", "a1"]}
    utt2embedding = {"a0": np.array([1.0, 0.0]), "a1": np.array([0.0, 1.0])}
    _, chosen = build_test_embeddings(spk2utt, utt2embedding, 1, "first")
    assert chosen["a"] == ["a0"]


def test_random_selection_requires_an_rng():
    spk2utt = {"a": ["a0", "a1"]}
    utt2embedding = {"a0": np.array([1.0, 0.0]), "a1": np.array([0.0, 1.0])}
    with pytest.raises(ValueError, match="requires an rng"):
        build_test_embeddings(spk2utt, utt2embedding, 1, "random")


def test_rejects_unknown_selection_and_bad_length():
    spk2utt = {"a": ["a0"]}
    utt2embedding = {"a0": np.array([1.0])}
    with pytest.raises(ValueError, match="selection must be"):
        build_test_embeddings(spk2utt, utt2embedding, 1, "nonsense")
    with pytest.raises(ValueError, match="at least 1"):
        build_test_embeddings(spk2utt, utt2embedding, 0, "first")


def test_rejects_speaker_with_too_few_utterances():
    spk2utt = {"a": ["a0"]}
    utt2embedding = {"a0": np.array([1.0, 0.0])}
    with pytest.raises(ValueError, match="fewer than the conversation length"):
        build_test_embeddings(spk2utt, utt2embedding, 3, "first")


def test_spk2utt_round_trip(tmp_path):
    path = tmp_path / "spk2utt"
    path.write_text("a a1 a2 a3\nb b1\n\nc c1 c2\n", encoding="utf-8")
    assert load_spk2utt(path) == {"a": ["a1", "a2", "a3"], "b": ["b1"], "c": ["c1", "c2"]}
    assert load_spk2utt(path, min_utterances=2) == {
        "a": ["a1", "a2", "a3"],
        "c": ["c1", "c2"],
    }
    assert utterances_of(load_spk2utt(path)) == ["a1", "a2", "a3", "b1", "c1", "c2"]


def test_utt2spk_round_trip(tmp_path):
    path = tmp_path / "utt2spk"
    path.write_text("a1 a\na2 a\nb1 b\n", encoding="utf-8")
    assert load_utt2spk(path) == {"a1": "a", "a2": "a", "b1": "b"}


def test_load_xvectors_from_npz(tmp_path):
    path = tmp_path / "xvectors.npz"
    np.savez(path, a1=np.array([[1.0, 2.0]]), a2=np.array([3.0, 4.0]))
    loaded = load_xvectors(path)
    assert set(loaded) == {"a1", "a2"}
    assert loaded["a1"].shape == (2,), "a leading singleton axis must be squeezed"
    assert np.allclose(load_xvectors(path, ["a2"])["a2"], [3.0, 4.0])


def test_load_xvectors_from_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "xvector.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("a1", data=np.array([[1.0, 2.0]]))
        handle.create_dataset("a2", data=np.array([3.0, 4.0]))
    loaded = load_xvectors(path)
    assert set(loaded) == {"a1", "a2"}
    assert loaded["a1"].shape == (2,)
    assert set(load_xvectors(path, ["a1"])) == {"a1"}


def test_score_matrix_round_trip(tmp_path):
    matrix = ScoreMatrix(
        np.arange(6.0).reshape(3, 2), ["e0", "e1", "e2"], ["t0", "t1"], {"L": 3}
    )
    path = tmp_path / "scores.npy"
    save_score_matrix(matrix, path)
    reloaded = load_score_matrix(path)
    assert np.allclose(reloaded.scores, matrix.scores)
    assert reloaded.enroll_speakers == matrix.enroll_speakers
    assert reloaded.test_speakers == matrix.test_speakers
    assert reloaded.metadata == {"L": 3}
    assert reloaded.enroll_index == {"e0": 0, "e1": 1, "e2": 2}


def test_score_matrix_validates_its_labels():
    with pytest.raises(ValueError, match="score matrix has shape"):
        ScoreMatrix(np.zeros((2, 2)), ["e0"], ["t0", "t1"], {})


def test_results_round_trip_converts_numpy_scalars(tmp_path):
    path = tmp_path / "results.json"
    write_results({"a": np.float64(0.5), "b": np.int64(3), "c": np.arange(2)}, path)
    assert read_results(path) == {"a": 0.5, "b": 3, "c": [0, 1]}
    assert json.loads(path.read_text())["a"] == 0.5
