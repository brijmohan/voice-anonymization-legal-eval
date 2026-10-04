"""End-to-end exercise of the command line interface."""

import json

import numpy as np
import pytest

from legal_eval.cli import main
from legal_eval.io import load_spk2utt


@pytest.fixture
def kaldi_dirs(tmp_path):
    """Two Kaldi-style directories with disjoint utterances per speaker."""
    from legal_eval.demo import split_corpus, synthetic_corpus

    spk2utt, embeddings = synthetic_corpus(n_speakers=30, n_utterances=12, seed=0)
    enroll_utts, test_utts = split_corpus(spk2utt, n_first=6)

    paths = {}
    for name, mapping in (("enroll", enroll_utts), ("test", test_utts)):
        directory = tmp_path / name
        directory.mkdir()
        with open(directory / "spk2utt", "w", encoding="utf-8") as handle:
            for speaker, utterances in mapping.items():
                handle.write(f"{speaker} {' '.join(utterances)}\n")
        np.savez(
            directory / "xvector.npz",
            **{utt: embeddings[utt] for utts in mapping.values() for utt in utts},
        )
        paths[name] = directory
    return paths


def test_score_matrix_then_linkability_and_eer(tmp_path, kaldi_dirs, capsys):
    scores = tmp_path / "scores.npy"
    assert main([
        "score-matrix",
        "--enroll-dir", str(kaldi_dirs["enroll"]),
        "--test-dir", str(kaldi_dirs["test"]),
        "--conversation-length", "3",
        "--attacker", "informed",
        "--output", str(scores),
    ]) == 0
    assert scores.exists() and scores.with_suffix(".npy.json").exists()

    sidecar = json.loads(scores.with_suffix(".npy.json").read_text())
    assert sidecar["metadata"]["conversation_length"] == 3
    assert sidecar["metadata"]["attacker"] == "informed"
    assert len(sidecar["enroll_speakers"]) == 30

    link = tmp_path / "linkability_informed_L3.json"
    assert main([
        "linkability", "--score-matrix", str(scores),
        "--speaker-counts", "5,15,30", "--output", str(link),
    ]) == 0
    payload = json.loads(link.read_text())
    assert payload["metric"] == "linkability"
    assert payload["conversation_length"] == 3
    assert sorted(payload["values"], key=int) == ["5", "15", "30"]
    assert "linkability" in capsys.readouterr().out

    eer = tmp_path / "eer_informed_L3.json"
    assert main([
        "eer", "--score-matrix", str(scores),
        "--speaker-counts", "5,30", "--output", str(eer),
    ]) == 0
    assert json.loads(eer.read_text())["metric"] == "eer"


def test_singling_out_command(tmp_path, kaldi_dirs):
    output = tmp_path / "singling_out_informed_L1.json"
    assert main([
        "singling-out",
        "--enroll-dir", str(kaldi_dirs["enroll"]),
        "--test-dir", str(kaldi_dirs["test"]),
        "--conversation-length", "1",
        "--n-enroll-speakers", "10",
        "--n-enroll-utterances", "6",
        "--n-folds", "2",
        "--speaker-counts", "5,20",
        "--output", str(output),
    ]) == 0
    payload = json.loads(output.read_text())
    assert payload["metric"] == "singling_out"
    assert payload["metadata"]["n_folds"] == 2
    assert sorted(payload["values"], key=int) == ["5", "20"]


def test_plot_command_builds_a_figure(tmp_path, kaldi_dirs):
    pytest.importorskip("matplotlib")
    results = tmp_path / "results"
    results.mkdir()
    scores = tmp_path / "scores.npy"
    main([
        "score-matrix", "--enroll-dir", str(kaldi_dirs["enroll"]),
        "--test-dir", str(kaldi_dirs["test"]), "--conversation-length", "1",
        "--output", str(scores),
    ])
    main([
        "linkability", "--score-matrix", str(scores), "--speaker-counts", "5,30",
        "--output", str(results / "linkability_informed_L1.json"),
    ])
    # A file that does not match the naming pattern must be skipped, not fatal.
    (results / "notes.json").write_text("{}", encoding="utf-8")

    figure = tmp_path / "figure.png"
    assert main(["plot", "--results-dir", str(results), "--output", str(figure)]) == 0
    assert figure.exists() and figure.stat().st_size > 0


def test_plot_command_errors_when_nothing_matches(tmp_path):
    results = tmp_path / "empty"
    results.mkdir()
    with pytest.raises(SystemExit, match="no result files"):
        main(["plot", "--results-dir", str(results), "--output", str(tmp_path / "f.png")])


def test_demo_command_writes_results_and_a_figure(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "demo"
    assert main([
        "demo", "--output-dir", str(out), "--n-speakers", "20",
        "--n-utterances", "12", "--n-folds", "2", "--n-runs", "2",
    ]) == 0
    assert (out / "demo_figure.png").exists()
    written = list((out / "results").glob("*.json"))
    # 3 metrics x 3 conditions x 2 conversation lengths
    assert len(written) == 18


def test_demo_drops_conversation_lengths_the_corpus_cannot_support(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "demo"
    # 8 utterances leaves 4 for test, so L=3 (needing 6) is dropped and only L=1 runs.
    assert main([
        "demo", "--output-dir", str(out), "--n-speakers", "20",
        "--n-utterances", "8", "--n-folds", "2", "--n-runs", "2",
    ]) == 0
    assert len(list((out / "results").glob("*.json"))) == 9


def test_demo_rejects_a_corpus_that_is_too_small(tmp_path):
    with pytest.raises(SystemExit, match="at least 2 are needed"):
        main([
            "demo", "--output-dir", str(tmp_path / "d"), "--n-speakers", "20",
            "--n-utterances", "2",
        ])


def test_missing_xvectors_is_reported_clearly(tmp_path):
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "spk2utt").write_text("a a1 a2\n", encoding="utf-8")
    assert load_spk2utt(directory / "spk2utt") == {"a": ["a1", "a2"]}
    with pytest.raises(SystemExit, match="no xvector"):
        main([
            "score-matrix", "--enroll-dir", str(directory),
            "--test-dir", str(directory), "--output", str(tmp_path / "s.npy"),
        ])
