"""Tests for the anonymized-output audio check.

The failure this guards is the one that reads as success: a system that emits
silence scores perfect privacy, because a silent embedding sits at chance
distance from every enrolled speaker. So the tests that matter are the ones
where the output is wrong and the check has to say so.

The second concern is the sample. A deep check over a few thousand files cannot
prove a quarter of a million are sound, and the script must not be able to
drift into claiming it did. The sample is seeded, and that is pinned here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

SPEC = importlib.util.spec_from_file_location(
    "check_anonymized_output",
    Path(__file__).resolve().parents[1] / "scripts"
    / "check_anonymized_output.py",
)
cao = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cao)

RATE = 16000
TAG = "sometag"


def _speech(rng, seconds=1.0):
    """Noise at speech-like amplitude, which the check should accept."""
    return rng.normal(0, 0.1, int(RATE * seconds)).clip(-0.99, 0.99)


def _build(root: Path, name: str, signals: dict[str, np.ndarray]) -> None:
    """Write a data directory with a wav.scp holding relative paths."""
    directory = root / "data" / name
    (directory / "wav").mkdir(parents=True, exist_ok=True)
    lines = []
    for utt, signal in signals.items():
        relative = Path("data") / name / "wav" / f"{utt}.wav"
        soundfile.write(root / relative, signal, RATE)
        lines.append(f"{utt} {relative}")
    (directory / "wav.scp").write_text("\n".join(lines) + "\n")


def _corpus(tmp_path, anon_signals, orig_signals=None):
    rng = np.random.default_rng(0)
    utts = list(anon_signals)
    originals = orig_signals or {u: _speech(rng) for u in utts}
    _build(tmp_path, "ds", originals)
    _build(tmp_path, f"ds_{TAG}", anon_signals)
    return tmp_path


def _run(root, extra=()):
    out = root / "report.json"
    code = cao.main(["--tag", TAG, "--datasets", "ds", "--root", str(root),
                     "--output", str(out), *extra])
    return code, json.loads(out.read_text())


# The scp reader and the pipe hazard.

def test_a_piped_scp_entry_is_skipped_rather_than_opened():
    """A Kaldi extended filename is a command. Handing it to a reader as a path
    is a defect this project has hit at four separate sites."""
    assert cao.resolve("gunzip -c a.gz |", Path("/root")) is None


def test_a_relative_path_resolves_against_the_root_not_the_scp():
    assert cao.resolve("data/x/wav/a.wav", Path("/root")) == Path(
        "/root/data/x/wav/a.wav")


def test_an_absolute_path_is_left_alone():
    assert cao.resolve("/elsewhere/a.wav", Path("/root")) == Path(
        "/elsewhere/a.wav")


def test_read_scp_splits_on_the_first_field_only(tmp_path):
    """A pipe entry contains spaces, so splitting on every space loses it."""
    scp = tmp_path / "wav.scp"
    scp.write_text("a sox a.wav -t wav - |\nb plain.wav\n")
    assert cao.read_scp(scp) == {"a": "sox a.wav -t wav - |", "b": "plain.wav"}


# Output that is speech.

def test_speech_output_passes_both_passes(tmp_path):
    rng = np.random.default_rng(1)
    code, report = _run(_corpus(tmp_path, {f"u{i}": _speech(rng)
                                           for i in range(12)}))
    assert code == 0
    body = report[f"ds_{TAG}"]
    assert body["unusable"] == []
    assert body["undersized"] == []
    assert body["missing"] == []
    assert body["active_median"] > 0.5


# Output that is not speech, which is the whole point.

def test_silence_is_caught(tmp_path):
    """The McAdams failure: a file of the right length that is digital silence."""
    rng = np.random.default_rng(2)
    signals = {f"u{i}": _speech(rng) for i in range(11)}
    signals["u0"] = np.zeros(RATE)
    code, report = _run(_corpus(tmp_path, signals))
    assert code == 0
    unusable = report[f"ds_{TAG}"]["unusable"]
    assert [row["utterance"] for row in unusable] == ["u0"]
    assert "silence" in unusable[0]["reason"]


def test_a_collapse_against_the_original_is_caught(tmp_path):
    """Audible in, almost nothing out, which is removal rather than change."""
    rng = np.random.default_rng(3)
    originals = {f"u{i}": _speech(rng) for i in range(11)}
    anonymized = dict(originals)
    collapsed = np.zeros(RATE)
    collapsed[:60] = 0.5
    anonymized["u3"] = collapsed
    code, report = _run(_corpus(tmp_path, anonymized, originals))
    assert code == 0
    assert [r["utterance"] for r in report[f"ds_{TAG}"]["unusable"]] == ["u3"]


def test_impulses_at_full_scale_are_caught(tmp_path):
    rng = np.random.default_rng(4)
    signals = {f"u{i}": _speech(rng) for i in range(11)}
    spikes = rng.normal(0, 0.1, RATE)
    spikes[::3] = 1.0
    signals["u5"] = spikes
    code, report = _run(_corpus(tmp_path, signals))
    assert code == 0
    reasons = {r["utterance"]: r["reason"]
               for r in report[f"ds_{TAG}"]["unusable"]}
    assert "u5" in reasons
    assert "full scale" in reasons["u5"]


def test_a_truncated_file_is_caught_by_the_size_pass(tmp_path):
    """A sample cannot find truncation, so every file is sized."""
    rng = np.random.default_rng(5)
    signals = {f"u{i}": _speech(rng) for i in range(11)}
    signals["u7"] = _speech(rng, seconds=0.05)
    code, report = _run(_corpus(tmp_path, signals))
    assert code == 0
    assert [r["utterance"]
            for r in report[f"ds_{TAG}"]["undersized"]] == ["u7"]


def test_a_missing_file_is_reported_rather_than_raising(tmp_path):
    rng = np.random.default_rng(6)
    root = _corpus(tmp_path, {f"u{i}": _speech(rng) for i in range(11)})
    (root / "data" / f"ds_{TAG}" / "wav" / "u2.wav").unlink()
    code, report = _run(root)
    assert code == 0
    assert report[f"ds_{TAG}"]["missing"] == ["u2"]


# The sample, which must stay honest about what it covers.

def test_the_sample_is_capped_and_recorded(tmp_path):
    rng = np.random.default_rng(7)
    code, report = _run(_corpus(tmp_path, {f"u{i:03d}": _speech(rng, 0.2)
                                           for i in range(40)}),
                        extra=("--sample", "9", "--seed", "3"))
    assert code == 0
    body = report[f"ds_{TAG}"]
    assert body["utterances"] == 40
    assert body["sampled"] == 9
    assert body["sample_seed"] == 3


def test_the_same_seed_draws_the_same_sample(tmp_path):
    """Otherwise the claim made about the output is not reproducible."""
    rng = np.random.default_rng(8)
    signals = {f"u{i:03d}": _speech(rng, 0.2) for i in range(40)}
    # One bad file, and whether it is drawn must depend only on the seed.
    signals["u013"] = np.zeros(RATE)
    root = _corpus(tmp_path, signals)
    first = _run(root, extra=("--sample", "20", "--seed", "11"))[1]
    second = _run(root, extra=("--sample", "20", "--seed", "11"))[1]
    assert first == second


def test_a_dataset_with_no_anonymized_scp_is_skipped_not_fatal(tmp_path):
    rng = np.random.default_rng(9)
    root = _corpus(tmp_path, {f"u{i}": _speech(rng) for i in range(11)})
    out = root / "r.json"
    code = cao.main(["--tag", TAG, "--datasets", "ds", "absent",
                     "--root", str(root), "--output", str(out)])
    assert code == 0
    assert sorted(json.loads(out.read_text())) == [f"ds_{TAG}"]


def test_nothing_to_check_is_an_error(tmp_path):
    """Reporting success over zero files is how a broken path reads as clean."""
    assert cao.main(["--tag", TAG, "--datasets", "absent",
                     "--root", str(tmp_path)]) == 1
