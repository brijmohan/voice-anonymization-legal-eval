"""Tests for the pipe-unsafe audio reader check.

The defect this guards against appeared at four separate sites in one pipeline,
and twice it was found by running the pipeline rather than by reading it. These
tests pin the checker against the real shape of each site, so a fifth site is
caught by the suite rather than by a failed overnight run.

A checker is only worth having if it both fires and stays quiet in the right
places, so the quiet cases are tested as carefully as the loud ones.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "check_scp_readers",
    Path(__file__).resolve().parents[1] / "scripts" / "check_scp_readers.py",
)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def messages(source: str) -> list[str]:
    return [m for _, _, m in check.findings_for_source(source)]


# The four sites that actually cost time.

def test_flags_soundfile_read_as_in_the_asr_extractor():
    """Site 1 and 2: sttts called soundfile.read directly on a wav.scp value."""
    source = "import soundfile as sf\ndef f(p):\n    return sf.read(p)\n"
    assert len(messages(source)) == 1
    assert "sf.read" in messages(source)[0]


def test_flags_torchaudio_load_as_in_speaker_extraction():
    """Site 3: the ASV extractor used torchaudio.load."""
    source = "import torchaudio\ndef f(p):\n    return torchaudio.load(p)\n"
    assert len(messages(source)) == 1
    assert "torchaudio.load" in messages(source)[0]


def test_flags_a_reader_inside_a_helper_reached_from_a_fixed_caller():
    """Site 4: the costly one.

    `_load_audio` is reached from a caller that had already been fixed, and it
    re-reads the file the caller just decoded. Fixing the caller did not fix
    it, so the checker has to see the inner function on its own terms.
    """
    source = (
        "import soundfile as sf\n"
        "class V:\n"
        "    def _load_audio(self, file_path):\n"
        "        audio_input, sr = sf.read(file_path)\n"
        "        return audio_input, sr\n"
    )
    assert len(messages(source)) == 1


def test_flags_literal_index_into_a_wav_scp_token_list():
    """The quiet half: wav_path[1] on a pipe entry yields an ffmpeg flag.

    This is worse than the loud failure because it does not raise where it is
    written.
    """
    source = (
        "def f(info):\n"
        "    wav_path = info['path']\n"
        "    if isinstance(wav_path, list):\n"
        "        wav_path = wav_path[1]\n"
        "    return wav_path\n"
    )
    found = messages(source)
    assert len(found) == 1
    assert "wav_path[1]" in found[0]


def test_flags_a_dotted_reader_chain():
    source = "import scipy.io.wavfile\ndef f(p):\n    return scipy.io.wavfile.read(p)\n"
    assert len(messages(source)) == 1


def test_reports_every_site_not_only_the_first():
    """Two sites, not one, was the exact mistake made twice."""
    source = (
        "import soundfile as sf\n"
        "import torchaudio\n"
        "def a(p):\n"
        "    return sf.read(p)\n"
        "def b(p):\n"
        "    return torchaudio.load(p)\n"
    )
    assert len(messages(source)) == 2


# Staying quiet where it should.

def test_accepts_the_pipe_aware_loader():
    """The fix at every site was to route through load_wav_from_scp."""
    source = (
        "from utils.data_io import load_wav_from_scp\n"
        "def f(entry):\n"
        "    return load_wav_from_scp(entry)\n"
    )
    assert messages(source) == []


def test_accepts_a_guarded_reader_with_a_waiver():
    source = (
        "import soundfile as sf\n"
        "def f(p):\n"
        "    # scp-reader-ok: this path is written by us, never a scp value\n"
        "    return sf.read(p)\n"
    )
    assert messages(source) == []


def test_waiver_on_the_call_line_also_counts():
    source = "import soundfile as sf\nx = sf.read(p)  # scp-reader-ok: fixture\n"
    assert messages(source) == []


def test_ignores_a_reader_named_only_in_a_string_or_comment():
    """An earlier grep-based attempt matched its own documentation."""
    source = (
        '"""Do not call sf.read on a wav.scp value."""\n'
        "# torchaudio.load is not pipe aware\n"
        "MESSAGE = 'soundfile.read raises TypeError'\n"
    )
    assert messages(source) == []


def test_ignores_ordinary_list_indexing():
    """The index rule is narrow on purpose; a wide one got ignored."""
    source = "def f(parts):\n    return parts[1]\n"
    assert messages(source) == []


def test_ignores_a_non_integer_index_on_a_scp_value():
    source = "def f(wav_path):\n    return wav_path['key']\n"
    assert messages(source) == []


def test_unparseable_file_is_reported_rather_than_skipped():
    """A file the checker cannot read must not pass silently."""
    found = check.findings_for_source("def f(:\n", "broken.py")
    assert len(found) == 1
    assert "could not parse" in found[0][2]


# The checker over a tree, which is how CI runs it.

def test_scan_walks_a_directory_and_skips_vendored_trees(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "reader.py").write_text(
        "import soundfile as sf\nx = sf.read(p)\n", encoding="utf-8"
    )
    vendored = tmp_path / "venv" / "lib"
    vendored.mkdir(parents=True)
    (vendored / "other.py").write_text(
        "import soundfile as sf\ny = sf.read(q)\n", encoding="utf-8"
    )
    found = check.scan(tmp_path)
    assert len(found) == 1
    assert "pkg/reader.py" in found[0][0]


def test_main_exits_non_zero_only_for_a_site_in_the_data_path(tmp_path):
    """The first run over a real checkout gave 25 findings, of which 3 mattered.

    A module with no sign of Kaldi data is reported but does not fail the
    check, because a check nobody can act on is a check nobody runs.
    """
    (tmp_path / "vendored.py").write_text(
        "import soundfile as sf\nx = sf.read(p)\n", encoding="utf-8"
    )
    assert check.main(["--root", str(tmp_path)]) == 0

    (tmp_path / "pipeline.py").write_text(
        "import soundfile as sf\n"
        "from utils import read_kaldi_format\n"
        "def f(d):\n"
        "    entry = read_kaldi_format(d / 'wav.scp')['utt']\n"
        "    return sf.read(entry)\n",
        encoding="utf-8",
    )
    assert check.main(["--root", str(tmp_path)]) == 1


def test_main_exits_zero_on_a_clean_tree(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    assert check.main(["--root", str(tmp_path)]) == 0


def test_main_exits_two_on_a_missing_root(tmp_path):
    assert check.main(["--root", str(tmp_path / "absent")]) == 2


def test_this_repository_is_clean():
    """Our own package must not acquire the defect it checks for."""
    root = Path(__file__).resolve().parents[1]
    for sub in ("legal_eval", "scripts", "examples"):
        assert check.scan(root / sub) == []


def test_scan_prunes_corpus_directories_rather_than_filtering_them(tmp_path):
    """A checkout mid-run holds hundreds of thousands of audio files.

    Filtering after the walk made the scan slower than the run it protects, so
    the walk has to refuse to descend.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "reader.py").write_text(
        "import soundfile as sf\nx = sf.read(p)\n", encoding="utf-8"
    )
    for skipped in ("data", "corpora", "exp", "models"):
        d = tmp_path / skipped / "deep"
        d.mkdir(parents=True)
        (d / "ignored.py").write_text(
            "import soundfile as sf\ny = sf.read(q)\n", encoding="utf-8"
        )
    found = check.scan(tmp_path)
    assert len(found) == 1
    assert "src/reader.py" in found[0][0]
    assert check._python_files(tmp_path) == [tmp_path / "src" / "reader.py"]


# Telling a finding worth acting on from one that is not.

def test_a_module_handling_kaldi_data_is_marked_reachable():
    source = (
        "import soundfile as sf\n"
        "from utils import read_kaldi_format\n"
        "def f(d):\n"
        "    scp = read_kaldi_format(d / 'wav.scp')\n"
        "    return sf.read(scp['utt'])\n"
    )
    found = check.findings_for_source(source)
    assert len(found) == 1
    assert found[0][2].startswith("reachable: ")


def test_a_vendored_module_is_marked_review_only():
    """The vendored TTS trees read audio constantly and never see a scp value."""
    source = (
        "import librosa\n"
        "def load_wav(path, sr):\n"
        "    return librosa.load(path, sr=sr)\n"
    )
    found = check.findings_for_source(source)
    assert len(found) == 1
    assert found[0][2].startswith("review: ")


def test_in_data_path_detects_each_marker():
    for marker in check.DATA_PATH_MARKERS:
        assert check.in_data_path(f"x = '{marker}'"), marker
    assert not check.in_data_path("x = 'nothing to do with it'")


def test_an_unparseable_module_is_always_counted():
    """A file we cannot read is not evidence that it is safe."""
    found = check.findings_for_source("def f(:\n", "broken.py")
    assert found[0][2].startswith("reachable: ")
