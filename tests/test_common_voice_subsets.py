"""Rebuilding the paper's Common Voice subsets from the published file lists."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_common_voice_subsets.py"
spec = importlib.util.spec_from_file_location("build_cv", SCRIPT)
build_cv = importlib.util.module_from_spec(spec)
sys.modules["build_cv"] = build_cv
spec.loader.exec_module(build_cv)


@pytest.fixture
def corpus(tmp_path):
    """A miniature Common Voice locale directory."""
    root = tmp_path / "en"
    (root / "clips").mkdir(parents=True)

    rows = []
    for speaker, gender, n in (("c_alice", "female", 4), ("c_bob", "male", 3),
                               ("c_carol", "", 2)):
        for i in range(n):
            clip = f"common_voice_en_{speaker}_{i}.mp3"
            (root / "clips" / clip).write_bytes(b"not really audio")
            rows.append({"client_id": speaker, "path": clip, "sentence": "x",
                         "up_votes": "2", "down_votes": "0", "age": "",
                         "gender": gender, "accents": "", "locale": "en",
                         "segment": ""})

    with open(root / "validated.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    filelist = tmp_path / "filelist"
    filelist.write_text(
        "".join(f"/home/ubuntu/CORPUS/cv-corpus-11.0-2022-09-21/en/clips/{r['path']}\n"
                for r in rows),
        encoding="utf-8",
    )
    return root, filelist


def test_filelist_reader_keeps_only_the_basename(tmp_path):
    path = tmp_path / "list"
    path.write_text("/a/b/c/common_voice_en_1.mp3\n/x/common_voice_en_2.mp3\n\n",
                    encoding="utf-8")
    assert build_cv.read_filelist(path) == ["common_voice_en_1.mp3",
                                            "common_voice_en_2.mp3"]


def test_groups_clips_by_speaker_and_maps_gender(corpus):
    root, filelist = corpus
    metadata = build_cv.read_clip_metadata(root)
    spk2utt, utt2spk, spk2gender, missing = build_cv.build_subset(
        build_cv.read_filelist(filelist), metadata, root
    )
    assert missing == []
    assert len(spk2utt) == 3
    assert sorted(len(v) for v in spk2utt.values()) == [2, 3, 4]
    assert sorted(spk2gender.values()) == ["f", "m", "u"]
    # Every utterance maps back to the speaker that owns it.
    for speaker, utterances in spk2utt.items():
        assert all(utt2spk[u] == speaker for u in utterances)


def test_speaker_ids_are_pseudonymised_and_stable(corpus):
    root, filelist = corpus
    metadata = build_cv.read_clip_metadata(root)
    clips = build_cv.read_filelist(filelist)
    first = build_cv.build_subset(clips, metadata, root)[0]
    second = build_cv.build_subset(clips, metadata, root)[0]
    assert set(first) == set(second), "a rebuild must produce the same ids"
    assert all(s.startswith("spk-") for s in first)
    # No Common Voice client id leaks into the output.
    assert not any("c_alice" in s or "c_bob" in s for s in first)


def test_client_ids_can_be_kept_when_asked(corpus):
    root, filelist = corpus
    metadata = build_cv.read_clip_metadata(root)
    spk2utt, _, _, _ = build_cv.build_subset(
        build_cv.read_filelist(filelist), metadata, root, keep_client_ids=True
    )
    assert set(spk2utt) == {"c_alice", "c_bob", "c_carol"}


def test_capping_limits_utterances_without_dropping_speakers(corpus):
    root, filelist = corpus
    metadata = build_cv.read_clip_metadata(root)
    spk2utt, utt2spk, _, _ = build_cv.build_subset(
        build_cv.read_filelist(filelist), metadata, root, max_utterances=2
    )
    assert len(spk2utt) == 3, "capping must not remove a speaker"
    assert all(len(v) <= 2 for v in spk2utt.values())
    assert len(utt2spk) == 6


def test_clips_absent_from_the_corpus_are_reported(corpus):
    root, filelist = corpus
    metadata = build_cv.read_clip_metadata(root)
    clips = build_cv.read_filelist(filelist) + ["common_voice_en_ghost.mp3"]
    _, _, _, missing = build_cv.build_subset(clips, metadata, root)
    assert missing == ["common_voice_en_ghost.mp3"]


def test_written_directory_is_consistent_and_decodes_mp3(corpus, tmp_path):
    root, filelist = corpus
    metadata = build_cv.read_clip_metadata(root)
    spk2utt, utt2spk, spk2gender, _ = build_cv.build_subset(
        build_cv.read_filelist(filelist), metadata, root
    )
    out = tmp_path / "kaldi"
    build_cv.write_kaldi_dir(out, spk2utt, utt2spk, spk2gender, root / "clips")

    from legal_eval.io import load_spk2utt, load_utt2spk

    written_spk2utt = load_spk2utt(out / "spk2utt")
    written_utt2spk = load_utt2spk(out / "utt2spk")
    assert written_spk2utt.keys() == spk2utt.keys()
    assert written_utt2spk == utt2spk
    # spk2utt and utt2spk must agree, or the recipe silently drops utterances.
    for speaker, utterances in written_spk2utt.items():
        assert all(written_utt2spk[u] == speaker for u in utterances)

    wav_scp = (out / "wav.scp").read_text(encoding="utf-8").splitlines()
    assert len(wav_scp) == len(utt2spk)
    assert all(line.endswith("|") for line in wav_scp), "must be a Kaldi pipe"
    assert "ffmpeg" in wav_scp[0] and "-ar 16000" in wav_scp[0]


def test_missing_tsv_is_reported_clearly(tmp_path):
    (tmp_path / "clips").mkdir()
    with pytest.raises(FileNotFoundError, match="no Common Voice TSV"):
        build_cv.read_clip_metadata(tmp_path)
