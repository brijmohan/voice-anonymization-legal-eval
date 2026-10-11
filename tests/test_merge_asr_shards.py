"""Tests for the ASR shard merge.

This script stands between a sharded ASR run and every later stage, so a silent
error here produces a short corpus that nothing downstream would notice. The
tests are mostly about refusing bad input rather than merging good input.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "merge_asr_shards",
    Path(__file__).resolve().parents[1] / "scripts" / "merge_asr_shards.py",
)
merge_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge_mod)


def write_shard(directory: Path, k: int, rows: dict[str, str], speakers=None):
    speakers = speakers or {u: u.split("-")[0] for u in rows}
    (directory / f"text_{k}").write_text(
        "".join(f"{u} {rows[u]}\n" for u in sorted(rows)), encoding="utf-8"
    )
    (directory / f"utt2spk_{k}").write_text(
        "".join(f"{u} {speakers[u]}\n" for u in sorted(speakers)), encoding="utf-8"
    )


def test_merges_disjoint_shards(tmp_path, capsys):
    write_shard(tmp_path, 0, {"s1-a": "hello there", "s1-b": "second"})
    write_shard(tmp_path, 1, {"s2-a": "third one"})
    assert merge_mod.merge(tmp_path, expect=3, check=False) == 0
    text = merge_mod.read_kaldi(tmp_path / "text")
    assert text == {"s1-a": "hello there", "s1-b": "second", "s2-a": "third one"}
    assert set(merge_mod.read_kaldi(tmp_path / "utt2spk")) == set(text)
    assert "read back and verified" in capsys.readouterr().out


def test_output_is_sorted_and_stable(tmp_path):
    write_shard(tmp_path, 0, {"z": "last", "a": "first"})
    write_shard(tmp_path, 1, {"m": "middle"})
    merge_mod.merge(tmp_path, None, False)
    lines = (tmp_path / "text").read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in lines] == ["a", "m", "z"]


def test_refuses_an_utterance_in_two_shards(tmp_path, capsys):
    """The shard rule must partition. Overlap means it was applied wrongly."""
    write_shard(tmp_path, 0, {"dup": "one", "x": "a"})
    write_shard(tmp_path, 1, {"dup": "two", "y": "b"})
    assert merge_mod.merge(tmp_path, None, False) == 1
    assert "more than one shard" in capsys.readouterr().err
    assert not (tmp_path / "text").exists()


def test_refuses_a_shard_missing_its_pair(tmp_path, capsys):
    write_shard(tmp_path, 0, {"a": "x"})
    (tmp_path / "utt2spk_0").unlink()
    assert merge_mod.merge(tmp_path, None, False) == 1
    assert "incomplete" in capsys.readouterr().err


def test_refuses_mismatched_keys_within_a_shard(tmp_path, capsys):
    """A truncated write shows up as text and utt2spk disagreeing."""
    (tmp_path / "text_0").write_text("a x\nb y\n", encoding="utf-8")
    (tmp_path / "utt2spk_0").write_text("a s1\n", encoding="utf-8")
    assert merge_mod.merge(tmp_path, None, False) == 1
    assert "truncated" in capsys.readouterr().err


def test_refuses_a_short_merge_when_a_count_is_given(tmp_path, capsys):
    """The guard against a shard that died without anyone noticing."""
    write_shard(tmp_path, 0, {"a": "x"})
    write_shard(tmp_path, 1, {"b": "y"})
    assert merge_mod.merge(tmp_path, expect=10, check=False) == 1
    err = capsys.readouterr().err
    assert "short by 8" in err
    assert not (tmp_path / "text").exists()


def test_check_mode_writes_nothing(tmp_path):
    write_shard(tmp_path, 0, {"a": "x"})
    assert merge_mod.merge(tmp_path, None, check=True) == 0
    assert not (tmp_path / "text").exists()


def test_no_shards_is_an_error(tmp_path, capsys):
    assert merge_mod.merge(tmp_path, None, False) == 2
    assert "no text_<K>" in capsys.readouterr().err


def test_missing_directory_is_an_error(tmp_path, capsys):
    assert merge_mod.merge(tmp_path / "nope", None, False) == 2
    assert "not a directory" in capsys.readouterr().err


def test_transcripts_with_spaces_survive_a_round_trip(tmp_path):
    """Transcripts contain spaces, so only the first field is the key."""
    write_shard(tmp_path, 0, {"a": "a sentence with several words"})
    merge_mod.merge(tmp_path, None, False)
    assert merge_mod.read_kaldi(tmp_path / "text")["a"] == (
        "a sentence with several words"
    )


def test_find_shards_ignores_unrelated_files(tmp_path):
    write_shard(tmp_path, 0, {"a": "x"})
    write_shard(tmp_path, 7, {"b": "y"})
    (tmp_path / "text").write_text("stale old\n", encoding="utf-8")
    (tmp_path / "text_X").write_text("junk\n", encoding="utf-8")
    assert merge_mod.find_shards(tmp_path) == [0, 7]


def test_an_existing_canonical_file_is_folded_in(tmp_path):
    """Transcripts from an earlier unsharded run must survive the merge.

    The pipeline's resume reads only the canonical 'text', so a shard skips
    anything already in it and its text_<K> covers the remainder. Discarding
    'text' at merge time would therefore lose exactly those utterances. This
    reverses an earlier version of this test, which asserted the opposite and
    would have silently truncated the corpus.
    """
    write_shard(tmp_path, 0, {"b": "from shard"})
    (tmp_path / "text").write_text("a from earlier run\n", encoding="utf-8")
    (tmp_path / "utt2spk").write_text("a s1\n", encoding="utf-8")
    assert merge_mod.merge(tmp_path, expect=2, check=False) == 0
    merged = merge_mod.read_kaldi(tmp_path / "text")
    assert merged == {"a": "from earlier run", "b": "from shard"}


def test_a_shard_wins_over_a_stale_canonical_entry(tmp_path):
    """If both have an utterance, the shard is the fresher write."""
    write_shard(tmp_path, 0, {"a": "new"})
    (tmp_path / "text").write_text("a old\n", encoding="utf-8")
    (tmp_path / "utt2spk").write_text("a s1\n", encoding="utf-8")
    merge_mod.merge(tmp_path, None, False)
    assert merge_mod.read_kaldi(tmp_path / "text")["a"] == "new"
