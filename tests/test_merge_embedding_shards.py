"""Tests for the embedding shard merge.

This script stands between a sharded extraction and every metric computed
afterwards. A short or double-counted merge changes the population, and every
metric's chance level is a function of the population, so the damage does not
look like an error. It looks like a result.

So the tests are mostly about refusing bad input, and the one test that merges
good input checks that a merged set is indistinguishable from the same
utterances extracted in one pass.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SPEC = importlib.util.spec_from_file_location(
    "merge_embedding_shards",
    Path(__file__).resolve().parents[1] / "scripts" / "merge_embedding_shards.py",
)
merge_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge_mod)


def write_shard(
    directory: Path,
    rows: dict[str, tuple[int, str]],
    dim: int = 4,
    gender: dict[str, str] | None = None,
    n_rows: int | None = None,
):
    """Write one shard directory.

    Args:
        directory: where to write.
        rows: identifier to (row index, speaker). Row indices are deliberately
            not in identifier order, because the extractor writes in batches
            and the real files are not ordered either.
        dim: embedding dimension.
        gender: speaker to gender, defaulting to the speaker name's own marker.
        n_rows: tensor row count, defaulting to len(rows).
    """
    directory.mkdir(parents=True, exist_ok=True)
    count = n_rows if n_rows is not None else len(rows)
    # Row i is the constant vector i, so a row can be traced after merging.
    tensor = torch.stack([torch.full((dim,), float(i)) for i in range(count)])
    torch.save(tensor, directory / "speaker_vectors.pt")
    (directory / "id2idx").write_text(
        "".join(f"{utt} {index}\n" for utt, (index, _) in rows.items()),
        encoding="utf-8",
    )
    (directory / "idx2spk").write_text(
        "".join(f"{index} {spk}\n" for _, (index, spk) in rows.items()),
        encoding="utf-8",
    )
    speakers = {spk for _, spk in rows.values()}
    gender = gender or {s: ("f" if "-f-" in s else "m") for s in speakers}
    (directory / "spk2gender").write_text(
        "".join(f"{s} {gender[s]}\n" for s in sorted(speakers)), encoding="utf-8"
    )
    return directory


def test_merges_two_shards_and_keeps_every_row_with_its_identifier(tmp_path):
    """The point of the merge: an identifier still names its own embedding."""
    a = write_shard(tmp_path / "a", {"u3": (1, "spk-f-1"), "u1": (0, "spk-f-1")})
    b = write_shard(tmp_path / "b", {"u2": (1, "spk-m-2"), "u4": (0, "spk-m-2")})
    out = tmp_path / "merged"

    counts = merge_mod.merge([a, b], out, expect=4)
    assert counts == {"identifiers": 4, "speakers": 2, "dim": 4, "shards": 2}

    vectors = torch.load(out / "speaker_vectors.pt", map_location="cpu",
                         weights_only=True)
    assert tuple(vectors.shape) == (4, 4)

    id2idx = {k: int(v) for k, v in merge_mod.read_mapping(out / "id2idx").items()}
    idx2spk = {int(k): v for k, v in
               merge_mod.read_mapping(out / "idx2spk").items()}

    # Shard a's row 0 was u1 and row 1 was u3; shard b's rows are offset by 2.
    assert float(vectors[id2idx["u1"]][0]) == 0.0
    assert float(vectors[id2idx["u3"]][0]) == 1.0
    assert float(vectors[id2idx["u4"]][0]) == 0.0
    assert float(vectors[id2idx["u2"]][0]) == 1.0

    assert idx2spk[id2idx["u1"]] == "spk-f-1"
    assert idx2spk[id2idx["u2"]] == "spk-m-2"

    assert merge_mod.read_mapping(out / "spk2gender") == {
        "spk-f-1": "f", "spk-m-2": "m"
    }


def test_a_speaker_split_across_shards_is_merged_not_duplicated(tmp_path):
    """crc32 shards by utterance, so one speaker's utterances do split."""
    a = write_shard(tmp_path / "a", {"u1": (0, "spk-f-1")})
    b = write_shard(tmp_path / "b", {"u2": (0, "spk-f-1")})
    counts = merge_mod.merge([a, b], tmp_path / "out", expect=2)
    assert counts["identifiers"] == 2
    assert counts["speakers"] == 1


def test_refuses_an_identifier_in_two_shards(tmp_path):
    """Shards must partition the corpus; an overlap double-counts an utterance."""
    a = write_shard(tmp_path / "a", {"u1": (0, "spk-f-1")})
    b = write_shard(tmp_path / "b", {"u1": (0, "spk-f-1")})
    with pytest.raises(ValueError, match="more than one shard"):
        merge_mod.merge([a, b], tmp_path / "out")


def test_refuses_a_short_merge_against_an_expected_count(tmp_path):
    """The failure this exists for: fewer identifiers reads as a result."""
    a = write_shard(tmp_path / "a", {"u1": (0, "spk-f-1")})
    b = write_shard(tmp_path / "b", {"u2": (0, "spk-m-2")})
    with pytest.raises(ValueError, match="expected 60000"):
        merge_mod.merge([a, b], tmp_path / "out", expect=60000)
    assert not (tmp_path / "out" / "speaker_vectors.pt").exists()


def test_refuses_an_incomplete_shard_and_names_every_missing_file(tmp_path):
    a = write_shard(tmp_path / "a", {"u1": (0, "spk-f-1")})
    (a / "idx2spk").unlink()
    (a / "spk2gender").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        merge_mod.merge([a], tmp_path / "out")
    assert "idx2spk" in str(exc.value)
    assert "spk2gender" in str(exc.value)


def test_refuses_a_disagreeing_embedding_dimension(tmp_path):
    """Two dimensions means two extractors, which must never be merged."""
    a = write_shard(tmp_path / "a", {"u1": (0, "spk-f-1")}, dim=192)
    b = write_shard(tmp_path / "b", {"u2": (0, "spk-m-2")}, dim=256)
    with pytest.raises(ValueError, match="not the same"):
        merge_mod.merge([a, b], tmp_path / "out")


def test_refuses_a_speaker_with_two_genders(tmp_path):
    a = write_shard(tmp_path / "a", {"u1": (0, "spk-x-1")}, gender={"spk-x-1": "f"})
    b = write_shard(tmp_path / "b", {"u2": (0, "spk-x-1")}, gender={"spk-x-1": "m"})
    with pytest.raises(ValueError, match="in one shard"):
        merge_mod.merge([a, b], tmp_path / "out")


def test_refuses_when_rows_and_identifiers_disagree(tmp_path):
    """A half-written shard has a full tensor and a short mapping."""
    a = write_shard(tmp_path / "a", {"u1": (0, "spk-f-1")}, n_rows=5)
    with pytest.raises(ValueError, match="rows but 1 identifiers"):
        merge_mod.merge([a], tmp_path / "out")


def test_refuses_an_index_outside_the_tensor(tmp_path):
    a = write_shard(tmp_path / "a", {"u1": (7, "spk-f-1")}, n_rows=1)
    with pytest.raises(ValueError, match="outside the tensor"):
        merge_mod.merge([a], tmp_path / "out")


def test_refuses_to_overwrite_an_existing_merge(tmp_path):
    """A rerun must not leave rows from two different merges."""
    a = write_shard(tmp_path / "a", {"u1": (0, "spk-f-1")})
    out = tmp_path / "out"
    merge_mod.merge([a], out, expect=1)
    with pytest.raises(FileExistsError, match="already holds"):
        merge_mod.merge([a], out, expect=1)


def test_refuses_no_shards(tmp_path):
    with pytest.raises(ValueError, match="no shard directories"):
        merge_mod.merge([], tmp_path / "out")


def test_read_mapping_refuses_a_repeated_key(tmp_path):
    path = tmp_path / "m"
    path.write_text("a 1\na 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repeated key"):
        merge_mod.read_mapping(path)


def test_read_mapping_refuses_a_wrong_field_count(tmp_path):
    path = tmp_path / "m"
    path.write_text("a 1 extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected two fields"):
        merge_mod.read_mapping(path)


def test_read_mapping_ignores_blank_lines(tmp_path):
    path = tmp_path / "m"
    path.write_text("a 1\n\nb 2\n", encoding="utf-8")
    assert merge_mod.read_mapping(path) == {"a": "1", "b": "2"}


def test_main_reports_a_refusal_and_exits_non_zero(tmp_path, capsys):
    a = write_shard(tmp_path / "a", {"u1": (0, "spk-f-1")})
    code = merge_mod.main([
        "--shard-dir", str(a), "--output", str(tmp_path / "out"), "--expect", "99",
    ])
    assert code == 1
    assert "refused:" in capsys.readouterr().err


def test_main_merges_and_exits_zero(tmp_path, capsys):
    a = write_shard(tmp_path / "a", {"u1": (0, "spk-f-1")})
    b = write_shard(tmp_path / "b", {"u2": (0, "spk-m-2")})
    code = merge_mod.main([
        "--shard-dir", str(a), "--shard-dir", str(b),
        "--output", str(tmp_path / "out"), "--expect", "2",
    ])
    assert code == 0
    assert "2 identifiers" in capsys.readouterr().out


def test_merge_is_independent_of_shard_order_up_to_row_numbering(tmp_path):
    """Shard order sets row indices, but must not change any embedding."""
    rows_a = {"u1": (0, "spk-f-1"), "u3": (1, "spk-f-1")}
    rows_b = {"u2": (0, "spk-m-2")}
    forward = merge_mod.merge(
        [write_shard(tmp_path / "a", rows_a), write_shard(tmp_path / "b", rows_b)],
        tmp_path / "fwd", expect=3,
    )
    reverse = merge_mod.merge(
        [write_shard(tmp_path / "b2", rows_b), write_shard(tmp_path / "a2", rows_a)],
        tmp_path / "rev", expect=3,
    )
    assert forward == reverse

    def vectors_by_identifier(out: Path):
        vectors = torch.load(out / "speaker_vectors.pt", map_location="cpu",
                             weights_only=True)
        id2idx = {k: int(v) for k, v in
                  merge_mod.read_mapping(out / "id2idx").items()}
        return {k: vectors[i].tolist() for k, i in id2idx.items()}

    assert vectors_by_identifier(tmp_path / "fwd") == vectors_by_identifier(
        tmp_path / "rev"
    )
