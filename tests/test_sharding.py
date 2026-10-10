"""Tests for the shard assignment rule.

These pin the properties the VoicePrivacy patch depends on. If one fails, the
patch in ``docs/gpu-utilisation.md`` is unsafe to run: it would either
synthesise an utterance twice or drop it.
"""

from __future__ import annotations

import subprocess
import sys
import zlib

import pytest

from legal_eval.sharding import balance, partition, shard_of


def utterance_ids(n: int = 5000) -> list[str]:
    """Ids shaped like LibriSpeech and Common Voice ones."""
    return [f"{100 + i // 20}-{121669 + i % 20}-{i:04d}" for i in range(n)]


def test_partition_is_disjoint_and_complete():
    """Every utterance lands in exactly one shard.

    A duplicate means two workers synthesise the same utterance and one
    overwrites the other. A miss means the corpus comes out short, which no
    later stage would notice.
    """
    ids = utterance_ids()
    for num_shards in (1, 2, 3, 4, 8, 16):
        groups = partition(ids, num_shards)
        assert len(groups) == num_shards
        flat = [u for g in groups for u in g]
        assert len(flat) == len(ids)
        assert set(flat) == set(ids)
        # Disjoint: no id appears in two groups.
        seen: set[str] = set()
        for group in groups:
            assert not (seen & set(group))
            seen |= set(group)


def test_assignment_is_stable_across_processes():
    """A separate interpreter with a different hash seed must agree.

    ``hash()`` of a str is salted per process, so using it here would give each
    worker a different split and silently break disjointness. This test is the
    reason the implementation uses crc32.
    """
    ids = utterance_ids(200)
    expected = [shard_of(u, 8) for u in ids]

    script = (
        "import zlib,sys;"
        "ids=sys.argv[1:];"
        "print(','.join(str(zlib.crc32(u.encode()) % 8) for u in ids))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script, *ids],
        capture_output=True, text=True, check=True,
        env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
    ).stdout.strip()
    assert [int(x) for x in out.split(",")] == expected


def test_assignment_does_not_depend_on_position():
    """Shuffling or shortening the list must not move anything.

    The pipeline drops already synthesised utterances before sharding, so on a
    resume the list is shorter than on a cold start. A position based split
    would reassign work and redo part of it.
    """
    ids = utterance_ids(1000)
    full = {u: shard_of(u, 5) for u in ids}

    reversed_ids = list(reversed(ids))
    assert {u: shard_of(u, 5) for u in reversed_ids} == full

    # A resume, with the first 400 already done.
    remaining = ids[400:]
    groups = partition(remaining, 5)
    for index, group in enumerate(groups):
        for u in group:
            assert full[u] == index


def test_partition_is_reasonably_balanced():
    """The run ends when the slowest shard ends, so imbalance is wasted time."""
    ids = utterance_ids(104014)  # the real train-clean-360 size
    for num_shards in (4, 8):
        groups = partition(ids, num_shards)
        assert balance(groups) < 1.05, f"{num_shards} shards too uneven"


def test_single_shard_returns_everything():
    ids = utterance_ids(50)
    assert partition(ids, 1) == [ids]
    assert all(shard_of(u, 1) == 0 for u in ids)


def test_fewer_ids_than_shards_leaves_empty_groups():
    groups = partition(["a", "b"], 8)
    assert sum(len(g) for g in groups) == 2
    assert sum(1 for g in groups if not g) >= 6


def test_rejects_nonsensical_shard_counts():
    with pytest.raises(ValueError, match="at least 1"):
        shard_of("x", 0)
    with pytest.raises(ValueError, match="at least 1"):
        shard_of("x", -3)


def test_matches_the_formula_used_in_the_vpc_patch():
    """Pin the exact expression, so the patch and this module cannot drift."""
    for u in utterance_ids(100):
        for n in (2, 4, 8):
            assert shard_of(u, n) == zlib.crc32(u.encode("utf-8")) % n


def test_balance_of_empty_partition_is_one():
    assert balance([[], [], []]) == 1.0
