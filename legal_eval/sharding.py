"""Splitting a corpus across concurrent workers, deterministically.

Anonymizing a corpus is embarrassingly parallel: each utterance is independent
and its output is a separate file. The VoicePrivacy pipeline does not exploit
that. Its parallelism is ``len(self.devices)``, so a single GPU gives a single
worker, and the synthesis stage, which dominates, has no parallelism at all.
Measured on an A100 that leaves the GPU at 0% utilisation and 8.7% of memory
while one CPU core does the work.

This module holds the assignment rule used to shard that work, so the rule has
one definition with tests rather than living only inside a patch to someone
else's repository. See ``docs/gpu-utilisation.md`` for the patch itself and the
measurements.

The rule has to satisfy four things, each pinned by a test:

* **Disjoint.** No utterance is assigned to two workers, or it is synthesised
  twice and one worker's output overwrites the other's.
* **Complete.** Every utterance is assigned to some worker, or the run silently
  produces a short corpus.
* **Stable across processes.** Workers are separate processes. Python randomises
  ``hash()`` of a string per process unless ``PYTHONHASHSEED`` is fixed, so
  ``hash()`` would give each worker a different split, which breaks both
  properties above. ``zlib.crc32`` is stable everywhere.
* **Stable across restarts.** The assignment must depend on the utterance id
  alone, never on its position in a list. The pipeline removes already
  synthesised utterances before sharding, so the list differs between a cold
  start and a resume, and a position based split would reassign work.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Sequence


def shard_of(utterance_id: str, num_shards: int) -> int:
    """The shard an utterance belongs to.

    Args:
        utterance_id: Utterance identifier.
        num_shards: Number of workers, at least 1.

    Returns:
        A shard index in ``[0, num_shards)``.

    Raises:
        ValueError: If ``num_shards`` is less than 1.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be at least 1, got {num_shards}")
    return zlib.crc32(utterance_id.encode("utf-8")) % num_shards


def partition(
    utterance_ids: Iterable[str], num_shards: int
) -> list[list[str]]:
    """Split utterance ids into ``num_shards`` disjoint, complete groups.

    Order within each group follows the input, so a caller that wants
    reproducible output ordering should pass a sorted sequence.

    Args:
        utterance_ids: The ids to split.
        num_shards: Number of workers.

    Returns:
        A list of ``num_shards`` lists. Some may be empty when there are fewer
        ids than shards.
    """
    groups: list[list[str]] = [[] for _ in range(num_shards)]
    for utterance_id in utterance_ids:
        groups[shard_of(utterance_id, num_shards)].append(utterance_id)
    return groups


def balance(groups: Sequence[Sequence[str]]) -> float:
    """How uneven a partition is, as the max group size over the mean.

    1.0 is perfectly even. A hash based split is not exactly even, and the
    imbalance is what sets the tail: the run finishes when the largest shard
    finishes, so a balance of 1.1 wastes roughly 10% of the wall clock.

    Returns:
        ``max / mean``, or 1.0 when every group is empty.
    """
    sizes = [len(g) for g in groups]
    total = sum(sizes)
    if total == 0:
        return 1.0
    return max(sizes) / (total / len(sizes))
