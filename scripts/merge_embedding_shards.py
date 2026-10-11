#!/usr/bin/env python3
"""Merge sharded ASV embedding directories into one.

The challenge's own extractor parallelises with
``tqdm.contrib.concurrent.process_map``, which pickles each worker's result
back to the parent. For the ``ecapa_ssl`` attacker that fails immediately:

    RuntimeError: Cowardly refusing to serialize non-leaf tensor which
    requires_grad, since autograd does not support crossing process boundaries.

Rather than repair someone else's multiprocessing, extract each shard as its
own single-process job, which is the path that works, and merge the results
here. Sharding uses the project's own rule, :func:`legal_eval.sharding.shard_of`,
so a shard's membership is stable across restarts and independent of position
in the file.

An embedding directory holds four files:

- ``speaker_vectors.pt``, an ``(n, dim)`` float tensor, one row per identifier
- ``id2idx``, ``<identifier> <row>``
- ``idx2spk``, ``<row> <speaker>``
- ``spk2gender``, ``<speaker> <f|m>``

Row order is not the order of the input list, because the extractor writes in
batches, so merging offsets each shard's own indices rather than rebuilding
them from a sorted identifier list.

This script refuses rather than guesses. A silent partial merge here produces a
smaller population, and the population is the one quantity the whole comparison
is about: every metric's chance level is a function of it, so a short merge
does not look like an error, it looks like a result.

Usage:
    python scripts/merge_embedding_shards.py \\
        --shard-dir exp/.../emb_xvect/cv27_enrolls_shard0/utt-level \\
        --shard-dir exp/.../emb_xvect/cv27_enrolls_shard1/utt-level \\
        --output exp/.../emb_xvect/cv27_enrolls/utt-level \\
        --expect 60000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED = ("speaker_vectors.pt", "id2idx", "idx2spk", "spk2gender")


def _load_torch():
    """Import torch, or explain what to install.

    Returns:
        The ``torch`` module.

    Raises:
        SystemExit: when torch is not installed.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised by the error path only
        raise SystemExit(
            "merging embedding shards needs torch, which the core package does "
            "not require. Install it with: pip install torch"
        ) from None
    return torch


def read_mapping(path: Path) -> dict[str, str]:
    """Read a two-column whitespace-separated Kaldi-style file.

    Args:
        path: file to read.

    Returns:
        First column to second column.

    Raises:
        ValueError: on a line that does not have exactly two fields, or on a
            repeated key. Both mean the file is not what we think it is.
    """
    out: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{number}: expected two fields, got {len(parts)}")
        key, value = parts
        if key in out:
            raise ValueError(f"{path}:{number}: repeated key {key!r}")
        out[key] = value
    return out


def check_shard(directory: Path) -> None:
    """Verify a shard directory holds all four files.

    Args:
        directory: the shard's ``utt-level`` directory.

    Raises:
        FileNotFoundError: naming every file that is missing, so a half-written
            shard is reported once rather than one file at a time.
    """
    missing = [name for name in REQUIRED if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{directory} is not a complete embedding directory, missing: "
            f"{', '.join(missing)}"
        )


def merge(
    shard_dirs: list[Path],
    output: Path,
    expect: int | None = None,
) -> dict[str, int]:
    """Merge shard directories into one embedding directory.

    Args:
        shard_dirs: the shards' ``utt-level`` directories, in any order. Rows
            are concatenated in the order given, which affects row indices but
            no value and no mapping.
        output: directory to write. Created if absent. Refused if it already
            holds an embedding set, so a rerun cannot half-overwrite one.
        expect: the total identifier count the merge must reach, when known.

    Returns:
        Counts: ``identifiers``, ``speakers``, ``dim``, ``shards``.

    Raises:
        ValueError: on no shards, a repeated identifier across shards, a
            disagreeing embedding dimension, a speaker with two genders, a row
            count that does not match the mapping, or a total below ``expect``.
        FileNotFoundError: on an incomplete shard.
        FileExistsError: when the output already holds an embedding set.
    """
    torch = _load_torch()

    if not shard_dirs:
        raise ValueError("no shard directories given")
    for directory in shard_dirs:
        check_shard(directory)

    if any((output / name).exists() for name in REQUIRED):
        raise FileExistsError(
            f"{output} already holds an embedding set. Remove it first: a "
            f"partial overwrite would leave rows from two different merges."
        )

    tensors = []
    id2idx: dict[str, int] = {}
    idx2spk: dict[int, str] = {}
    spk2gender: dict[str, str] = {}
    offset = 0
    dim: int | None = None

    for directory in shard_dirs:
        vectors = torch.load(
            directory / "speaker_vectors.pt", map_location="cpu", weights_only=True
        )
        if vectors.ndim != 2:
            raise ValueError(
                f"{directory}: expected a 2-D tensor, got shape {tuple(vectors.shape)}"
            )
        if dim is None:
            dim = int(vectors.shape[1])
        elif int(vectors.shape[1]) != dim:
            raise ValueError(
                f"{directory}: embedding dimension {vectors.shape[1]} does not "
                f"match {dim} from an earlier shard. These are not the same "
                f"extractor and must not be merged."
            )

        shard_id2idx = {k: int(v) for k, v in read_mapping(directory / "id2idx").items()}
        shard_idx2spk = {
            int(k): v for k, v in read_mapping(directory / "idx2spk").items()
        }
        shard_gender = read_mapping(directory / "spk2gender")

        rows = int(vectors.shape[0])
        if len(shard_id2idx) != rows:
            raise ValueError(
                f"{directory}: {rows} rows but {len(shard_id2idx)} identifiers in "
                f"id2idx"
            )
        out_of_range = [i for i in shard_id2idx.values() if not 0 <= i < rows]
        if out_of_range:
            raise ValueError(
                f"{directory}: id2idx points outside the tensor, e.g. row "
                f"{out_of_range[0]} of {rows}"
            )

        for identifier, index in shard_id2idx.items():
            if identifier in id2idx:
                raise ValueError(
                    f"{identifier!r} appears in more than one shard. Shards must "
                    f"partition the corpus, so this is a sharding error and the "
                    f"merge would double-count a speaker's utterance."
                )
            id2idx[identifier] = index + offset

        for index, speaker in shard_idx2spk.items():
            idx2spk[index + offset] = speaker

        for speaker, gender in shard_gender.items():
            if speaker in spk2gender and spk2gender[speaker] != gender:
                raise ValueError(
                    f"speaker {speaker!r} is {spk2gender[speaker]!r} in one shard "
                    f"and {gender!r} in another"
                )
            spk2gender[speaker] = gender

        tensors.append(vectors)
        offset += rows

    if expect is not None and len(id2idx) != expect:
        raise ValueError(
            f"merged {len(id2idx)} identifiers but expected {expect}. A short "
            f"merge changes the population every metric is computed over, so it "
            f"is refused rather than written."
        )

    merged = torch.cat(tensors, dim=0)
    if int(merged.shape[0]) != len(id2idx):
        raise ValueError(
            f"merged tensor has {merged.shape[0]} rows but {len(id2idx)} "
            f"identifiers"
        )

    output.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output / "speaker_vectors.pt")
    (output / "id2idx").write_text(
        "".join(f"{k} {v}\n" for k, v in sorted(id2idx.items())), encoding="utf-8"
    )
    (output / "idx2spk").write_text(
        "".join(f"{k} {idx2spk[k]}\n" for k in sorted(idx2spk)), encoding="utf-8"
    )
    (output / "spk2gender").write_text(
        "".join(f"{k} {v}\n" for k, v in sorted(spk2gender.items())), encoding="utf-8"
    )

    return {
        "identifiers": len(id2idx),
        "speakers": len(set(idx2spk.values())),
        "dim": dim or 0,
        "shards": len(shard_dirs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard-dir", type=Path, action="append", default=[],
                        dest="shard_dirs", required=True,
                        help="a shard's utt-level directory, repeatable")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory to write the merged set to")
    parser.add_argument("--expect", type=int, default=None,
                        help="total identifiers the merge must reach")
    args = parser.parse_args(argv)

    try:
        counts = merge(args.shard_dirs, args.output, args.expect)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    print(
        f"merged {counts['shards']} shard(s): {counts['identifiers']} identifiers, "
        f"{counts['speakers']} speakers, dim {counts['dim']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
