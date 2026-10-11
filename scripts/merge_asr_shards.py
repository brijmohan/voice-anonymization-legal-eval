#!/usr/bin/env python3
"""Merge per-shard ASR transcripts into the canonical files the pipeline reads.

The VoicePrivacy ASR stage writes one aggregate ``text`` and ``utt2spk`` per
dataset. Concurrent workers writing those paths would race and lose
transcripts, so the sharding patch has each worker write ``text_<K>`` and
``utt2spk_<K>`` instead, using the per-job ``add_suffix`` that
``recognition_job`` already implements. This combines them.

Run after every shard has exited and before the next pipeline stage. The next
stage loads ``text``, finds every utterance present, and skips ASR entirely.

It refuses rather than guesses:

* a shard file missing its pair is an error, not something to work around;
* an utterance appearing in two shards means the shard rule was not applied
  consistently, which would silently halve or duplicate a corpus;
* a mismatch between ``text`` and ``utt2spk`` keys means one write was
  truncated.

Usage::

    python scripts/merge_asr_shards.py \\
        exp/anon_pipeline_sttts/transcription/asr_branchformer_tts-phn_en/cv27_B_test
    python scripts/merge_asr_shards.py <dir> --expect 195580
    python scripts/merge_asr_shards.py <dir> --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Upstream's own per-job convention, from recognition_job:
#     add_suffix = f'_{job_id}' if job_id is not None else None
# The sharding patch passes the shard index as job_id, so the shards land
# as text_0, utt2spk_0 and so on. Inventing a different suffix would mean
# the incremental saves inside recognition_job used one name and the final
# save another.
SHARD = re.compile(r"^(text|utt2spk)_(\d+)$")


def read_kaldi(path: Path) -> dict[str, str]:
    """Read a Kaldi two-column file as a mapping, first field is the key."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            key, _, value = line.partition(" ")
            out[key] = value
    return out


def write_kaldi(path: Path, rows: dict[str, str]) -> None:
    """Write sorted, so the canonical files are byte-stable across merges."""
    with open(path, "w", encoding="utf-8") as handle:
        for key in sorted(rows):
            handle.write(f"{key} {rows[key]}\n")


def find_shards(directory: Path) -> list[int]:
    found = set()
    for entry in directory.iterdir():
        match = SHARD.match(entry.name)
        if match:
            found.add(int(match.group(2)))
    return sorted(found)


def merge(directory: Path, expect: int | None, check: bool) -> int:
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2

    shards = find_shards(directory)
    if not shards:
        print(f"no text_<K> files in {directory}", file=sys.stderr)
        return 2
    print(f"{directory}\n  shards found: {shards}")

    texts: dict[str, str] = {}
    speakers: dict[str, str] = {}
    owner: dict[str, int] = {}
    duplicates: list[tuple[str, int, int]] = []

    # Fold in any existing canonical file first. The pipeline's resume reads
    # only 'text', so a shard skips whatever is already in it and its text_<K>
    # covers the remainder. Discarding 'text' here would lose exactly the
    # utterances an earlier unsharded run completed. Shard entries are applied
    # afterwards and win, being the fresher write.
    existing = directory / "text"
    existing_spk = directory / "utt2spk"
    if existing.exists() and existing_spk.exists():
        texts.update(read_kaldi(existing))
        speakers.update(read_kaldi(existing_spk))
        print(f"  existing text: {len(texts)} transcripts folded in")

    for k in shards:
        text_path = directory / f"text_{k}"
        spk_path = directory / f"utt2spk_{k}"
        for path in (text_path, spk_path):
            if not path.exists():
                print(f"  shard {k} is incomplete, {path.name} is missing",
                      file=sys.stderr)
                return 1
        t, s = read_kaldi(text_path), read_kaldi(spk_path)
        if set(t) != set(s):
            print(f"  shard {k}: text has {len(t)} keys, utt2spk has {len(s)}, "
                  f"and they differ. One write was truncated.", file=sys.stderr)
            return 1
        for utt in t:
            if utt in owner:
                duplicates.append((utt, owner[utt], k))
            owner[utt] = k
        texts.update(t)
        speakers.update(s)
        print(f"  shard {k}: {len(t)} transcripts")

    if duplicates:
        print(f"  {len(duplicates)} utterance(s) appear in more than one shard, "
              f"so the shard rule was not applied consistently. "
              f"First: {duplicates[0]}", file=sys.stderr)
        return 1

    total = len(texts)
    print(f"  merged: {total} unique transcripts")

    if expect is not None and total != expect:
        print(f"  expected {expect}, got {total}, short by {expect - total}. "
              f"A shard probably did not finish.", file=sys.stderr)
        return 1

    if check:
        print("  check only, nothing written")
        return 0

    write_kaldi(directory / "text", texts)
    write_kaldi(directory / "utt2spk", speakers)
    print(f"  wrote {directory / 'text'} and {directory / 'utt2spk'}")

    back = read_kaldi(directory / "text")
    if back != texts:
        print("  read-back mismatch after writing", file=sys.stderr)
        return 1
    print("  read back and verified")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path,
                        help="the transcription directory for one dataset")
    parser.add_argument("--expect", type=int, default=None,
                        help="utterance count the dataset should have")
    parser.add_argument("--check", action="store_true",
                        help="validate and report, write nothing")
    args = parser.parse_args()
    return merge(args.directory, args.expect, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
