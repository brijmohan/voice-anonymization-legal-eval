#!/usr/bin/env python3
"""Rebuild the paper's Common Voice subsets as Kaldi data directories.

The Interspeech 2025 results use two subsets of Mozilla Common Voice 11.0:
subset **A**, 22,024 speakers with at least 2 minutes of speech each, and subset
**B**, 4,949 speakers with at least 3 minutes each, whose speakers are a subset
of A's with disjoint utterances. The exact clip lists are published at
https://doi.org/10.5281/zenodo.14976869 as ``cv11-A-filelist`` and
``cv11-B-filelist``.

This turns those lists plus a local copy of the corpus into the ``wav.scp``,
``utt2spk``, ``spk2utt`` and ``spk2gender`` files that the VoicePrivacy recipe's
anonymization and evaluation stages consume, so the paper's evaluation sets can
be rebuilt by anyone who has the corpus.

Speaker grouping comes from Common Voice's own ``client_id``. Those are
pseudonymous but stable, so by default they are replaced with local identifiers
of the form ``spk-f-00001``, matching the convention in the published score
matrices. Pass ``--keep-client-ids`` to keep the originals, and read
``docs/data-release.md`` before publishing anything that contains them.

Clips stay as MP3 on disk and are decoded by a pipe in ``wav.scp``, which is a
Kaldi convention the recipe understands. Nothing is transcoded up front, so this
runs in minutes rather than hours and costs no extra disk.

Example::

    python scripts/build_common_voice_subsets.py \\
        --filelist cv11-A-filelist \\
        --cv-root /data/cv-corpus-11.0-2022-09-21/en \\
        --output-dir data/cv11_A \\
        --name cv11_A

    python scripts/build_common_voice_subsets.py \\
        --filelist cv11-B-filelist \\
        --cv-root /data/cv-corpus-11.0-2022-09-21/en \\
        --output-dir data/cv11_B --name cv11_B \\
        --max-utterances-per-speaker 30
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from build_cv_common import (  # noqa: E402
    read_clip_metadata,
    write_kaldi_dir,
)


def read_filelist(path: Path) -> list[str]:
    """Read a published file list and return clip basenames in order.

    The published lists hold absolute paths from the original experiment
    machine, so only the basename is meaningful here.
    """
    clips = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            name = line.strip().rsplit("/", 1)[-1]
            if name:
                clips.append(name)
    return clips



def build_subset(
    clips: list[str],
    metadata: dict[str, tuple[str, str]],
    cv_root: Path,
    max_utterances: int | None = None,
    keep_client_ids: bool = False,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str], list[str]]:
    """Group clips by speaker and assign identifiers.

    Args:
        clips: Clip basenames from a published list.
        metadata: Output of :func:`read_clip_metadata`.
        cv_root: Locale directory, used to check that clips exist.
        max_utterances: Keep at most this many utterances per speaker, in list
            order. Subset B holds every utterance of each speaker, which is far
            more than any conversation length needs, so capping it removes most
            of the audio without affecting a metric computed at ``L <= 30``.
        keep_client_ids: Use Common Voice ``client_id`` values as speaker ids
            instead of local pseudonyms.

    Returns:
        ``(spk2utt, utt2spk, spk2gender, missing)`` where ``missing`` lists
        clips absent from the corpus or its metadata.
    """
    clips_dir = cv_root / "clips"
    by_speaker: dict[str, list[str]] = defaultdict(list)
    gender_of: dict[str, str] = {}
    missing: list[str] = []

    for clip in clips:
        entry = metadata.get(clip)
        if entry is None or not (clips_dir / clip).exists():
            missing.append(clip)
            continue
        client_id, gender = entry
        by_speaker[client_id].append(clip)
        gender_of.setdefault(client_id, gender)

    # Stable pseudonyms: sorted by client id so a rebuild is reproducible, and
    # numbered within gender to match the published score matrices' convention.
    counters: dict[str, int] = defaultdict(int)
    renamed: dict[str, str] = {}
    for client_id in sorted(by_speaker):
        if keep_client_ids:
            renamed[client_id] = client_id
            continue
        gender = gender_of[client_id]
        counters[gender] += 1
        renamed[client_id] = f"spk-{gender}-{counters[gender]:05d}"

    spk2utt: dict[str, list[str]] = {}
    utt2spk: dict[str, str] = {}
    spk2gender: dict[str, str] = {}
    for client_id, speaker_clips in by_speaker.items():
        speaker = renamed[client_id]
        kept = speaker_clips[:max_utterances] if max_utterances else speaker_clips
        utterances = [Path(c).stem for c in kept]
        spk2utt[speaker] = utterances
        spk2gender[speaker] = gender_of[client_id]
        for utt in utterances:
            utt2spk[utt] = speaker
    return spk2utt, utt2spk, spk2gender, missing



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filelist", type=Path, required=True,
                        help="cv11-A-filelist or cv11-B-filelist from the Zenodo record")
    parser.add_argument("--cv-root", type=Path, required=True,
                        help="corpus locale directory, holding clips/ and validated.tsv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default=None, help="label for the summary line")
    parser.add_argument("--max-utterances-per-speaker", type=int, default=None)
    parser.add_argument("--keep-client-ids", action="store_true",
                        help="do not pseudonymise speaker ids; see docs/data-release.md")
    parser.add_argument("--allow-missing", type=float, default=1.0,
                        help="fail if more than this fraction of clips is absent")
    args = parser.parse_args()

    clips = read_filelist(args.filelist)
    print(f"{len(clips):,} clips listed in {args.filelist.name}")

    metadata = read_clip_metadata(args.cv_root)
    print(f"{len(metadata):,} clips described by the corpus metadata")

    spk2utt, utt2spk, spk2gender, missing = build_subset(
        clips, metadata, args.cv_root,
        max_utterances=args.max_utterances_per_speaker,
        keep_client_ids=args.keep_client_ids,
    )

    fraction = len(missing) / len(clips) if clips else 0.0
    if missing:
        print(f"WARNING: {len(missing):,} clips ({fraction:.1%}) absent from the corpus")
        print(f"  first few: {', '.join(missing[:3])}")
    if fraction > args.allow_missing:
        print(f"FAIL: more than {args.allow_missing:.0%} of clips are missing. "
              "Is this the right Common Voice release? Clip ids differ between them.",
              file=sys.stderr)
        return 1

    write_kaldi_dir(args.output_dir, spk2utt, utt2spk, spk2gender, args.cv_root / "clips")

    by_gender: dict[str, int] = defaultdict(int)
    for gender in spk2gender.values():
        by_gender[gender] += 1
    print(
        f"\n{args.name or args.output_dir.name}: {len(spk2utt):,} speakers "
        f"({by_gender['f']:,} f, {by_gender['m']:,} m, {by_gender['u']:,} unknown), "
        f"{len(utt2spk):,} utterances"
    )
    if args.max_utterances_per_speaker:
        print(f"  capped at {args.max_utterances_per_speaker} utterances per speaker")
    print(f"  written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
