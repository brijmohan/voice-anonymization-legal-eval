#!/usr/bin/env python3
"""Build paper-style enrollment and test subsets from any Common Voice release.

The Interspeech 2025 results use two subsets of Common Voice 11.0, whose exact
clip lists are published on Zenodo. Mozilla has since restricted access to old
releases in order to honour deletion requests, so reproducing those exact lists
is no longer straightforward, and arguably should not be: a current release has
already removed speakers who withdrew consent.

This builds subsets with the *same construction* from whatever release you
have. The absolute numbers will not match the paper, but the comparison the
metrics exist for, between anonymization systems and between attacker models,
is preserved, because every system is measured on identical data.

The construction follows the paper:

* **Enrollment (A)**: speakers with at least ``--min-enroll-minutes`` of speech.
* **Test (B)**: a subset of A's speakers, with at least ``--min-test-minutes``
  of *further* speech, using utterances disjoint from their enrollment ones.

A speaker therefore needs roughly the sum of both thresholds to appear in B.
Disjointness is what keeps the evaluation honest: if an utterance appeared on
both sides, the metrics would measure recording identity rather than speaker
identity, and every number would be inflated.

Sizing matters more than fidelity here. The interesting range for the
population axis runs from tens to a few thousand speakers, and anonymizing
audio is the expensive step, so ``--max-speakers`` and the per-speaker
utterance caps let you buy the population you need without paying for a corpus
you will not use.

Example::

    python scripts/construct_common_voice_subsets.py \\
        --cv-root /data/cv-corpus-27.0/en \\
        --output-dir data/cv27 \\
        --max-speakers 10000 \\
        --max-enroll-utterances 10 \\
        --max-test-utterances 30
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))

from build_cv_common import (  # noqa: E402
    read_clip_durations,
    read_clip_metadata,
    write_kaldi_dir,
)


def assign_utterances(
    utterances: list[tuple[str, float]],
    enroll_seconds: float,
    test_seconds: float,
    max_enroll: int | None,
    max_test: int | None,
) -> tuple[list[str], list[str]]:
    """Split one speaker's utterances into disjoint enrollment and test halves.

    Enrollment is filled first, then test from what remains, so a speaker who
    can only satisfy one threshold still contributes to the enrollment
    population rather than being discarded.

    Args:
        utterances: ``(clip, duration_seconds)`` pairs for one speaker.
        enroll_seconds: Minimum speech required for the enrollment side.
        test_seconds: Minimum speech required for the test side.
        max_enroll: Cap on enrollment utterances, or ``None``.
        max_test: Cap on test utterances, or ``None``.

    Returns:
        ``(enroll_clips, test_clips)``. ``test_clips`` is empty when the speaker
        lacks enough further speech, which is the common case.
    """
    enroll: list[str] = []
    total = 0.0
    index = 0
    while index < len(utterances) and (
        total < enroll_seconds or (max_enroll is not None and len(enroll) < 1)
    ):
        clip, duration = utterances[index]
        enroll.append(clip)
        total += duration
        index += 1
        if max_enroll is not None and len(enroll) >= max_enroll and total >= enroll_seconds:
            break
    if total < enroll_seconds:
        return [], []

    test: list[str] = []
    total = 0.0
    while index < len(utterances) and total < test_seconds:
        clip, duration = utterances[index]
        test.append(clip)
        total += duration
        index += 1
        if max_test is not None and len(test) >= max_test:
            break
    if total < test_seconds:
        return enroll, []
    return enroll, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True,
                        help="locale directory holding clips/ and the TSVs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-enroll-minutes", type=float, default=2.0)
    parser.add_argument("--min-test-minutes", type=float, default=3.0)
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="cap the enrollment population, largest speakers first")
    parser.add_argument("--max-enroll-utterances", type=int, default=None)
    parser.add_argument("--max-test-utterances", type=int, default=None)
    parser.add_argument("--keep-client-ids", action="store_true",
                        help="do not pseudonymise; see docs/data-release.md")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    metadata = read_clip_metadata(args.cv_root)
    print(f"{len(metadata):,} clips described by the corpus metadata")
    durations = read_clip_durations(args.cv_root, metadata)
    print(f"{len(durations):,} clips with a known duration")

    clips_dir = args.cv_root / "clips"
    by_speaker: dict[str, list[tuple[str, float]]] = defaultdict(list)
    gender_of: dict[str, str] = {}
    for clip, (client_id, gender) in metadata.items():
        duration = durations.get(clip)
        if duration is None or not client_id:
            continue
        by_speaker[client_id].append((clip, duration))
        gender_of.setdefault(client_id, gender)
    print(f"{len(by_speaker):,} speakers before filtering")

    enroll_seconds = args.min_enroll_minutes * 60
    test_seconds = args.min_test_minutes * 60

    # Longest speakers first, so a --max-speakers cap keeps those most likely to
    # satisfy both thresholds rather than an arbitrary slice.
    order = sorted(by_speaker, key=lambda s: -sum(d for _, d in by_speaker[s]))

    enroll_spk2utt: dict[str, list[str]] = {}
    test_spk2utt: dict[str, list[str]] = {}
    for client_id in order:
        if args.max_speakers and len(enroll_spk2utt) >= args.max_speakers:
            break
        utterances = sorted(by_speaker[client_id])
        enroll, test = assign_utterances(
            utterances, enroll_seconds, test_seconds,
            args.max_enroll_utterances, args.max_test_utterances,
        )
        if not enroll:
            continue
        enroll_spk2utt[client_id] = enroll
        if test:
            test_spk2utt[client_id] = test

    if not enroll_spk2utt:
        print("No speaker met the enrollment threshold. Lower "
              "--min-enroll-minutes or check the corpus.", file=sys.stderr)
        return 1

    counters: dict[str, int] = defaultdict(int)
    renamed: dict[str, str] = {}
    for client_id in sorted(enroll_spk2utt):
        if args.keep_client_ids:
            renamed[client_id] = client_id
            continue
        gender = gender_of[client_id]
        counters[gender] += 1
        renamed[client_id] = f"spk-{gender}-{counters[gender]:05d}"

    for label, mapping in (("A_enroll", enroll_spk2utt), ("B_test", test_spk2utt)):
        spk2utt = {renamed[c]: [Path(x).stem for x in v] for c, v in mapping.items()}
        utt2spk = {u: s for s, us in spk2utt.items() for u in us}
        spk2gender = {renamed[c]: gender_of[c] for c in mapping}
        out = args.output_dir / label
        write_kaldi_dir(out, spk2utt, utt2spk, spk2gender, clips_dir)

        hours = sum(
            durations[f"{u}.mp3"] for us in mapping.values() for u in
            [Path(x).stem for x in us] if f"{u}.mp3" in durations
        ) / 3600
        by_gender: dict[str, int] = defaultdict(int)
        for g in spk2gender.values():
            by_gender[g] += 1
        print(f"\n{label}: {len(spk2utt):,} speakers "
              f"({by_gender['f']:,} f, {by_gender['m']:,} m, {by_gender['u']:,} unknown), "
              f"{len(utt2spk):,} utterances, {hours:.1f} h")
        print(f"  written to {out}")

    overlap = {u for us in enroll_spk2utt.values() for u in us} & {
        u for us in test_spk2utt.values() for u in us
    }
    print(f"\nutterance overlap between A and B: {len(overlap)} (must be 0)")
    print(f"B speakers also in A: {len(set(test_spk2utt) & set(enroll_spk2utt))}"
          f" / {len(test_spk2utt)}")
    return 0 if not overlap else 1


if __name__ == "__main__":
    raise SystemExit(main())
