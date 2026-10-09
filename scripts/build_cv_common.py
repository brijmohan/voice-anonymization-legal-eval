"""Shared Common Voice reading and Kaldi writing.

Used by both subset builders: the one that replays the paper's published clip
lists, and the one that constructs equivalent subsets from a current release.
"""

from __future__ import annotations

import csv
from pathlib import Path

#: Common Voice ships several TSVs. validated.tsv is the superset of clips that
#: passed community review, which is what a subset should be drawn from.
TSV_CANDIDATES = ("validated.tsv", "other.tsv", "train.tsv", "dev.tsv", "test.tsv")

#: Common Voice gender strings, mapped to the single letter Kaldi expects.
#: The vocabulary changed around release 17, so both spellings are accepted.
GENDER_MAP = {
    "male": "m",
    "male_masculine": "m",
    "female": "f",
    "female_feminine": "f",
}


def read_clip_metadata(cv_root: Path) -> dict[str, tuple[str, str]]:
    """Map clip filename to ``(client_id, gender letter)``.

    Args:
        cv_root: Locale directory holding ``clips/`` and the TSVs.

    Returns:
        Mapping from clip filename to speaker and gender, where gender is
        ``f``, ``m``, or ``u`` when Common Voice records none.

    Raises:
        FileNotFoundError: If no usable TSV is present.
    """
    found = [name for name in TSV_CANDIDATES if (cv_root / name).exists()]
    if not found:
        raise FileNotFoundError(
            f"no Common Voice TSV in {cv_root}. Expected one of "
            f"{', '.join(TSV_CANDIDATES)} beside a clips/ directory."
        )

    metadata: dict[str, tuple[str, str]] = {}
    for name in found:
        with open(cv_root / name, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                clip = (row.get("path") or "").strip()
                if not clip or clip in metadata:
                    continue
                gender = (row.get("gender") or "").strip().lower()
                metadata[clip] = (
                    (row.get("client_id") or "").strip(),
                    GENDER_MAP.get(gender, "u"),
                )
    return metadata


def read_clip_durations(
    cv_root: Path, metadata: dict[str, tuple[str, str]] | None = None
) -> dict[str, float]:
    """Read per-clip durations in seconds.

    Releases from about version 13 onward ship ``clip_durations.tsv``, which is
    the only practical source: probing a million MP3s takes hours, and the
    subset construction needs durations for every candidate speaker.

    Args:
        cv_root: Locale directory.
        metadata: If given, restrict to these clips.

    Returns:
        Mapping from clip filename to duration in seconds.

    Raises:
        FileNotFoundError: If the release ships no duration table, with a
            pointer to the workaround.
    """
    path = cv_root / "clip_durations.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Releases before about version 13 do not ship "
            "clip durations. Either use a newer release, or generate the file "
            "with a line per clip as 'clip<TAB>duration[ms]'."
        )

    durations: dict[str, float] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        clip_field = next((f for f in fields if f.strip().lower() in ("clip", "path")), None)
        ms_field = next((f for f in fields if "duration" in f.strip().lower()), None)
        if not clip_field or not ms_field:
            raise ValueError(f"{path} has unexpected columns: {fields}")
        for row in reader:
            clip = (row.get(clip_field) or "").strip()
            if not clip or (metadata is not None and clip not in metadata):
                continue
            try:
                durations[clip] = float(row[ms_field]) / 1000.0
            except (TypeError, ValueError):
                continue
    return durations


def write_kaldi_dir(
    output_dir: Path,
    spk2utt: dict[str, list[str]],
    utt2spk: dict[str, str],
    spk2gender: dict[str, str],
    clips_dir: Path,
) -> None:
    """Write a Kaldi data directory.

    ``wav.scp`` decodes each MP3 through a pipe rather than transcoding up
    front, which keeps this fast and adds no second copy of the corpus.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "wav.scp", "w", encoding="utf-8") as handle:
        for utt in sorted(utt2spk):
            clip = clips_dir / f"{utt}.mp3"
            handle.write(f"{utt} ffmpeg -v 8 -i {clip} -f wav -ar 16000 -ac 1 - |\n")

    with open(output_dir / "utt2spk", "w", encoding="utf-8") as handle:
        for utt in sorted(utt2spk):
            handle.write(f"{utt} {utt2spk[utt]}\n")

    with open(output_dir / "spk2utt", "w", encoding="utf-8") as handle:
        for speaker in sorted(spk2utt):
            handle.write(f"{speaker} {' '.join(sorted(spk2utt[speaker]))}\n")

    with open(output_dir / "spk2gender", "w", encoding="utf-8") as handle:
        for speaker in sorted(spk2gender):
            handle.write(f"{speaker} {spk2gender[speaker]}\n")
