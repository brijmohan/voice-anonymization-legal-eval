#!/usr/bin/env python3
"""Package the cosine score matrices for public release.

Converts each matrix to float32, writes it with the sidecar this package reads,
and emits a manifest with SHA-256 checksums and sizes.

Run with::

    python scripts/prepare_score_matrices.py --root <archive> --output-dir release/

Why float32: the scores were computed from float32 x-vectors, so every value is
already exactly representable in float32. The conversion is verified to be
bit-exact before anything is written, and it halves the release from 10.5 GB to
5.2 GB. ``--allow-lossy`` proceeds anyway if a matrix ever fails that check, but
the default is to stop, because a silent precision loss in a privacy metric is
not a tradeoff worth making by accident.

Before releasing anything, read ``docs/data-release.md``. The speaker labels are
internal pseudonyms, and the mapping from those to Common Voice client ids must
not be published alongside them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from legal_eval.io import ScoreMatrix, save_score_matrix

#: attacker -> L -> (directory, matrix filename). Mirrors examples/03.
CONDITIONS = {
    "original": {
        1: ("cnil_linkability", "plot1_score_matrix.npy"),
        3: ("cnil_linkability_plot2", "plot2_score_matrix_L3.npy"),
        30: ("cnil_linkability_plot2", "plot2_score_matrix_L30.npy"),
    },
    "informed": {
        1: ("cnil_linkability_plot1_anon", "plot1_score_matrix_anon.npy"),
        3: ("cnil_linkability_plot2_anon", "plot2_score_matrix_L3.npy"),
        30: ("cnil_linkability_plot2_anon", "plot2_score_matrix_L30.npy"),
    },
    "semi_informed": {
        1: ("cnil_linkability_plot1_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310",
            "plot1_score_matrix_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310.npy"),
        3: ("cnil_linkability_plot2_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310",
            "plot2_score_matrix_L3.npy"),
        30: ("cnil_linkability_plot2_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310",
             "plot2_score_matrix_L30.npy"),
    },
    "ignorant": {
        1: ("cnil_linkability_plot1_anon_ATTACKED_BY_IGNORANT__CNIL202310",
            "plot1_score_matrix_anon_ATTACKED_BY_IGNORANT__CNIL202310.npy"),
        3: ("cnil_linkability_plot2_anon_ATTACKED_BY_IGNORANT__CNIL202310",
            "plot2_score_matrix_L3.npy"),
        30: ("cnil_linkability_plot2_anon_ATTACKED_BY_IGNORANT__CNIL202310",
             "plot2_score_matrix_L30.npy"),
    },
}


def speaker_order(path: Path) -> list[str]:
    """Invert a ``{speaker: index}`` JSON file into index order."""
    with open(path, encoding="utf-8") as handle:
        mapping = json.load(handle)
    order: list[str | None] = [None] * len(mapping)
    for speaker, index in mapping.items():
        order[index] = speaker
    if any(s is None for s in order):
        raise ValueError(f"{path} does not cover a contiguous index range")
    return order  # type: ignore[return-value]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so a 436 MB matrix is not held twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="the experiment archive")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-lossy", action="store_true",
        help="continue even if float32 conversion is not bit-exact",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    total_in = total_out = 0

    for attacker, by_length in CONDITIONS.items():
        for length, (directory, filename) in by_length.items():
            source = args.root / directory / filename
            if not source.exists():
                print(f"  skip {attacker} L={length}: {source} not found")
                continue

            scores = np.load(source)
            as_float32 = scores.astype(np.float32)
            error = float(np.abs(scores - as_float32.astype(np.float64)).max())
            if error != 0.0 and not args.allow_lossy:
                print(
                    f"  STOP {attacker} L={length}: float32 conversion loses "
                    f"up to {error:.3e}. Re-run with --allow-lossy to accept it.",
                    file=sys.stderr,
                )
                return 1

            base = args.root / directory
            matrix = ScoreMatrix(
                scores=as_float32,
                enroll_speakers=speaker_order(base / "enroll_spk2idx"),
                test_speakers=speaker_order(base / "trial_spk2idx"),
                metadata={
                    "attacker": attacker,
                    "conversation_length": length,
                    "dataset": "Common Voice 11.0",
                    "enrollment_set": "A",
                    "test_set": "B",
                    "dtype": "float32",
                    "float32_conversion_exact": error == 0.0,
                    "paper": "Vauquier et al., Interspeech 2025",
                    "speaker_labels": "internal pseudonyms, not Common Voice client ids",
                },
            )
            destination = args.output_dir / f"scores_{attacker}_L{length}.npy"
            save_score_matrix(matrix, destination)

            size = destination.stat().st_size
            total_in += source.stat().st_size
            total_out += size
            manifest.append({
                "file": destination.name,
                "attacker": attacker,
                "conversation_length": length,
                "shape": list(matrix.scores.shape),
                "dtype": "float32",
                "bytes": size,
                "sha256": sha256(destination),
                "float32_conversion_exact": error == 0.0,
            })
            print(f"  {destination.name:<34} {size / 1e6:7.1f} MB  exact={error == 0.0}")

    if not manifest:
        print("Nothing was packaged. Is --root pointing at the archive?", file=sys.stderr)
        return 2

    manifest_path = args.output_dir / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "description": (
                    "Cosine score matrices behind the Linkability and EER results of "
                    "Vauquier et al., 'Legally validated evaluation framework for "
                    "voice anonymization', Interspeech 2025."
                ),
                "rows": "enrollment speakers, Common Voice 11.0 subset A",
                "columns": "test speakers, Common Voice 11.0 subset B",
                "speaker_labels": "internal pseudonyms, not Common Voice client ids",
                "see": "docs/data-release.md",
                "files": manifest,
            },
            handle,
            indent=2,
        )
    print(f"\n{len(manifest)} matrices: {total_in / 1e9:.1f} GB -> {total_out / 1e9:.1f} GB")
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
