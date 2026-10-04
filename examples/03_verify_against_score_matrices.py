#!/usr/bin/env python3
"""Recompute the paper's Linkability curves from the original score matrices.

This is the strongest check available on this implementation: it takes the exact
cosine score matrices the paper's numbers came from, recomputes Linkability with
the code in this package, and compares point by point against the published
curves.

Run with::

    python examples/03_verify_against_score_matrices.py --root /path/to/IS25

The score matrices are 832 MB each (22,024 x 4,949 float64) and are not in this
repository. ``docs/reproduction.md`` says where to get them. Each condition is
loaded, checked and freed in turn, so peak memory stays around 1 GB.

Expected result: mean absolute difference below 0.002 and worst case below 0.01
at every point. The residual is Monte Carlo noise from averaging only five runs,
and it shrinks under ``--estimator exact``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from legal_eval.io import ScoreMatrix
from legal_eval.metrics.linkability import count_beaters, linkability

#: attacker -> L -> (directory, score matrix, published json, key within it)
CONDITIONS = {
    "original": {
        1: ("cnil_linkability", "plot1_score_matrix.npy", "plot1_scores_step100.json", None),
        3: ("cnil_linkability_plot2", "plot2_score_matrix_L3.npy", "plot2_scores_step100.json", "3"),
        30: ("cnil_linkability_plot2", "plot2_score_matrix_L30.npy", "plot2_scores_step100.json", "30"),
    },
    "informed": {
        1: ("cnil_linkability_plot1_anon", "plot1_score_matrix_anon.npy", "plot1_scores_step100_anon.json", None),
        3: ("cnil_linkability_plot2_anon", "plot2_score_matrix_L3.npy", "plot2_scores_step100_anon.json", "3"),
        30: ("cnil_linkability_plot2_anon", "plot2_score_matrix_L30.npy", "plot2_scores_step100_anon.json", "30"),
    },
    "semi_informed": {
        1: ("cnil_linkability_plot1_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310",
            "plot1_score_matrix_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310.npy",
            "plot1_scores_step100_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310.json", None),
        3: ("cnil_linkability_plot2_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310",
            "plot2_score_matrix_L3.npy",
            "plot2_scores_step100_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310.json", "3"),
        30: ("cnil_linkability_plot2_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310",
             "plot2_score_matrix_L30.npy",
             "plot2_scores_step100_anon_ATTACKED_BY_am_nsf_dense_random__CNIL202310.json", "30"),
    },
    "ignorant": {
        1: ("cnil_linkability_plot1_anon_ATTACKED_BY_IGNORANT__CNIL202310",
            "plot1_score_matrix_anon_ATTACKED_BY_IGNORANT__CNIL202310.npy",
            "plot1_scores_step100_anon_ATTACKED_BY_IGNORANT__CNIL202310.json", None),
        3: ("cnil_linkability_plot2_anon_ATTACKED_BY_IGNORANT__CNIL202310",
            "plot2_score_matrix_L3.npy",
            "plot2_scores_step100_anon_ATTACKED_BY_IGNORANT__CNIL202310.json", "3"),
        30: ("cnil_linkability_plot2_anon_ATTACKED_BY_IGNORANT__CNIL202310",
             "plot2_score_matrix_L30.npy",
             "plot2_scores_step100_anon_ATTACKED_BY_IGNORANT__CNIL202310.json", "30"),
    },
}


def speaker_order(index_file: Path) -> list[str]:
    """Invert a ``{speaker: index}`` JSON file into index order."""
    with open(index_file, encoding="utf-8") as handle:
        mapping = json.load(handle)
    order: list[str | None] = [None] * len(mapping)
    for speaker, index in mapping.items():
        order[index] = speaker
    if any(s is None for s in order):
        raise ValueError(f"{index_file} does not cover a contiguous index range")
    return order  # type: ignore[return-value]


def check(root: Path, attacker: str, length: int, spec, n_runs: int, estimator: str):
    """Recompute one condition and compare against its published curve."""
    directory, matrix_file, published_file, key = spec
    base = root / directory
    matrix = ScoreMatrix(
        np.load(base / matrix_file),
        speaker_order(base / "enroll_spk2idx"),
        speaker_order(base / "trial_spk2idx"),
        {"attacker": attacker, "conversation_length": length},
    )
    beaters = count_beaters(matrix)

    with open(base / published_file, encoding="utf-8") as handle:
        published = json.load(handle)
    if key is not None:
        published = published[key]

    differences = []
    for count_str, runs in published.items():
        count = int(count_str)
        if count > matrix.scores.shape[0]:
            continue
        expected = float(np.mean(runs))
        observed = float(
            np.mean(
                linkability(
                    matrix,
                    n_enroll_speakers=count,
                    n_runs=n_runs,
                    seed=0,
                    estimator=estimator,
                    beaters=beaters,
                )
            )
        )
        differences.append(abs(observed - expected))

    del matrix, beaters
    gc.collect()
    return np.array(differences)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True,
        help="directory holding the cnil_linkability* experiment directories",
    )
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--estimator", choices=("sampling", "exact"), default="sampling")
    parser.add_argument(
        "--tolerance", type=float, default=0.01,
        help="largest acceptable absolute difference at any point",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"{args.root} is not a directory", file=sys.stderr)
        return 2

    print(f"Recomputing Linkability from the score matrices in {args.root}")
    print(f"estimator={args.estimator}, runs={args.n_runs}, tolerance={args.tolerance}\n")
    print(f"{'attacker':<15}{'L':>4}{'points':>8}{'mean diff':>12}{'max diff':>11}  ")
    print("-" * 50)

    worst = 0.0
    checked = 0
    missing = []
    for attacker, by_length in CONDITIONS.items():
        for length, spec in by_length.items():
            if not (args.root / spec[0] / spec[1]).exists():
                missing.append(f"{attacker} L={length}")
                continue
            differences = check(args.root, attacker, length, spec, args.n_runs, args.estimator)
            worst = max(worst, float(differences.max()))
            checked += differences.size
            print(
                f"{attacker:<15}{length:>4}{differences.size:>8}"
                f"{differences.mean():>12.5f}{differences.max():>11.5f}"
            )

    print("-" * 50)
    if missing:
        print(f"skipped (score matrix not found): {', '.join(missing)}")
    if not checked:
        print("\nNothing was checked. Is --root pointing at the right directory?")
        return 2

    print(f"\n{checked} published points recomputed, worst difference {worst:.5f}")
    if worst > args.tolerance:
        print(f"FAIL: exceeds the tolerance of {args.tolerance}")
        return 1
    print("PASS: every point agrees within tolerance.")
    print(
        "\nThe residual is Monte Carlo noise from averaging five runs. Pass\n"
        "--estimator exact to compare against the closed form instead."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
