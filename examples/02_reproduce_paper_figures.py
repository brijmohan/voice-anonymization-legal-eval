#!/usr/bin/env python3
"""Redraw the published Linkability curves and check them against the paper.

Run with::

    python examples/02_reproduce_paper_figures.py --output-dir figures

This uses the result files shipped in ``data/paper_results/``, which are the
original experiment's outputs for the *Original* (non-anonymized) condition. It
redraws them and asserts that the values quoted in the paper's text come back
out, which is a regression test on the shipped data and on the plotting code.

The anonymized conditions and the Singling Out and EER rows are not shipped as
results: regenerating those needs the cosine score matrices. See
``docs/reproduction.md`` for how to obtain them and what to run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from legal_eval.io import read_results
from legal_eval.plotting import plot_paper_figure, plot_single_metric
from legal_eval.sweeps import SweepResult

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "paper_results"

#: Values stated in Section 5 of the paper, with the tolerance they are quoted to.
PAPER_CLAIMS = [
    # (L, N', expected value, tolerance, quotation from the paper)
    (1, 20, 0.82, 0.01, "at L=1 the original data exhibit a Linkability of 82% with 20 enrollment speakers"),
    (1, 10020, 0.35, 0.01, "decreasing to 35% with 10,000 speakers"),
    (3, 20, 0.95, 0.01, "for L=3 and L=30, the original data reach very high values (up to 94-95%)"),
    (30, 20, 0.94, 0.01, "for L=3 and L=30, the original data reach very high values (up to 94-95%)"),
]


def load(conversation_length: int) -> SweepResult:
    path = DATA_DIR / f"linkability_original_L{conversation_length}.json"
    return SweepResult.from_dict(read_results(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    lengths = (1, 3, 10, 30)
    results = {length: load(length) for length in lengths}

    print("Checking the published results against the paper's text\n")
    failures = 0
    for length, count, expected, tolerance, quote in PAPER_CLAIMS:
        observed = results[length].mean()[count]
        ok = abs(observed - expected) <= tolerance
        failures += not ok
        print(
            f"  [{'ok' if ok else 'FAIL'}] L={length:<2} N'={count:<6} "
            f"observed {observed:.3f}, paper states {expected:.2f}\n"
            f"         \"{quote}\""
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    nested = {"linkability": {length: {"original": results[length]} for length in (1, 3, 30)}}
    grid = args.output_dir / "paper_figure_linkability.pdf"
    plot_paper_figure(nested, grid, conversation_lengths=(1, 3, 30))
    print(f"\nWrote {grid}")
    print("  (the Singling Out and 1-EER rows are empty: see docs/reproduction.md)")

    for length in lengths:
        path = args.output_dir / f"linkability_original_L{length}.pdf"
        plot_single_metric(
            {"original": results[length]},
            metric="linkability",
            path=path,
            title=f"Linkability, original speech, $L = {length}$",
        )
    print(f"Wrote {len(lengths)} per-length figures to {args.output_dir}")

    if failures:
        print(f"\n{failures} claim(s) did not match the shipped data.")
        return 1
    print("\nAll of the paper's quoted Linkability values reproduce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
