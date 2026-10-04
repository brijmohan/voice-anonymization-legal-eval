#!/usr/bin/env python3
"""Redraw the paper's figure and check it against the values quoted in its text.

Run with::

    python examples/02_reproduce_paper_figures.py --output-dir figures

This uses the published results shipped in ``data/paper_results/``, which cover
all nine panels of the paper's Figure 1: Singling Out, Linkability and 1-EER,
each at L = 1, 3 and 30, each with four attacker curves.

Checking a figure against the prose that describes it is a weak test of the code
but a real test of the data: it catches a results file that has drifted from what
was actually published. For a test of the implementation, see
``examples/03_verify_against_score_matrices.py``, which recomputes every
published Linkability point from the original score matrices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from legal_eval.paper import load_paper_results
from legal_eval.plotting import plot_paper_figure, plot_single_metric

#: (metric, L, attacker, N, expected, tolerance, quotation from Section 5).
PAPER_CLAIMS = [
    ("linkability", 1, "original", 20, 0.82, 0.01,
     "at L=1 the original data exhibit a Linkability of 82% with 20 enrollment speakers"),
    ("linkability", 1, "original", 10020, 0.35, 0.01,
     "decreasing to 35% with 10,000 speakers"),
    ("linkability", 1, "informed", 20, 0.77, 0.01,
     "the Informed attacker's Linkability decreases to 77% for 20 enrollment speakers"),
    ("linkability", 1, "informed", 10020, 0.21, 0.01,
     "and 21% for 10,000 speakers"),
    ("linkability", 3, "original", 20, 0.95, 0.01,
     "for L=3 and L=30, the original data reach very high values (up to 94-95%)"),
    ("linkability", 30, "original", 20, 0.94, 0.01,
     "for L=3 and L=30, the original data reach very high values (up to 94-95%)"),
    ("linkability", 30, "informed", 20, 0.96, 0.01,
     "even after anonymization the Informed attacker achieves up to 96% (Linkability)"),
    ("singling_out", 30, "informed", 30, 0.99, 0.01,
     "and 99% (Singling Out)"),
    ("singling_out", 1, "original", 30, 0.57, 0.01,
     "for Singling Out, the original data range from 57% down to 38%"),
]


def check_claims(results) -> int:
    """Check each quoted value. Returns the number that did not match."""
    print("Checking the published results against the paper's text\n")
    failures = 0
    for metric, length, attacker, count, expected, tolerance, quote in PAPER_CLAIMS:
        curve = results[metric][length][attacker]
        nearest = min(curve.speaker_counts, key=lambda n: abs(n - count))
        observed = curve.mean()[nearest]
        ok = abs(observed - expected) <= tolerance
        failures += not ok
        print(
            f"  [{'ok' if ok else 'FAIL'}] {metric:<13} L={length:<3} {attacker:<14}"
            f" N={nearest:<6} observed {observed:.3f}, paper states {expected:.2f}"
        )
        print(f'         "{quote}"')
    return failures


def check_eer_is_flatter_than_the_legal_metrics(results) -> int:
    """The paper's central contrast, as a property of the published data.

    Each metric is swept over the same speaker counts. Within a curve, the EER
    should move far less than either legal metric.
    """
    print("\nChecking that 1-EER varies less than the legal metrics\n")

    def spread(metric: str) -> float:
        widths = []
        for length in (1, 3, 30):
            for curve in results[metric][length].values():
                values = list(curve.mean().values())
                widths.append(max(values) - min(values))
        return sum(widths) / len(widths)

    eer, linkability, singling = spread("eer"), spread("linkability"), spread("singling_out")
    print(f"  mean within-curve range   1-EER        {eer:.3f}")
    print(f"                            Linkability  {linkability:.3f}")
    print(f"                            Singling Out {singling:.3f}")

    if eer < linkability and eer < singling:
        print("\n  [ok] 1-EER is the flattest of the three, as the paper reports.")
        return 0
    print("\n  [FAIL] 1-EER is not the flattest curve.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    results = load_paper_results()
    failures = check_claims(results)
    failures += check_eer_is_flatter_than_the_legal_metrics(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = args.output_dir / "paper_figure.pdf"
    plot_paper_figure(results, figure, conversation_lengths=(1, 3, 30))
    print(f"\nWrote the full nine-panel figure to {figure}")

    for metric in ("singling_out", "linkability", "eer"):
        for length in (1, 3, 30):
            path = args.output_dir / f"{metric}_L{length}.pdf"
            plot_single_metric(
                results[metric][length],
                metric=metric,
                path=path,
                title=f"{metric.replace('_', ' ').title()}, $L = {length}$",
            )
    print(f"Wrote 9 per-panel figures to {args.output_dir}")

    if failures:
        print(f"\n{failures} check(s) did not match.")
        return 1
    print("\nEvery value the paper quotes reproduces from the shipped results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
