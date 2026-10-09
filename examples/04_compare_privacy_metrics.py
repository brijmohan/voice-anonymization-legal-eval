#!/usr/bin/env python3
"""Run every recent privacy metric on one artifact and compare them.

Computes the ROCCH EER, Singling Out, Linkability, Similarity Rank Disclosure
and Local Information Disclosure from one set of embeddings and one split, so
that any difference between the metrics is the metric and not the pipeline.
Published numbers cannot be compared this way: they come from different
corpora, different attackers and different splits, and those differences are
larger than the differences between the metrics.

Three questions are asked of the results:

1. Do the metrics rank the same systems in the same order?
2. Which way does each one move as the attacker's search space ``N`` grows?
3. How much of each number is set by the data, and how much by the protocol?

Question 3 matters most. Linkability, SRD and Singling Out are determined by
the data and the split. LID additionally needs a calibration, and because the
softmax cancels the intercept, the single fitted slope is the only thing that
survives into the reported bits. Varying that slope alone moves ALID by
several bits on identical data, so an ALID figure is not interpretable without
the calibration protocol that produced it.

LID and SRD are our reading of their authors' published definitions. See
``legal_eval/comparison.py``.

Usage::

    python examples/04_compare_privacy_metrics.py --demo
    python examples/04_compare_privacy_metrics.py --results-dir exp/asv_ssl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from legal_eval.comparison import (  # noqa: E402
    calibration_samples,
    fit_calibration,
    local_information_disclosure,
    similarity_rank_disclosure,
)
from legal_eval.embeddings import (  # noqa: E402
    build_speaker_embeddings,
    build_test_embeddings,
)
from legal_eval.io import ScoreMatrix  # noqa: E402
from legal_eval.metrics.eer import rocch_eer  # noqa: E402
from legal_eval.metrics.linkability import linkability  # noqa: E402
from legal_eval.scoring import cosine_score_matrix  # noqa: E402
from legal_eval.sweeps import singling_out_sweep  # noqa: E402

COUNTS = (2, 5, 10, 20, 50)
SLOPES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@dataclass
class System:
    """One anonymization system, as everything the five metrics need.

    All of it comes from one embedding set and one enrollment and test split,
    which is what makes the metrics comparable.
    """

    name: str
    enroll_embeddings: Mapping[str, np.ndarray]
    test_spk2utt: Mapping[str, Sequence[str]]
    utt2embedding: Mapping[str, np.ndarray]
    matrix: ScoreMatrix = field(init=False)

    def __post_init__(self) -> None:
        test, _ = build_test_embeddings(
            self.test_spk2utt, self.utt2embedding, 1, "first"
        )
        self.matrix = cosine_score_matrix(self.enroll_embeddings, test)

    @property
    def true_rows(self) -> np.ndarray:
        index = self.matrix.enroll_index
        return np.array([index[s] for s in self.matrix.test_speakers])


def build_demo_systems(n_speakers: int = 300, seed: int = 0) -> list[System]:
    """Three synthetic systems of differing strength, needing no corpus."""
    from legal_eval.demo import simulate_anonymization, split_corpus, synthetic_corpus

    spk2utt, embeddings = synthetic_corpus(
        n_speakers=n_speakers, n_utterances=10, seed=seed
    )
    sources = {
        "original": embeddings,
        "weak anonymization": simulate_anonymization(embeddings, 0.60, seed=seed + 1),
        "strong anonymization": simulate_anonymization(embeddings, 0.95, seed=seed + 2),
    }
    systems = []
    for name, source in sources.items():
        enroll_utts, test_utts = split_corpus(spk2utt, n_first=4)
        systems.append(
            System(
                name=name,
                enroll_embeddings=build_speaker_embeddings(enroll_utts, source),
                test_spk2utt=test_utts,
                utt2embedding=source,
            )
        )
    return systems


def build_vpc_systems(results_dir: Path) -> list[System]:
    """One system per original and fully anonymized VoicePrivacy dataset pair."""
    from legal_eval.vpc import find_embedding_dirs, load_vpc_embeddings, pair_datasets

    systems = []
    for pair in pair_datasets(find_embedding_dirs(results_dir)):
        if pair.scenario not in ("oo", "aa"):
            continue
        enroll = load_vpc_embeddings(pair.enroll_dir)
        test = load_vpc_embeddings(pair.trial_dir)
        shared = sorted(set(enroll.speakers) & set(test.speakers))
        if len(shared) < 10:
            continue
        label = {"oo": "original", "aa": "anonymized"}[pair.scenario]
        systems.append(
            System(
                name=f"{pair.base} {label}",
                enroll_embeddings=build_speaker_embeddings(
                    {s: u for s, u in enroll.spk2utt.items() if s in shared},
                    enroll.embeddings,
                ),
                test_spk2utt={
                    s: u for s, u in test.spk2utt.items() if s in shared
                },
                utt2embedding=test.embeddings,
            )
        )
    return systems


def fitted_slope(system: System) -> float:
    """LID's Platt slope, fit on this system's own scores after z-normalisation."""
    mated, nonmated = calibration_samples(system.matrix.scores, system.true_rows)
    if nonmated.size > 200_000:
        nonmated = np.random.default_rng(0).choice(nonmated, 200_000, replace=False)
    slope, _ = fit_calibration(mated, nonmated)
    return slope


def across_population(system: System, slope: float, counts, n_folds: int, n_runs: int):
    """Every metric at each closed-set size, from the one embedding set."""
    rows = system.true_rows
    scores = system.matrix.scores
    n_enroll, n_trials = scores.shape
    usable = [n for n in counts if 2 <= n <= n_enroll]

    mated = scores[rows, np.arange(n_trials)]
    mask = np.ones_like(scores, dtype=bool)
    mask[rows, np.arange(n_trials)] = False
    eer = rocch_eer(mated, scores[mask])

    iso = singling_out_sweep(
        system.enroll_embeddings,
        system.test_spk2utt,
        system.utt2embedding,
        conversation_length=1,
        speaker_counts=usable,
        n_runs=n_runs,
        n_folds=n_folds,
        seed=0,
    ).mean()

    print(f"\n=== {system.name} ===")
    print(
        f"ROCCH EER {eer:.4f}   1-EER {1 - eer:.4f}   "
        f"LID calibration slope w {slope:.3f}"
    )
    print("The EER has no population axis at all, which is the axis at issue.")
    head = (
        f"{'N':>5} {'SingOut':>8} {'Link':>8} {'SRD IdR':>8} {'SRD MeanD':>10} "
        f"{'SRD MaxD':>9} {'LID ALID':>9} {'LID PDR':>8} {'log2 N':>7}"
    )
    print(head)
    print("-" * len(head))
    out = []
    for n in usable:
        link = float(
            np.mean(linkability(system.matrix, n_enroll_speakers=n, n_runs=5, seed=0))
        )
        srd = similarity_rank_disclosure(
            scores, rows, n, rng=np.random.default_rng(1)
        )
        lid = local_information_disclosure(
            scores, rows, n, slope=slope, rng=np.random.default_rng(1)
        )
        out.append(
            {
                "n": n,
                "singling_out": iso.get(n),
                "linkability": link,
                "srd_idr": srd.identification_rate,
                "srd_mean": srd.mean,
                "srd_max": srd.maximum,
                "lid_mean": lid.mean,
                "lid_pdr": lid.positive_rate,
                "lid_max": lid.maximum,
            }
        )
        so = iso.get(n)
        so_text = f"{so:>8.4f}" if so is not None else f"{'-':>8}"
        print(
            f"{n:>5} {so_text} {link:>8.4f} {srd.identification_rate:>8.4f} "
            f"{srd.mean:>10.4f} {srd.maximum:>9.4f} {lid.mean:>9.4f} "
            f"{lid.positive_rate:>8.4f} {np.log2(n):>7.3f}"
        )
    return {"eer": eer, "one_minus_eer": 1 - eer, "slope": slope, "by_n": out}


def calibration_sensitivity(system: System, slope: float, n: int = 50):
    """Hold the data and the split fixed, vary only the calibration."""
    rows = system.true_rows
    scores = system.matrix.scores
    n = min(n, scores.shape[0])

    srd = similarity_rank_disclosure(scores, rows, n, rng=np.random.default_rng(1))
    link = float(
        np.mean(linkability(system.matrix, n_enroll_speakers=n, n_runs=5, seed=0))
    )

    print(f"\n--- calibration sensitivity, {system.name}, N = {n} ---")
    print(
        f"Linkability {link:.4f} and SRD MeanD {srd.mean:.4f} use the ranking only, "
        "so no"
    )
    print("choice of calibration can move them. LID is a function of the slope:")
    print(f"{'slope w':>9} {'LID ALID':>10} {'LID PDR':>9} {'LID max':>9}")
    print("-" * 40)
    slopes = sorted(set(SLOPES) | {round(slope, 3)})
    out = []
    for w in slopes:
        lid = local_information_disclosure(
            scores, rows, n, slope=w, rng=np.random.default_rng(1)
        )
        tag = "   <- fitted" if abs(w - round(slope, 3)) < 1e-9 else ""
        out.append(
            {
                "slope": w,
                "lid_mean": lid.mean,
                "lid_pdr": lid.positive_rate,
                "lid_max": lid.maximum,
            }
        )
        print(
            f"{w:>9.3f} {lid.mean:>10.4f} {lid.positive_rate:>9.4f} "
            f"{lid.maximum:>9.4f}{tag}"
        )
    means = [r["lid_mean"] for r in out]
    print(
        f"ALID spans {min(means):.3f} to {max(means):.3f} bits, a range of "
        f"{max(means) - min(means):.3f} bits, on identical data."
    )
    return {
        "n": n,
        "linkability": link,
        "srd_mean": srd.mean,
        "lid_range_bits": max(means) - min(means),
        "by_slope": out,
    }


def report_rankings(systems: dict) -> dict:
    """Do the metrics order the systems the same way?"""
    if len(systems) < 2:
        return {}
    print("\n" + "=" * 78)
    print("Do the metrics agree on the ordering, least disclosure first?")
    print("=" * 78)
    at = {
        name: next((r for r in res["by_n"] if r["n"] == 50), res["by_n"][-1])
        for name, res in systems.items()
    }
    keys = {
        "1 - EER": lambda n: systems[n]["one_minus_eer"],
        "Singling Out": lambda n: at[n]["singling_out"] or 0.0,
        "Linkability": lambda n: at[n]["linkability"],
        "SRD MeanD": lambda n: at[n]["srd_mean"],
        "LID ALID": lambda n: at[n]["lid_mean"],
    }
    orders = {}
    for metric, get in keys.items():
        order = sorted(systems, key=get)
        orders[metric] = order
        print(f"  {metric:<14} " + "  <  ".join(order))
    distinct = {tuple(o) for o in orders.values()}
    if len(distinct) == 1:
        print("\n  All five agree on the ordering. They disagree on the margin,")
        print("  which is what a release decision turns on.")
    else:
        print(f"\n  {len(distinct)} different orderings, so the metrics are not")
        print("  interchangeable even as a ranking.")
    return {k: list(v) for k, v in orders.items()}


def report_directions(systems: dict) -> None:
    print("\n" + "=" * 78)
    print("Direction of travel as the search space grows")
    print("=" * 78)
    for name, res in systems.items():
        rows = res["by_n"]
        if len(rows) < 2:
            continue
        print(f"  {name}")
        for label, key in (
            ("Singling Out", "singling_out"),
            ("Linkability ", "linkability"),
            ("SRD MeanD   ", "srd_mean"),
            ("LID ALID    ", "lid_mean"),
        ):
            a, b = rows[0][key], rows[-1][key]
            if a is None or b is None:
                continue
            print(f"    {label} {a:>8.3f} -> {b:>8.3f}   "
                  f"{'falls' if b < a else 'rises'}")


def report_cross_calibration(built, slopes, n: int = 50) -> dict:
    """Evaluate every system under every system's fitted calibration.

    LID's calibration is normally fit on the system being evaluated. That makes
    the inverse temperature a function of the system, and a system whose scores
    carry less information gets a flatter posterior, which pushes its ALID
    towards zero whatever it leaked. Reusing one system's calibration across all
    of them removes that feedback, and the numbers move a long way, so which
    protocol is used is part of the result.

    Rows are the system evaluated, columns the calibration used.
    """
    print("\n" + "=" * 78)
    print(f"LID ALID in bits under each calibration, N = {n}")
    print("=" * 78)
    names = [s.name for s in built]
    width = max(len(x) for x in names) + 2
    print(" " * width + "".join(f"{'w=' + format(slopes[c], '.2f'):>12}" for c in names))
    print(" " * width + "".join(f"{'(' + c.split()[0] + ')':>12}" for c in names))
    print("-" * (width + 12 * len(names)))
    table = {}
    for system in built:
        scores, rows = system.matrix.scores, system.true_rows
        size = min(n, scores.shape[0])
        row = {}
        for column in names:
            lid = local_information_disclosure(
                scores, rows, size, slope=slopes[column],
                rng=np.random.default_rng(1),
            )
            row[column] = lid.mean
        table[system.name] = row
        print(f"{system.name:<{width}}" + "".join(f"{row[c]:>12.3f}" for c in names))

    print("\nThe diagonal is what a paper reports: each system calibrated on")
    print("itself. Off the diagonal the same system, on the same data, reports a")
    print("different number, and can change sign. ALID is therefore a property of")
    print("the system and the calibration protocol jointly, not of the system.")

    diagonal = [table[x][x] for x in names]
    print(f"\nDiagonal spans {min(diagonal):.3f} to {max(diagonal):.3f} bits.")
    shared = names[0]
    column = [table[x][shared] for x in names]
    print(f"Under one shared calibration ({shared}, w = {slopes[shared]:.2f}) the same")
    print(f"systems span {min(column):.3f} to {max(column):.3f} bits.")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="VoicePrivacy results directory, e.g. exp/asv_ssl")
    parser.add_argument("--demo", action="store_true",
                        help="use synthetic data, needs no corpus")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the full table as JSON")
    parser.add_argument("--counts", default=",".join(str(c) for c in COUNTS))
    parser.add_argument("--folds", type=int, default=3,
                        help="Singling Out folds, lower is faster")
    parser.add_argument("--runs", type=int, default=3,
                        help="Singling Out draws per speaker and fold")
    args = parser.parse_args()

    counts = tuple(int(c) for c in args.counts.split(","))
    built = (
        build_demo_systems() if args.demo or not args.results_dir
        else build_vpc_systems(args.results_dir)
    )
    if not built:
        print("no usable dataset pair found", file=sys.stderr)
        return 1

    results, sensitivity = {}, {}
    slopes = {system.name: fitted_slope(system) for system in built}
    for system in built:
        slope = slopes[system.name]
        results[system.name] = across_population(
            system, slope, counts, args.folds, args.runs
        )
        sensitivity[system.name] = calibration_sensitivity(system, slope)

    orders = report_rankings(results)
    report_directions(results)
    cross = report_cross_calibration(built, slopes) if len(built) > 1 else {}

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "systems": results,
                    "calibration_sensitivity": sensitivity,
                    "orderings": orders,
                    "cross_calibration": cross,
                    "fitted_slopes": slopes,
                },
                indent=2,
                default=float,
            )
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
