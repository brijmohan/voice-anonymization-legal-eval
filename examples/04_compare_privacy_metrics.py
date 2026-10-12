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
    is_separable,
    local_information_disclosure,
    similarity_rank_disclosure,
)
from legal_eval.embeddings import (  # noqa: E402
    build_speaker_embeddings,
    build_test_embeddings,
)
from legal_eval.io import ScoreMatrix  # noqa: E402
from legal_eval.metrics.eer import (  # noqa: E402
    rocch_eer,  # noqa: E402
    rocch_eer_standard_error,
    standard_error_of_mean,
)
from legal_eval.metrics.linkability import linkability  # noqa: E402
from legal_eval.scoring import cosine_score_matrix  # noqa: E402
from legal_eval.sweeps import singling_out_sweep  # noqa: E402

COUNTS = (2, 5, 10, 20, 50)
SLOPES = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
N_DRAWS = 10


def _bootstrap_eer_sem(scores, true_rows, n_reps: int = 20, seed: int = 0):
    """Kept as a name; the implementation now lives in the package.

    It moved so that somebody following docs/protocol.md can call it
    rather than copy it out of an example.
    """
    return rocch_eer_standard_error(scores, true_rows, n_reps, seed)


def _sem_from(values) -> float | None:
    """Kept as a name; see legal_eval.metrics.standard_error_of_mean."""
    return standard_error_of_mean(values)


@dataclass
class System:
    """One anonymization system, as everything the five metrics need.

    All of it comes from one embedding set and one enrollment and test split,
    which is what makes the metrics comparable.

    Attributes:
        dataset: The dataset these systems share, for example ``libri_dev``.
            Only systems on the same dataset can be ranked against each other.
        label: The anonymization system, or ``original``.
    """

    dataset: str
    label: str
    enroll_embeddings: Mapping[str, np.ndarray]
    test_spk2utt: Mapping[str, Sequence[str]]
    utt2embedding: Mapping[str, np.ndarray]
    matrix: ScoreMatrix = field(init=False)

    @property
    def name(self) -> str:
        """Dataset and system together, for printing."""
        return f"{self.dataset} {self.label}"

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
    for label, source in sources.items():
        enroll_utts, test_utts = split_corpus(spk2utt, n_first=4)
        systems.append(
            System(
                dataset="synthetic",
                label=label,
                enroll_embeddings=build_speaker_embeddings(enroll_utts, source),
                test_spk2utt=test_utts,
                utt2embedding=source,
            )
        )
    return systems


def build_vpc_systems(results_dir: Path) -> list[System]:
    """One system per original and fully anonymized VoicePrivacy dataset pair."""
    from legal_eval.vpc import (
        find_embedding_dirs,
        load_vpc_embeddings,
        pair_datasets,
        unpaired_datasets,
    )

    discovered = find_embedding_dirs(results_dir)
    pairs = pair_datasets(discovered)
    # A directory that does not follow VPC's naming is discovered and then
    # dropped, so the comparison would quietly cover less than it appears to.
    orphans = unpaired_datasets(discovered, pairs)
    if orphans:
        print(f"WARNING: {len(orphans)} dataset(s) hold embeddings but pair with "
              f"nothing, so they contribute no system:")
        for name in orphans:
            print(f"  {name}")
        print("  VPC naming is <base>_enrolls<suffix> and "
              "<base>_trials_<kind><suffix>.")

    systems, seen = [], set()
    for pair in pairs:
        if pair.scenario not in ("oo", "aa"):
            continue
        # A shared ASV directory holds one original copy and several anonymized
        # ones, and the original is paired against each, so it would otherwise
        # be built once per system.
        key = (pair.base, pair.scenario, pair.system)
        if key in seen:
            continue
        seen.add(key)
        enroll = load_vpc_embeddings(pair.enroll_dir)
        test = load_vpc_embeddings(pair.trial_dir)
        shared = sorted(set(enroll.speakers) & set(test.speakers))
        if len(shared) < 10:
            continue
        label = pair.system.lstrip("_") if pair.scenario == "aa" else "original"
        systems.append(
            System(
                dataset=pair.base,
                label=label,
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


def fitted_slope(system: System) -> tuple[float, bool]:
    """LID's Platt slope, fit on this system's own scores after z-normalisation.

    Also reports whether the development scores are separable, in which case the
    logistic fit has no finite maximiser and the slope returned is the
    optimiser's stopping point rather than a property of the data.
    """
    mated, nonmated = calibration_samples(system.matrix.scores, system.true_rows)
    separable = is_separable(mated, nonmated)
    if nonmated.size > 200_000:
        nonmated = np.random.default_rng(0).choice(nonmated, 200_000, replace=False)
    slope, _ = fit_calibration(mated, nonmated)
    return slope, separable


def across_population(
    system: System, slope: float, separable: bool, counts, n_folds: int, n_runs: int
):
    """Every metric at each closed-set size, from the one embedding set."""
    rows = system.true_rows
    scores = system.matrix.scores
    n_enroll, n_trials = scores.shape
    usable = [n for n in counts if 2 <= n <= n_enroll]

    mated = scores[rows, np.arange(n_trials)]
    mask = np.ones_like(scores, dtype=bool)
    mask[rows, np.arange(n_trials)] = False
    nonmated = scores[mask]
    eer = rocch_eer(mated, nonmated)
    # The EER is usually quoted as a single number, which makes it the only
    # metric here that cannot be compared against noise. Bootstrap over trials
    # so its ordering of two systems can be judged on the same footing.
    eer_sem = rocch_eer_standard_error(scores, rows)

    sweep = singling_out_sweep(
        system.enroll_embeddings,
        system.test_spk2utt,
        system.utt2embedding,
        conversation_length=1,
        speaker_counts=usable,
        n_runs=n_runs,
        n_folds=n_folds,
        seed=0,
    )
    iso_mean = sweep.mean()
    iso_values = sweep.values

    print(f"\n=== {system.name} ===")
    print(
        f"ROCCH EER {eer:.4f}   1-EER {1 - eer:.4f}   "
        f"LID calibration slope w {slope:.3f}"
    )
    if separable:
        print(
            "  WARNING: the development scores are separable, so the logistic "
            "fit has no"
        )
        print(
            "  finite maximiser. The slope above is where the optimiser stopped, "
            "and ALID"
        )
        print("  saturates at log2(N) regardless of how far it stopped.")
    print("The EER has no population axis at all, which is the axis at issue.")
    head = (
        f"{'N':>5} {'SingOut':>8} {'Link':>8} {'SRD IdR':>8} {'SRD MeanD':>10} "
        f"{'SRD MaxD':>9} {'LID ALID':>9} {'LID PDR':>8} {'log2 N':>7}"
    )
    print(head)
    print("-" * len(head))
    out = []
    for n in usable:
        # Each replicate resamples trials with replacement and then draws the
        # closed sets. Drawing closed sets alone is not enough: when N equals
        # the number of enrollment speakers every speaker is always in the set,
        # the draw is deterministic, and the spread collapses to exactly zero.
        # That is the largest and most interesting N, so a spread that vanishes
        # there would make every difference look resolved.
        link_runs, srd_runs, lid_runs = [], [], []
        for d in range(N_DRAWS):
            boot = np.random.default_rng(1000 + d)
            pick = boot.integers(0, n_trials, size=n_trials)
            sub, sub_rows = scores[:, pick], rows[pick]
            sub_matrix = ScoreMatrix(
                scores=sub,
                enroll_speakers=list(system.matrix.enroll_speakers),
                test_speakers=[system.matrix.test_speakers[i] for i in pick],
                metadata=dict(system.matrix.metadata),
            )
            link_runs.append(
                float(np.mean(linkability(sub_matrix, n_enroll_speakers=n,
                                          n_runs=1, seed=d)))
            )
            srd_runs.append(similarity_rank_disclosure(
                sub, sub_rows, n, rng=np.random.default_rng(d)))
            lid_runs.append(local_information_disclosure(
                sub, sub_rows, n, slope=slope, rng=np.random.default_rng(d)))
        row = {
            "n": n,
            "singling_out": iso_mean.get(n),
            "singling_out_sem": _sem_from(iso_values.get(n)),
            "linkability": float(np.mean(link_runs)),
            "linkability_sem": _sem_from(link_runs),
            "srd_idr": float(np.mean([r.identification_rate for r in srd_runs])),
            "srd_mean": float(np.mean([r.mean for r in srd_runs])),
            "srd_mean_sem": _sem_from([r.mean for r in srd_runs]),
            "srd_max": float(np.mean([r.maximum for r in srd_runs])),
            "lid_mean": float(np.mean([r.mean for r in lid_runs])),
            "lid_mean_sem": _sem_from([r.mean for r in lid_runs]),
            "lid_pdr": float(np.mean([r.positive_rate for r in lid_runs])),
            "lid_max": float(np.mean([r.maximum for r in lid_runs])),
        }
        out.append(row)
        so = row["singling_out"]
        so_text = f"{so:>8.4f}" if so is not None else f"{'-':>8}"
        print(
            f"{n:>5} {so_text} {row['linkability']:>8.4f} {row['srd_idr']:>8.4f} "
            f"{row['srd_mean']:>10.4f} {row['srd_max']:>9.4f} "
            f"{row['lid_mean']:>9.4f} {row['lid_pdr']:>8.4f} {np.log2(n):>7.3f}"
        )
    return {
        "eer": eer,
        "eer_sem": eer_sem,
        "one_minus_eer": 1 - eer,
        "slope": slope,
        "by_n": out,
    }


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


def _partial_order(items, value, error):
    """Group systems into tiers, where a tier holds systems not shown to differ.

    Two systems are placed in the same tier when the gap between their means is
    no larger than the sum of their standard errors. Without this an ordering
    can be reported from a gap of 0.001, or from an exact tie broken by sort
    order, and two metrics then appear to disagree when neither has resolved
    anything. At the ceiling that is the normal situation.
    """
    ordered = sorted(items, key=value)
    tiers = [[ordered[0]]]
    for item in ordered[1:]:
        previous = tiers[-1][-1]
        gap = value(item) - value(previous)
        tolerance = (error(item) or 0.0) + (error(previous) or 0.0)
        if gap <= tolerance:
            tiers[-1].append(item)
        else:
            tiers.append([item])
    return tiers


def _render(tiers) -> str:
    return "  <  ".join(" ~ ".join(sorted(tier)) for tier in tiers)


def report_rankings(systems: dict) -> dict:
    """Do the metrics order the systems the same way, within each dataset?

    Comparing across datasets would not be a comparison of systems, so the
    question is asked separately for each dataset. Systems a metric has not
    separated are shown joined by ``~``, and a metric only counts as
    disagreeing with another when the two produce different tiers.
    """
    groups: dict[str, list[str]] = {}
    for name, res in systems.items():
        groups.setdefault(res["dataset"], []).append(name)

    print("\n" + "=" * 78)
    print("Do the metrics agree on the ordering, least disclosure first?")
    print("=" * 78)
    print("`~` joins systems a metric has not separated, judged against the")
    print("standard error over runs rather than against an arbitrary epsilon.")

    out: dict[str, dict[str, list[list[str]]]] = {}
    for dataset, names in sorted(groups.items()):
        if len(names) < 2:
            continue
        at = {
            name: next(
                (r for r in systems[name]["by_n"] if r["n"] == 50),
                systems[name]["by_n"][-1],
            )
            for name in names
        }
        size = at[names[0]]["n"]
        # The EER has no runs here, so it has no standard error to quote.
        metrics = {
            "1 - EER": (
                lambda n, rows=at: systems[n]["one_minus_eer"],
                lambda n, rows=at: systems[n].get("eer_sem"),
            ),
            "Singling Out": (
                lambda n, rows=at: rows[n]["singling_out"] or 0.0,
                lambda n, rows=at: rows[n].get("singling_out_sem"),
            ),
            "Linkability": (
                lambda n, rows=at: rows[n]["linkability"],
                lambda n, rows=at: rows[n].get("linkability_sem"),
            ),
            "SRD MeanD": (
                lambda n, rows=at: rows[n]["srd_mean"],
                lambda n, rows=at: rows[n].get("srd_mean_sem"),
            ),
            "LID ALID": (
                lambda n, rows=at: rows[n]["lid_mean"],
                lambda n, rows=at: rows[n].get("lid_mean_sem"),
            ),
        }
        print(f"\n  {dataset}, at N = {size}")
        orders = {}
        for metric, (value, error) in metrics.items():
            tiers = _partial_order(names, value, error)
            orders[metric] = [
                [systems[x]["label"] for x in tier] for tier in tiers
            ]
            print(f"    {metric:<14} " + _render(orders[metric]))
        # A metric that has not resolved a pair another metric has resolved is
        # coarser, not contradictory. Only a resolved pair ordered one way by
        # one metric and the other way by another is a real disagreement.
        resolved: dict[str, dict[tuple[str, str], int]] = {}
        for metric, (value, error) in metrics.items():
            pairs = {}
            for first in names:
                for second in names:
                    if first >= second:
                        continue
                    gap = value(second) - value(first)
                    tolerance = (error(first) or 0.0) + (error(second) or 0.0)
                    if abs(gap) > tolerance:
                        pairs[(first, second)] = 1 if gap > 0 else -1
            resolved[metric] = pairs

        inversions = []
        proposed = list(metrics)
        for i, left in enumerate(proposed):
            for right in proposed[i + 1:]:
                shared = set(resolved[left]) & set(resolved[right])
                for pair in shared:
                    if resolved[left][pair] != resolved[right][pair]:
                        inversions.append((left, right, pair))

        coarser = {m: len(resolved[m]) for m in proposed}
        if inversions:
            print(f"    {len(inversions)} genuine inversion(s): a pair that two")
            print("    metrics both resolve, and order oppositely.")
            for left, right, pair in inversions[:4]:
                a, b = (systems[x]["label"] for x in pair)
                print(f"      {left} and {right} disagree on {a} against {b}")
        else:
            print("    No inversions. Where two metrics both separate a pair,")
            print("    they order it the same way. They differ in how much they")
            print("    resolve: " + ", ".join(
                f"{m} {coarser[m]}" for m in proposed
            ) + f" of {len(names) * (len(names) - 1) // 2} pairs.")
        out[dataset] = {"tiers": orders, "inversions": [
            [left, right, list(pair)] for left, right, pair in inversions
        ], "pairs_resolved": coarser}
    return out


def report_saturation(systems: dict) -> dict:
    """Is the population large enough for the metrics to have any range left?

    Every metric here except the EER is bounded by the population: Linkability
    cannot exceed 1 and the bit metrics cannot exceed ``log2(N)``. When most
    systems sit near those bounds the comparison is measuring the size of the
    evaluation set rather than the systems, and the right response is a larger
    set, not a different metric.
    """
    print("\n" + "=" * 78)
    print("Is the evaluation set large enough to tell the systems apart?")
    print("=" * 78)
    head = (
        f"{'system':<42} {'N':>4} {'Link':>7} {'SRD/log2N':>10} {'LID/log2N':>10}"
    )
    print(head)
    print("-" * len(head))
    out, pinned = {}, 0
    for name, res in systems.items():
        row = res["by_n"][-1]
        ceiling = np.log2(row["n"])
        srd_share = row["srd_mean"] / ceiling
        lid_share = row["lid_mean"] / ceiling
        out[name] = {
            "n": row["n"],
            "linkability": row["linkability"],
            "srd_share_of_ceiling": srd_share,
            "lid_share_of_ceiling": lid_share,
        }
        at_ceiling = row["linkability"] > 0.9 and srd_share > 0.9
        pinned += at_ceiling
        flag = "  <- at the ceiling" if at_ceiling else ""
        print(
            f"{name:<42} {row['n']:>4} {row['linkability']:>7.4f} "
            f"{srd_share:>10.3f} {lid_share:>10.3f}{flag}"
        )
    if pinned:
        print(
            f"\n{pinned} of {len(systems)} conditions are pinned against the "
            "bound. For those the"
        )
        print("metrics cannot separate systems, and no choice among the metrics")
        print("would change that. The binding constraint is the number of")
        print("speakers, not the metric.")
    return out


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
    # Every system is read at the same closed-set size, which is the largest
    # all of them support. Printing the requested size while silently clamping
    # to a smaller one mislabels the whole table.
    n = min([n] + [s.matrix.scores.shape[0] for s in built])
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
        row = {}
        for column in names:
            lid = local_information_disclosure(
                scores, rows, n, slope=slopes[column],
                rng=np.random.default_rng(1),
            )
            row[column] = lid.mean
        table[system.name] = row
        print(f"{system.name:<{width}}" + "".join(f"{row[c]:>12.3f}" for c in names))

    print("\nThe diagonal is what a paper reports: each system calibrated on")
    print("itself. Off the diagonal the same system, on the same data, reports a")
    print("different number. ALID is therefore a property of the system and the")
    print("calibration protocol jointly, not of the system alone.")

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
        slope, separable = slopes[system.name]
        results[system.name] = across_population(
            system, slope, separable, counts, args.folds, args.runs
        )
        results[system.name]["separable"] = separable
        results[system.name]["dataset"] = system.dataset
        results[system.name]["label"] = system.label
        sensitivity[system.name] = calibration_sensitivity(system, slope)

    orders = report_rankings(results)
    report_directions(results)
    saturation = report_saturation(results)
    cross = (
        report_cross_calibration(built, {k: v[0] for k, v in slopes.items()})
        if len(built) > 1
        else {}
    )

    excluded: dict[str, object] = {"count": 0}
    if args.results_dir:
        from legal_eval.vpc import excluded_utterances

        per_dataset = excluded_utterances(args.results_dir)
        total = sum(len(v) for v in per_dataset.values())
        if total:
            from legal_eval.audio_integrity import optimistic_bias_note

            attempted = sum(
                len(s.matrix.test_ids) if hasattr(s.matrix, "test_ids") else 0
                for s in built
            ) or total
            excluded = {
                "count": total,
                "ids": per_dataset,
                "reason": optimistic_bias_note(total, attempted + total),
            }
            print(f"\nWARNING: {total} utterance(s) were excluded by the "
                  f"extraction because the attacker could not score them.")
            for dataset, ids in per_dataset.items():
                print(f"  {dataset}: {len(ids)}, e.g. {ids[0]}")
            print("  A destroyed utterance scores at chance, which is the best "
                  "privacy score attainable, so the reported privacy is an "
                  "upper bound.")

    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "systems": results,
                    "calibration_sensitivity": sensitivity,
                    "orderings": orders,
                    "saturation": saturation,
                    "cross_calibration": cross,
                    "fitted_slopes": {k: v[0] for k, v in slopes.items()},
                    "separable": {k: v[1] for k, v in slopes.items()},
                    # Declared one way or the other, always. A destroyed
                    # utterance scores at chance, which is the best privacy
                    # score there is, so a silent exclusion flatters the
                    # system. preflight refuses a report that omits this.
                    "excluded": excluded,
                },
                indent=2,
                default=float,
            )
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
