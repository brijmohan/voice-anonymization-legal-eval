#!/usr/bin/env python3
"""Put the three Article 29 criteria side by side for every system.

The field reports verification numbers. Opinion 05/2014 asks three questions,
and the quantities in common use answer one and a half of them: six speak to
linkability, one to singling out, and one to inference. A system can therefore
post a large, honest, well-calibrated improvement and have addressed one
criterion of three.

This script makes that visible. Each metric is converted to its distance above
chance, and each system is scored on the fraction of the original recordings'
distance that it removed. Expressed that way the criteria sit on one scale
without pretending the metrics are interchangeable.

What to read. The ``removed`` column is the fraction of the room a system took
away: 1.00 is down to chance, 0.00 is no change, and a negative number means
the anonymized copy is more disclosive than the original. The ``room`` column
is what there was to remove, and a row whose room is small carries no
conclusion however large its ratio looks.

Usage:
    python scripts/report_criteria_split.py \\
        --comparison results/comparison/cv27_full.json \\
        --trial-metrics results/comparison/trial_metrics.json \\
        --inference results/comparison/cv27_inference.json \\
        --base cv27 --output results/comparison/criteria_split.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from legal_eval.criteria import (
    CRITERIA,
    MIN_ROOM,
    criteria_split,
    read_criterion,
)

#: Metrics taken from the population sweep, which depend on N.
SWEEP_METRICS = ("singling_out", "linkability", "srd_idr", "lid_pdr")

#: Metrics taken from the trial-score report, which do not depend on N.
TRIAL_METRICS = ("eer", "cllr_min", "gomez_barrero_dsys", "zebra_population")

#: The CV27 subsets, and how much evidence per speaker each gives the
#: attacker. Reporting both is the point: an inference result that holds with
#: six utterances per speaker is a different claim from one that needs twenty.
INFERENCE_SUBSETS = ("enrolls", "trials_mixed")

#: Printed when a criterion has no reading at all, which is the common case in
#: the literature and must not look like a zero.
NOT_MEASURED = "not measured"


def _sweep_row(system: dict, n: int) -> dict | None:
    """The sweep row for one population size, or None if it was not run."""
    for row in system.get("by_n", []):
        if row.get("n") == n:
            return row
    return None


def _available_counts(comparison: dict, base: str) -> list[int]:
    """Population sizes present for every system of this corpus."""
    sets = []
    for name, body in comparison["systems"].items():
        if not name.startswith(base + " "):
            continue
        sets.append({r["n"] for r in body.get("by_n", [])
                     if isinstance(r.get("n"), int)})
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common)


def _label(name: str, base: str) -> str:
    """The system part of a comparison key such as ``cv27 mcadams``."""
    return name[len(base) + 1:]


def collect(
    comparison: dict,
    trial: dict,
    inference: dict,
    base: str,
    n: int,
) -> dict[str, dict]:
    """Build every reading for every anonymized system of one corpus.

    Args:
        comparison: Parsed population sweep, with a ``systems`` mapping.
        trial: Parsed trial-score metrics, keyed the same way.
        inference: Parsed attribute inference, keyed by embedding dataset.
        base: Corpus prefix, for example ``cv27``.
        n: Population size to read the sweep metrics at.

    Returns:
        System label to a mapping with ``readings`` and ``summary``. The
        original is excluded, since it is the reference every row is against.

    Raises:
        SystemExit: when the corpus has no original to compare against, which
            would leave nothing to measure room with.
    """
    original_key = f"{base} original"
    if original_key not in comparison["systems"]:
        raise SystemExit(f"no {original_key!r} in the comparison, so there is "
                         f"no reference to measure room against")
    original = comparison["systems"][original_key]
    original_sweep = _sweep_row(original, n)
    if original_sweep is None:
        raise SystemExit(f"{original_key} was not evaluated at N={n}")

    out: dict[str, dict] = {}
    for name, body in sorted(comparison["systems"].items()):
        if not name.startswith(base + " ") or name == original_key:
            continue
        label = _label(name, base)
        row = _sweep_row(body, n)
        if row is None:
            print(f"  {label}: not evaluated at N={n}, skipping")
            continue

        readings, rows = [], []
        for metric in SWEEP_METRICS:
            if metric not in row or metric not in original_sweep:
                continue
            reading = read_criterion(metric, original_sweep[metric],
                                     row[metric], n=n)
            readings.append(reading)
            rows.append((metric, reading))

        # The trial-score metrics come from a different file and have no N.
        o_trial, s_trial = trial.get(original_key), trial.get(name)
        if o_trial and s_trial:
            for metric in TRIAL_METRICS:
                if metric not in o_trial or metric not in s_trial:
                    continue
                reading = read_criterion(metric, o_trial[metric],
                                         s_trial[metric])
                readings.append(reading)
                rows.append((metric, reading))

        # Inference, at each level of evidence per speaker.
        for subset in INFERENCE_SUBSETS:
            o_key = f"{base}_{subset}"
            s_key = f"{base}_{subset}_{label}"
            o_inf, s_inf = inference.get(o_key), inference.get(s_key)
            if not o_inf or not s_inf:
                continue
            reading = read_criterion("inference_advantage",
                                     o_inf["advantage"], s_inf["advantage"])
            readings.append(reading)
            per_speaker = o_inf.get("utterances_per_speaker")
            shown = (f"inference_advantage ({per_speaker:.0f} utt/spk)"
                     if per_speaker else f"inference_advantage ({subset})")
            rows.append((shown, reading))

        out[label] = {
            "rows": rows,
            "readings": readings,
            "summary": criteria_split(readings),
        }
    return out


def print_system(label: str, body: dict) -> None:
    """Print one system's readings and its summary across the criteria."""
    print(f"\n  {label}")
    print(f"    {'metric':<38} {'criterion':<14} {'value':>8} {'chance':>8} "
          f"{'room':>8} {'removed':>9}")
    print("    " + "-" * 90)
    for shown, r in body["rows"]:
        if not r.reliable:
            mark = "   room too small to read"
        elif r.beyond_chance:
            mark = "   past chance, explain rather than score"
        else:
            mark = ""
        removed = "  past" if r.beyond_chance else f"{r.reduction:>8.1%}"
        print(f"    {shown:<38} {r.criterion:<14} {r.value:>8.4f} "
              f"{r.chance:>8.4f} {r.room:>8.4f} {removed}{mark}")
    summary = body["summary"]
    print(f"    {'':<38} {'':<14} {'':>8} {'':>8} {'':>8}")
    for criterion in CRITERIA:
        median = summary["median_reduction"][criterion]
        count = summary["readings_per_criterion"][criterion]
        if median is None:
            print(f"    {criterion:<38} {NOT_MEASURED}")
        else:
            print(f"    {criterion:<38} median of {count} reading(s): "
                  f"{median:>7.1%} of the room removed")
    spread = summary["spread"]
    if spread is not None:
        print(f"    {'spread across criteria':<38} {spread:>7.1%}")


def print_matrix(collected: dict[str, dict]) -> None:
    """One row per system, one column per criterion, for reading at a glance."""
    print(f"\n{'system':<46} " + " ".join(f"{c:>16}" for c in CRITERIA)
          + f" {'spread':>8}")
    print("-" * (46 + 17 * len(CRITERIA) + 9))
    for label, body in sorted(collected.items()):
        summary = body["summary"]
        cells = []
        for criterion in CRITERIA:
            median = summary["median_reduction"][criterion]
            cells.append(f"{'n/a':>16}" if median is None
                         else f"{median:>15.1%} ")
        spread = summary["spread"]
        tail = f"{'n/a':>8}" if spread is None else f"{spread:>7.1%} "
        print(f"{label:<46} " + " ".join(cells) + f" {tail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--trial-metrics", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--base", default="cv27",
                        help="corpus prefix of the comparison keys")
    parser.add_argument("--counts", default=None,
                        help="comma-separated population sizes, or omit for "
                             "the usable window and the full population")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    trial = json.loads(args.trial_metrics.read_text(encoding="utf-8"))
    inference = json.loads(args.inference.read_text(encoding="utf-8"))

    available = _available_counts(comparison, args.base)
    if not available:
        print(f"no system of corpus {args.base!r} in {args.comparison}",
              file=sys.stderr)
        return 1
    if args.counts:
        counts = [int(c) for c in args.counts.split(",")]
        missing = [c for c in counts if c not in available]
        if missing:
            print(f"not evaluated at N={missing}, available: {available}",
                  file=sys.stderr)
            return 1
    else:
        # The largest population is where a controller would read, and a
        # smaller one is where Singling Out still has room. Both, so the
        # collapse is visible rather than inferred.
        window = [c for c in available if c <= 250]
        counts = sorted({window[-1] if window else available[0], available[-1]})

    out: dict[str, dict] = {}
    for n in counts:
        print(f"\n=== {args.base}, N={n}, against the original recordings ===")
        collected = collect(comparison, trial, inference, args.base, n)
        if not collected:
            print("  no anonymized system to compare")
            continue
        for label, body in sorted(collected.items()):
            print_system(label, body)
        print_matrix(collected)
        out[str(n)] = {
            label: {
                "summary": body["summary"],
                "readings": [
                    {"metric": shown, "criterion": r.criterion,
                     "value": r.value, "chance": r.chance, "room": r.room,
                     "reduction": r.reduction, "reliable": r.reliable,
                     "beyond_chance": r.beyond_chance}
                    for shown, r in body["rows"]
                ],
            }
            for label, body in collected.items()
        }

    print(f"\nA room below {MIN_ROOM} is not reported as a finding: the ratio "
          f"divides by almost")
    print("nothing. Where a criterion reads 'not measured', nothing was "
          "measured, which is")
    print("a stronger statement than a reduction of zero.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
