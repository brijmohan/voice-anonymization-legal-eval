#!/usr/bin/env python3
"""Report each metric against its chance level and the attacker's real ceiling.

`docs/protocol.md` requires three numbers at every population: the chance
level, the anonymized system's value, and **the same attacker's value on the
original audio**. This produces that report from a comparison's JSON.

The third number is the one usually missing and it is the one that decides
whether a result means anything. Measured on Common Voice, the same attacker
scores 0.9131 Linkability on original audio at `N = 29` and 0.4515 at
`N = 9,779`. At the first population the metric is pinned near its bound, so no
system can be shown to differ from another. At the second the attacker already
fails on more than half the unprotected audio, so more than half of any privacy
measured there is the task being hard rather than the anonymization working.

So this script reports, per metric and population:

- **chance**, what an attacker who learned nothing obtains
- **original**, the same attacker on unprotected audio, its real ceiling
- **system**, the anonymized value
- **room**, ``original - chance``, the whole range available to a system
- **closed**, the fraction of that room the system actually closed

``closed`` is the quantity to compare across metrics, because it is the only
one normalised by what was achievable rather than by an unreachable bound.

Usage:
    python scripts/report_three_point.py results.json --original "cv27 original"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

#: Metrics whose chance level is a probability, with how to compute it from N.
PROBABILITY_CHANCE = {
    "singling_out": lambda n: math.exp(-1),
    "linkability": lambda n: 1.0 / n,
    "srd_idr": lambda n: 1.0 / n,
    "lid_pdr": lambda n: 1.0 / n,
}

#: Metrics in bits, which are zero at chance.
BIT_CHANCE = {m: (lambda n: 0.0) for m in
              ("srd_mean", "srd_max", "lid_mean", "lid_max")}

CHANCE = {**PROBABILITY_CHANCE, **BIT_CHANCE}

#: Reported in this order, probabilities first.
ORDER = ("singling_out", "linkability", "srd_idr", "srd_mean", "lid_mean")


def rows_by_n(body: dict) -> dict[int, dict]:
    """Index one system's per-population rows by N."""
    out = {}
    for row in body.get("by_n") or []:
        if isinstance(row, dict) and isinstance(row.get("n"), int):
            out[row["n"]] = row
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="comparison JSON")
    parser.add_argument("--original", required=True,
                        help="name of the unprotected system in that JSON")
    parser.add_argument("--metric", default="linkability",
                        choices=sorted(CHANCE),
                        help="which metric to tabulate")
    args = parser.parse_args(argv)

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"no such file: {args.report}", file=sys.stderr)
        return 2

    systems = report.get("systems") or {}
    if args.original not in systems:
        print(f"{args.original!r} is not in the report. Available:",
              file=sys.stderr)
        for name in sorted(systems):
            print(f"  {name}", file=sys.stderr)
        return 2

    baseline = rows_by_n(systems[args.original])
    chance_of = CHANCE[args.metric]
    metric = args.metric

    for name in sorted(systems):
        if name == args.original:
            continue
        rows = rows_by_n(systems[name])
        shared = sorted(set(rows) & set(baseline))
        if not shared:
            continue
        print(f"\n=== {name}, against {args.original}, metric {metric} ===")
        print(f"{'N':>6} {'chance':>8} {'original':>9} {'system':>8} "
              f"{'room':>8} {'closed':>8}  note")
        print("-" * 68)
        for n in shared:
            sys_v = rows[n].get(metric)
            orig_v = baseline[n].get(metric)
            if not isinstance(sys_v, (int, float)):
                continue
            if not isinstance(orig_v, (int, float)):
                continue
            ch = chance_of(n)
            room = orig_v - ch
            note = ""
            if room <= 0:
                closed_s = "n/a"
                note = "the original is already at chance here"
            else:
                closed_s = f"{(orig_v - sys_v) / room:>8.3f}"
                # Both failure modes, named rather than silently tolerated.
                if metric in PROBABILITY_CHANCE and orig_v > 0.90:
                    note = "original pinned near bound, cannot discriminate"
                elif metric in PROBABILITY_CHANCE and orig_v < 0.75:
                    note = "original already failing, privacy confounded with difficulty"
            print(f"{n:>6} {ch:>8.4f} {orig_v:>9.4f} {sys_v:>8.4f} "
                  f"{room:>8.4f} {closed_s:>8}  {note}")

    print("\n'closed' is the fraction of the available room the system closed, so it")
    print("is the figure comparable across metrics. A row with a note is a row where")
    print("the population, not the system, is deciding the number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
