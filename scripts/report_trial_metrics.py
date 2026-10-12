#!/usr/bin/env python3
"""Report the metrics that need no candidate population, for every system.

Cllr, the Gomez-Barrero linkability and ZEBRA are functionals of the mated and
non-mated score distributions. They need no sweep over population sizes, so
they cost a fraction of the full comparison: one score matrix per system and no
Singling Out draws at all.

That makes this worth having separately. The full comparison spends most of its
time on the population sweep, and these figures are available long before it
finishes.

Printed alongside the EER, because the EER belongs in the same family: all four
answer "how distinguishable are these two score distributions" rather than "can
the right speaker be found among N".

ZEBRA's second figure is the one to read last and worry about first. It is the
only quantity here about a single individual rather than an average, and an
average that looks acceptable can conceal one person about whom the evidence is
overwhelming.

Usage:
    python scripts/report_trial_metrics.py --results-dir exp/asv_ssl \\
        --output results/trial_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from legal_eval.embeddings import build_speaker_embeddings, build_test_embeddings
from legal_eval.metrics import rocch_eer
from legal_eval.scoring import cosine_score_matrix
from legal_eval.trial_metrics import (
    cllr,
    gomez_barrero_linkability,
    min_cllr,
    zebra,
)
from legal_eval.vpc import (
    find_embedding_dirs,
    load_vpc_embeddings,
    pair_datasets,
    suspicious_datasets,
)


def mated_and_nonmated(scores: np.ndarray, true_rows: np.ndarray):
    """Split a score matrix into same-speaker and different-speaker scores.

    Args:
        scores: ``(n_enroll, n_trials)``.
        true_rows: ``(n_trials,)`` enrollment row of each trial's true speaker.

    Returns:
        ``(mated, nonmated)`` as 1-D arrays.
    """
    columns = np.arange(scores.shape[1])
    mated = scores[true_rows, columns]
    mask = np.ones_like(scores, dtype=bool)
    mask[true_rows, columns] = False
    return mated, scores[mask]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="a VPC results directory, or its emb_xvect subtree")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the table as JSON")
    args = parser.parse_args(argv)

    discovered = find_embedding_dirs(args.results_dir)
    intermediates = suspicious_datasets(discovered)
    if intermediates:
        print(f"skipping {len(intermediates)} dataset(s) that look like "
              f"extraction intermediates rather than systems")
        discovered = {k: v for k, v in discovered.items() if k not in intermediates}

    out: dict[str, dict] = {}
    seen = set()
    for pair in pair_datasets(discovered):
        if pair.scenario not in ("oo", "aa"):
            continue
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
        name = f"{pair.base} {label}"

        enroll_embeddings = build_speaker_embeddings(
            {s: u for s, u in enroll.spk2utt.items() if s in shared},
            enroll.embeddings,
        )
        test_vectors, _ = build_test_embeddings(
            {s: u for s, u in test.spk2utt.items() if s in shared},
            test.embeddings, 1, "first",
        )
        matrix = cosine_score_matrix(enroll_embeddings, test_vectors)
        # The matrix does not carry which row holds each column's true
        # speaker, so derive it the way the comparison example does.
        index = matrix.enroll_index
        true_rows = np.array([index[s] for s in matrix.test_speakers])
        mated, nonmated = mated_and_nonmated(matrix.scores, true_rows)

        eer = rocch_eer(mated, nonmated)
        try:
            actual = cllr(mated, nonmated)
        except ValueError as exc:
            actual = float("nan")
            print(f"  {name}: actual Cllr unavailable, {exc}")
        minimum = min_cllr(mated, nonmated)
        try:
            dsys = gomez_barrero_linkability(mated, nonmated)
        except ValueError as exc:
            dsys = float("nan")
            print(f"  {name}: D_sys unavailable, {exc}")
        profile = zebra(mated, nonmated)

        out[name] = {
            "speakers": len(shared),
            "eer": float(eer),
            "cllr_actual": float(actual),
            "cllr_min": float(minimum),
            "gomez_barrero_dsys": float(dsys),
            "zebra_population": profile.population,
            "zebra_worst_case": profile.worst_case,
            "zebra_tag": profile.tag,
        }
        print(f"  computed {name} over {len(shared)} speakers", flush=True)

    if not out:
        print("no usable system found", file=sys.stderr)
        return 1

    print(f"\n{'system':<46} {'spk':>6} {'EER':>7} {'Cllr':>7} {'minC':>7} "
          f"{'D_sys':>7} {'ZEBRA pop':>10} {'worst':>7} {'tag':>4}")
    print("-" * 108)
    for name, row in sorted(out.items()):
        print(f"{name:<46} {row['speakers']:>6} {row['eer']:>7.4f} "
              f"{row['cllr_actual']:>7.4f} {row['cllr_min']:>7.4f} "
              f"{row['gomez_barrero_dsys']:>7.4f} {row['zebra_population']:>10.4f} "
              f"{row['zebra_worst_case']:>7.4f} {row['zebra_tag']:>4}")

    print("\nCllr: 1.000 is an attacker that learned nothing, and the gap "
          "between actual and")
    print("minimum is calibration loss, which is the attacker's to fix rather "
          "than a property")
    print("of the anonymization. D_sys and the ZEBRA figures are 0 for perfect "
          "privacy.")
    print("The ZEBRA worst case is the only figure about one individual rather "
          "than an average.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
