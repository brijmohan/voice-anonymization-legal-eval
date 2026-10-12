#!/usr/bin/env python3
"""Check `legal_eval.trial_metrics` against the authors' own reference code.

Cllr, the Gomez-Barrero linkability and ZEBRA all have reference
implementations that the VoicePrivacy Challenge installs, so unlike SRD and
LID these do not have to be taken from the papers alone. This script runs both
implementations on the same scores and reports the worst disagreement.

This matters more than a usual port check. The point of adding these metrics is
to compare them against ours in one artifact, and a comparison is worthless if
our version of somebody else's metric is not their version of it. If anything
here disagrees beyond floating-point noise, our implementation is wrong until
shown otherwise, not theirs.

Usage, on a machine with the challenge checked out:

    python scripts/verify_trial_metrics.py \\
        --reference ~/datasets/vpc_legal_bench/Voice-Privacy-Challenge-2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from legal_eval.trial_metrics import (
    cllr,
    gomez_barrero_linkability,
    min_cllr,
    optimal_llr,
    pav,
    zebra,
)

#: Agreement we require. These are different code paths over the same
#: arithmetic, so anything above float noise is a real difference.
TOLERANCE = 1e-9


def load_reference(root: Path):
    """Import the challenge's metric modules, or explain what is missing."""
    asv = root / "evaluation" / "privacy" / "asv"
    if not asv.is_dir():
        raise SystemExit(
            f"no evaluation/privacy/asv under {root}. Point --reference at a "
            f"VoicePrivacy Challenge checkout."
        )
    sys.path.insert(0, str(root))
    try:
        from evaluation.privacy.asv.metrics import helpers as ref_helpers
        from evaluation.privacy.asv.metrics import zebra as ref_zebra
        from evaluation.privacy.asv.metrics.cllr import cllr as ref_cllr
        from evaluation.privacy.asv.metrics.cllr import min_cllr as ref_min_cllr
        from evaluation.privacy.asv.metrics.linkability import (
            linkability as ref_linkability,
        )
    except ImportError as exc:
        raise SystemExit(
            f"could not import the reference metrics: {exc}. The challenge's "
            f"evaluation package pulls in heavy dependencies; run this with its "
            f"own interpreter."
        ) from None
    return ref_helpers, ref_zebra, ref_cllr, ref_min_cllr, ref_linkability


def cases(seed: int = 0, n: int = 30):
    """Score sets spanning separable, overlapping, tied and skewed shapes."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        n_mated = int(rng.integers(40, 400))
        n_non = int(rng.integers(200, 3000))
        if i % 5 == 0:                      # heavy ties
            mated = rng.integers(0, 6, n_mated).astype(float)
            non = rng.integers(0, 6, n_non).astype(float)
        elif i % 5 == 1:                    # well separated
            mated = rng.normal(4.0, 1.0, n_mated)
            non = rng.normal(0.0, 1.0, n_non)
        elif i % 5 == 2:                    # indistinguishable
            mated = rng.normal(0.0, 1.0, n_mated)
            non = rng.normal(0.0, 1.0, n_non)
        elif i % 5 == 3:                    # cosine-like, bounded
            mated = rng.uniform(0.2, 1.0, n_mated)
            non = rng.uniform(-0.2, 0.8, n_non)
        else:
            mated = rng.normal(rng.uniform(0.5, 3), rng.uniform(0.5, 2), n_mated)
            non = rng.normal(0.0, 1.0, n_non)
        yield mated, non


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True,
                        help="VoicePrivacy Challenge checkout")
    parser.add_argument("--cases", type=int, default=30)
    args = parser.parse_args(argv)

    ref_helpers, ref_zebra, ref_cllr, ref_min_cllr, ref_linkability = load_reference(
        args.reference
    )

    worst = dict.fromkeys(("pav", "optimal_llr", "cllr", "min_cllr", "dsys", "zebra_population", "zebra_worst"), 0.0)
    tags_agree = 0
    total = 0

    for mated, non in cases(n=args.cases):
        total += 1

        # PAV, on the ideal posterior vector the calibration actually uses.
        ideal = np.concatenate([np.zeros(non.size), np.ones(mated.size)])
        ideal = ideal[np.argsort(np.concatenate([non, mated]), kind="mergesort")]
        ours, _, _ = pav(ideal)
        theirs, _, _ = ref_helpers.pavx(ideal.copy())
        worst["pav"] = max(worst["pav"], float(np.abs(ours - theirs).max()))

        # Optimal LLRs, both with and without the Laplace padding.
        #
        # Without padding, separable scores give infinite LLRs, and inf - inf
        # is nan. A nan flowing into a max would make this check pass by being
        # silent, which is the failure mode this script exists to prevent. So
        # the infinities are compared as positions and only the finite entries
        # are compared numerically.
        for laplace in (False, True):
            o_t, o_n = optimal_llr(mated, non, laplace=laplace)
            r_t, r_n = ref_helpers.optimal_llr(mated, non, laplace=laplace)
            for ours_a, theirs_a in ((o_t, r_t), (o_n, r_n)):
                if not np.array_equal(np.isfinite(ours_a), np.isfinite(theirs_a)):
                    print("  infinite LLRs occur in different positions",
                          file=sys.stderr)
                    worst["optimal_llr"] = float("inf")
                    continue
                finite = np.isfinite(ours_a)
                if finite.any():
                    d = float(np.abs(ours_a[finite] - theirs_a[finite]).max())
                    worst["optimal_llr"] = max(worst["optimal_llr"], d)
                infinite = ~finite
                if infinite.any() and not np.array_equal(
                    np.sign(ours_a[infinite]), np.sign(theirs_a[infinite])
                ):
                    print("  infinite LLRs differ in sign", file=sys.stderr)
                    worst["optimal_llr"] = float("inf")

        # Cllr on calibrated LLRs, and the minimum over calibrations.
        t_llr, n_llr = optimal_llr(mated, non)
        worst["cllr"] = max(worst["cllr"],
                            abs(cllr(t_llr, n_llr) - ref_cllr(t_llr, n_llr)))
        worst["min_cllr"] = max(worst["min_cllr"],
                                abs(min_cllr(mated, non)
                                    - ref_min_cllr(mated, non)))

        # Gomez-Barrero D_sys.
        ref_dsys = ref_linkability(mated, non)[0]
        worst["dsys"] = max(worst["dsys"],
                            abs(gomez_barrero_linkability(mated, non) - ref_dsys))

        # ZEBRA: both figures and the categorical tag.
        ours_z = zebra(mated, non)
        zt, zn = ref_helpers.optimal_llr(mated, non, laplace=True)
        ref_pop = ref_zebra.dece(zt, zn)
        ref_worst = ref_zebra.max_abs_LLR(zt, zn)
        ref_tag = ref_zebra.category_tag_evidence(ref_worst)
        worst["zebra_population"] = max(worst["zebra_population"],
                                        abs(ours_z.population - ref_pop))
        worst["zebra_worst"] = max(worst["zebra_worst"],
                                   abs(ours_z.worst_case - ref_worst))
        tags_agree += int(ours_z.tag == ref_tag)

    print(f"{total} score sets, spanning separable, overlapping, tied, "
          f"bounded and skewed shapes\n")
    print(f"{'quantity':<20} {'worst |difference|':>20}  verdict")
    print("-" * 56)
    failed = []
    for name, value in worst.items():
        # np.nan <= tol is False, but be explicit: a non-finite disagreement
        # is a failure and must never read as agreement.
        ok = np.isfinite(value) and value <= TOLERANCE
        if not ok:
            failed.append(name)
        print(f"{name:<20} {value:>20.3e}  {'ok' if ok else 'DIFFERS'}")
    print(f"{'zebra tag':<20} {f'{tags_agree}/{total} agree':>20}  "
          f"{'ok' if tags_agree == total else 'DIFFERS'}")

    if failed or tags_agree != total:
        print(f"\n{len(failed)} quantity(ies) disagree beyond {TOLERANCE:g}. "
              f"Our implementation is wrong until shown otherwise.")
        return 1
    print(f"\nEvery quantity agrees to within {TOLERANCE:g}. These are the "
          f"authors' metrics, not our reading of them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
