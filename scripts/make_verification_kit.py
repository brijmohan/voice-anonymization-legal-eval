#!/usr/bin/env python3
"""Build a self-contained kit for checking our LID and SRD implementations.

The point of the kit is to make verification a short reading and running task
rather than a reimplementation. It holds one score matrix, the true identity per
trial, and the values our implementation produces, so another group can run
their own implementation on the same input and diff the numbers.

The matrix is synthetic by default, so the kit can be shared without any data
licence attached. Pass ``--from-vpc`` to build it from a VoicePrivacy run
instead, which is what makes the comparison concrete but carries that corpus's
terms.

Usage::

    python scripts/make_verification_kit.py --output-dir verification_kit
"""

from __future__ import annotations

import argparse
import json
import sys
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

COUNTS = (2, 5, 10, 25, 50, 100)
SLOPES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
N_DRAWS = 10


def synthetic_matrix(n_enroll=100, n_trials=2000, margin=1.1, seed=20261009):
    """A score matrix with a known true identity per trial.

    Deliberately not drawn from a corpus, so the kit carries no data licence.
    The margin is chosen to put the metrics in their interesting range rather
        than against the ceiling.
    """
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n_enroll, n_trials))
    true_rows = rng.integers(0, n_enroll, size=n_trials)
    scores[true_rows, np.arange(n_trials)] += margin
    return scores, true_rows


def vpc_matrix(results_dir: Path, scenario: str = "aa"):
    from legal_eval.embeddings import build_speaker_embeddings, build_test_embeddings
    from legal_eval.scoring import cosine_score_matrix
    from legal_eval.vpc import find_embedding_dirs, load_vpc_embeddings, pair_datasets

    for pair in pair_datasets(find_embedding_dirs(results_dir)):
        if pair.scenario != scenario:
            continue
        enroll = load_vpc_embeddings(pair.enroll_dir)
        test = load_vpc_embeddings(pair.trial_dir)
        shared = sorted(set(enroll.speakers) & set(test.speakers))
        if len(shared) < 10:
            continue
        e = build_speaker_embeddings(
            {s: u for s, u in enroll.spk2utt.items() if s in shared}, enroll.embeddings
        )
        t, order = build_test_embeddings(
            {s: u for s, u in test.spk2utt.items() if s in shared},
            test.embeddings, 1, "first",
        )
        matrix = cosine_score_matrix(e, t)
        index = matrix.enroll_index
        rows = np.array([index[s] for s in matrix.test_speakers])
        return matrix.scores, rows, f"{pair.base}_{pair.scenario}{pair.system}"
    raise SystemExit(f"no {scenario} pair with enough shared speakers")


def expected_values(scores, true_rows):
    """Everything another implementation should be able to match."""
    mated, nonmated = calibration_samples(scores, true_rows)
    separable = is_separable(mated, nonmated)
    pool = nonmated
    if pool.size > 200_000:
        pool = np.random.default_rng(0).choice(pool, 200_000, replace=False)
    slope, intercept = fit_calibration(mated, pool)

    out = {
        "calibration": {
            "fitted_slope": slope,
            "fitted_intercept": intercept,
            "separable": separable,
            "note": (
                "The softmax is shift invariant, so the intercept and any prior "
                "ratio cancel. Only the slope reaches the reported bits. If your "
                "implementation disagrees, this is the first place to look."
            ),
        },
        "srd": [],
        "lid": [],
    }

    for n in COUNTS:
        if n > scores.shape[0]:
            continue
        runs = [
            similarity_rank_disclosure(scores, true_rows, n,
                                       rng=np.random.default_rng(d))
            for d in range(N_DRAWS)
        ]
        out["srd"].append({
            "n": n,
            "mean_d_bits": float(np.mean([r.mean for r in runs])),
            "max_d_bits": float(np.mean([r.maximum for r in runs])),
            "identification_rate": float(
                np.mean([r.identification_rate for r in runs])
            ),
            "log2_n": float(np.log2(n)),
        })
        for w in SLOPES:
            runs = [
                local_information_disclosure(scores, true_rows, n, slope=w,
                                             rng=np.random.default_rng(d))
                for d in range(N_DRAWS)
            ]
            out["lid"].append({
                "n": n,
                "slope": w,
                "alid_bits": float(np.mean([r.mean for r in runs])),
                "lid_max_bits": float(np.mean([r.maximum for r in runs])),
                "pdr": float(np.mean([r.positive_rate for r in runs])),
                "log2_n": float(np.log2(n)),
            })
    return out


README = """# Verification kit: LID and SRD

One score matrix, the true identity per trial, and the numbers our
implementation produces. The purpose is to let you check our reading of your
metric against your own implementation without reimplementing anything.

## Files

- `scores.npz`: `scores`, shape (enrollment speakers, trials), cosine
  similarities. And `true_rows`, length (trials), the row index of the true
  identity for each trial column.
- `expected.json`: what our implementation reports.
- `reference.py`: our implementation, standalone, numpy only, about 300 lines.

## The closed-set draw

Every metric here is defined on a closed set of `N` candidates per trial. We
draw that set per trial as the true identity plus `N-1` others sampled without
replacement from the remaining enrollment speakers, seeded, and we average over
10 draws. `expected.json` reports the mean over draws. If your implementation
forms the closed set differently, that difference alone will move the numbers,
so it is worth comparing before anything else.

## To check

```python
import numpy as np, json
d = np.load("scores.npz")
scores, true_rows = d["scores"], d["true_rows"]
expected = json.load(open("expected.json"))
# run your implementation on (scores, true_rows) for each n in expected["srd"]
```

## The three things we would most like checked

1. **SRD.** We read the disclosure at rank `j` as `log2(N * gamma_j)` over the
   empirical rank histogram `gamma`, with `MeanD` the trial-weighted mean,
   `MaxD` the maximum over observed ranks, and `IdR` the rank-1 share. Is that
   right?

2. **LID's calibration.** We z-normalise each trial's scores across the `N`
   candidates, then apply `LLR = w * s_hat + b`, then a softmax, then report
   `log2(N * p_true)`. Because a softmax is shift invariant, `b` and the
   prior-ratio term cancel exactly, leaving `w` as an inverse temperature. Is
   that the intended reading?

3. **What `w` should be.** This is the question we cannot answer from the paper
   and it matters more than the rest. Which development set is the Platt mapping
   fit on, and is it fit per system or once and shared? On our VoicePrivacy data
   one condition is perfectly separable, so the logistic fit has no finite
   maximiser and the slope diverges; ours stopped near 200, at which point ALID
   equals `log2 N` exactly. `expected.json` reports ALID across a grid of slopes
   precisely so that this is separable from the rest of the comparison.

## If we have it wrong

Then our comparison is wrong and we would like to know before it goes further
than this kit. Reply with a correction and we will fix the implementation, rerun
and re-send.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("verification_kit"))
    parser.add_argument("--from-vpc", type=Path, default=None,
                        help="build from a VoicePrivacy results directory instead")
    parser.add_argument("--scenario", default="aa")
    args = parser.parse_args()

    if args.from_vpc:
        scores, true_rows, tag = vpc_matrix(args.from_vpc, args.scenario)
    else:
        scores, true_rows = synthetic_matrix()
        tag = "synthetic"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "scores.npz",
                        scores=scores.astype(np.float64), true_rows=true_rows)

    payload = expected_values(scores, true_rows)
    payload["source"] = tag
    payload["shape"] = {"enroll_speakers": int(scores.shape[0]),
                        "trials": int(scores.shape[1])}
    payload["n_draws"] = N_DRAWS
    (args.output_dir / "expected.json").write_text(json.dumps(payload, indent=2))
    (args.output_dir / "README.md").write_text(README)

    reference = Path(__file__).resolve().parents[1] / "legal_eval" / "comparison.py"
    (args.output_dir / "reference.py").write_text(reference.read_text())

    print(f"kit written to {args.output_dir}  (source: {tag}, "
          f"{scores.shape[0]} speakers, {scores.shape[1]} trials)")
    print(f"  separable: {payload['calibration']['separable']}, "
          f"fitted slope {payload['calibration']['fitted_slope']:.3f}")
    for row in payload["srd"]:
        print(f"  N={row['n']:>4}  SRD MeanD {row['mean_d_bits']:.4f}  "
              f"IdR {row['identification_rate']:.4f}  "
              f"ceiling log2 N {row['log2_n']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
