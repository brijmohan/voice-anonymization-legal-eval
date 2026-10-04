# Differences from the original experiment code

The mathematics here matches the internal scripts that produced the paper. Three
reproducibility defects in those scripts are fixed, and a handful of conventions
are made explicit. Everything that changes a number is listed, with the original
behaviour still reachable where it matters.

## Fixed: random state shared across parallel workers

**Original.** The Linkability sweep ran its 5 "random draws" with
`joblib.Parallel(n_jobs=5)` over a function seeded only by a module-level
`random.seed(42)`. Every worker started from the same state, so runs were not
independent, and how much they differed depended on how joblib happened to
schedule and reuse workers. The published `plot1_scores_step100.json` shows the
symptom directly: at `N' = 20` two of the five "independent" runs are identical
to 16 digits.

```json
"20": [0.8231966053748232, 0.8231966053748232, 0.8227924833299657, ...]
```

**Here.** Every run draws from its own `numpy.random.Generator` seeded on
`(seed, speaker_count, run)`. Runs are genuinely independent and results do not
depend on worker scheduling, thread count, or evaluation order.

**Effect.** Means are unaffected. The reported standard deviation becomes a
meaningful estimate of run-to-run spread rather than an artefact of scheduling.
The error bars in the shipped `data/paper_results/` files are therefore
understated relative to what this code produces.

## Fixed: the fold loop reused a single data split

**Original.** In `PSO/compute_iso_for_plot2.py`, the per-speaker conversation
split was seeded on the speaker's position in the dictionary:

```python
for fold in range(n_conversations_per_speaker):
    X = joblib.Parallel(...)(
        joblib.delayed(_plot2_get_conversations_xvectors)(
            (s, ..., seed)
        )
        for seed, s in enumerate(dataset_speakers)   # <- no dependence on `fold`
    )
```

The seed does not depend on `fold`, so all 10 folds built the *same* test and
calibration conversations. The cross-validation the paper describes did not vary
the thing it was meant to vary. (Only the draw of the `N` test speakers differed
between folds, through the global RNG.)

**Here.** The split is seeded on `(seed, speaker_position, fold)`, so folds are
genuinely different. Pass `vary_folds=False`, or `--freeze-folds` on the command
line, to reproduce the original behaviour.

**Effect.** Singling Out means shift slightly and the spread widens, since it now
reflects variation over data splits as well as over speaker sets.

## Fixed: `random.sample` on a set

**Original.** `random.sample(all_enroll_idx - set([true_spk_idx]), n_extra_spks)`
passes a `set`. That has been deprecated since Python 3.9 and **raises
`TypeError` on Python 3.11 and later**, so the original scripts no longer run on
a current interpreter. Even where it did run, iteration order over a set of ints
is an implementation detail, so the draw was not reproducible across versions.

**Here.** Sampling is over explicit index ranges with a seeded generator.

## Made explicit: which utterances form a test conversation

The original used two different rules, in two scripts, for what became two rows
of the same figure:

- `cnil_plot1.py` (`L = 1`): one utterance drawn at random.
- `cnil_plot2.py` (`L ∈ {3, 10, 30}`): the **first** `L` utterances in `spk2utt`
  order, deterministically.

Neither is wrong, but the asymmetry is an accident of how the scripts grew. It is
preserved under `selection="reference"`, the default, so the published curves
reproduce. `selection="random"` applies random sampling at every `L` and is the
better choice for new experiments; `selection="first"` is deterministic
throughout.

## Documented discrepancy: enrollment utterances for Singling Out

The paper states that Singling Out enrollment embeddings average **30**
utterances per speaker:

> *"We randomly select 495 speakers from set B as enrollment speakers (S) and
> compute the corresponding averaged x-vectors from 30 utterances per speaker."*

The final internal script, `PSO/compute_iso_for_plot2.py`, used
`N_UTTERANCES_PER_SPEAKER = 10` for that average. An earlier version averaged all
available utterances, with a docstring saying 30.

This package follows the paper: `--n-enroll-utterances` defaults to 30. Set it to
10 to match the final script. **This is an open discrepancy between the paper and
the code that produced it**, and anyone reproducing the Singling Out row should
be aware that the enrollment averaging depth is not settled.

## Reimplemented: ROCCH-EER

The original called Sidekit's `bosaris.detplot.rocch` / `rocch2eer`, a
translation of the BOSARIS toolkit. BOSARIS is licensed by Agnitio for
**non-commercial use only**, which is incompatible with this repository's MIT
licence, so it is not vendored. `legal_eval/metrics/eer.py` is an independent
implementation via the ROC convex hull, pinned by `tests/test_eer.py` against a
brute-force oracle and against the analytic Gaussian EER.

Differences against BOSARIS should be at the level of floating-point noise, since
both compute the same quantity, but they have not been compared numerically
head-to-head — doing so would require accepting the BOSARIS licence.

## Reconstructed: the EER sweep

No script sweeping the EER over `N'` and `L` was ever committed; that row of the
paper's figure was produced ad hoc. `legal_eval.sweeps.eer_sweep` reconstructs it
from the description in the paper, reusing the same score matrices and candidate
sets as Linkability. It is the one part of the pipeline with no surviving
reference implementation to check against.

## Performance: hypergeometric sampling for Linkability

Explicit candidate-set construction is replaced by an equivalent hypergeometric
draw. The sampling distribution is identical — this is a change of algorithm, not
of definition — and the equivalence is asserted in
`tests/test_linkability.py`. See [metrics.md](metrics.md#why-the-fast-path-is-exact)
for the derivation. `linkability_naive` keeps the literal version for testing.

## Not reimplemented

**Anonymization and x-vector extraction.** The paper's B1 / B1.a systems and the
Sidekit ECAPA-TDNN extractors are external. Re-implementing them here would
produce a worse copy of code that already exists; see
[reproduction.md](reproduction.md) for the pinned upstream versions.

**The worst-case speaker analysis** (`cnil_plot3.py`), which ranks speakers by how
far their true score sits above the best impostor and reports Linkability for the
most exposed ones. It appears in the CNIL report, not in the Interspeech paper.
The score-matrix layout here supports it — a speaker may label several columns —
but it is not implemented.
