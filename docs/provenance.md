# Provenance

Where the implementation comes from, and how it was checked.

## Source of the algorithms

The paper's experiments were run from a private Nijta GitLab repository, a fork
of [Sidekit](https://git-lium.univ-lemans.fr/speaker/sidekit) with the evaluation
scripts added. The relevant scripts:

| Script | Branch | What it produced |
|---|---|---|
| `bin/kaldi/cnil_plot1.py` | `master` | Linkability, `L = 1` |
| `bin/kaldi/cnil_plot2.py` | `master` | Linkability, `L ∈ {3, 10, 30}` |
| `bin/kaldi/cnil_plot3.py` | `master` | Linkability for the worst-case speakers (CNIL report only) |
| `PSO/compute_iso_for_plot2.py` | `pso` | Singling Out — the final, most evolved version |
| `PSO/compute_isolation_probabilities{,_parallel}.py` | `pso` | Earlier Singling Out versions |
| `bin/kaldi/compute_metrics.py` | `master` | ROCCH-EER for a fixed trial list |

Two traps for anyone reading that history:

1. **`cnil_plot1.py` on the `pso` branch is an older file than the one on
   `master`.** The `pso` version computes the Gomez-Barrero `Dsys` linkability,
   which the paper explicitly disavows in a footnote. The `master` version is the
   one that matches the paper.
2. **The Singling Out code lives only on the `pso` branch**, which also *deletes*
   `cnil_plot2.py` and `cnil_plot3.py`. Neither branch alone holds the whole
   paper.

This package was written from those scripts' algorithms, not copied from them:
Sidekit is LGPL and the BOSARIS code it carries is non-commercial-only, neither
of which is compatible with this repository's MIT licence.

## Shipped reference data

`data/paper_results/linkability_original_L{1,3,10,30}.json` are the original
experiment's outputs for the **Original** (non-anonymized) condition: 220
population sizes from 20 to 21,920, five runs each. They come from
`cnil_linkability/plot1_scores_step100.json` and
`cnil_linkability_plot2/plot2_scores_step100.json`, converted to this package's
schema with no change to the numbers.

`examples/02_reproduce_paper_figures.py` checks them against the values quoted in
the paper's Section 5. All four reproduce:

| Claim in the paper | Shipped data |
|---|---|
| `L=1`, 20 speakers: 82% | 0.822 |
| `L=1`, 10,000 speakers: 35% | 0.349 |
| `L=3`, 20 speakers: 94–95% | 0.950 |
| `L=30`, 20 speakers: 94–95% | 0.944 |

The tidy CSVs covering all nine panels of the paper's figure come from the
experiment archive, which also holds the 20 cosine score matrices (832 MB each),
the per-attacker result JSONs, and the raw PSO isolation outcomes. None of that
was ever committed to version control.

## How correctness was established

### Against the original score matrices

The strongest check: the cosine score matrices the paper's numbers came from
were recovered from the experiment archive, and every published Linkability
point was recomputed with this implementation.

| attacker | L | points | mean abs. diff | worst diff |
|---|---|---|---|---|
| original | 1 / 3 / 30 | 220 each | 0.0011 / 0.0009 / 0.0006 | 0.0049 / 0.0030 / 0.0034 |
| informed | 1 / 3 / 30 | 220 each | 0.0010 / 0.0011 / 0.0008 | 0.0051 / 0.0037 / 0.0029 |
| semi-informed | 1 / 3 / 30 | 220 each | 0.0003 / 0.0007 / 0.0008 | 0.0038 / 0.0039 / 0.0040 |
| ignorant | 1 / 3 / 30 | 220 each | 0.0002 / 0.0004 / 0.0006 | 0.0015 / 0.0023 / 0.0026 |

2,640 points, worst disagreement 0.0051, correlation above 0.9987 everywhere.
The residual is Monte Carlo noise from averaging five runs, and shrinks under the
`exact` estimator. Each condition's full 220-point sweep takes about 0.6 seconds.

Reproduce with `python examples/03_verify_against_score_matrices.py --root <dir>`.

This also confirms the attacker-to-experiment mapping, which the directory names
do not make obvious:

| Attacker | Experiment directory suffix |
|---|---|
| Original | `cnil_linkability`, `cnil_linkability_plot2` |
| Informed | `_plot1_anon`, `_plot2_anon` |
| Semi-Informed | `_ATTACKED_BY_am_nsf_dense_random__CNIL202310` |
| Ignorant | `_ATTACKED_BY_IGNORANT__CNIL202310` |

`am_nsf` is the acoustic model plus neural source-filter vocoder of VPC 2022
B1.a, which is what the paper's Semi-Informed attacker trains on. Note that an
`_ATTACKED_BY_IGNORANT_v2_` directory also exists; the published curves come
from the one without `v2`.

### Against oracles and baselines

Singling Out and the EER have no surviving per-point reference, so they are
pinned by properties and by independent oracles:

- **Singling Out.** An attacker whose predicate carries no information about the
  data isolates at `exp(-1)`. This is the PSO baseline the entire metric is read
  against, and it falls out of the calibration rule only if that rule is right.
  Separately, the vectorised implementation is checked outcome-by-outcome against
  a naive oracle that re-derives the threshold and the isolation decision from
  the definitions in plain Python loops.
- **Linkability.** The fast hypergeometric estimator is checked against a literal
  candidate-set-sampling implementation, in mean and in standard deviation, and
  against the closed form. Uninformative scores give the `1/N'` chance level.
- **ROCCH-EER.** Checked against a brute-force search over randomised decision
  rules on small inputs, and against the analytic EER of equal-variance Gaussians
  on large ones. The hull is verified convex, monotone, and spanning.
- **Qualitative findings.** The paper's two central claims are asserted as tests:
  both legal metrics rise with conversation length, and the EER stays nearly flat
  across population sizes.

## Versions

- `1.0.0` — a fabricated implementation that did not match the paper. Withdrawn.
- `2.0.0` — this implementation. See [CHANGELOG.md](../CHANGELOG.md).
