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

**The anonymized conditions were never committed.** The result files for the
Informed, Semi-Informed and Ignorant attackers, and all Singling Out and EER
results, existed only on the machines the experiments ran on. Regenerating those
rows requires the cosine score matrices; see [reproduction.md](reproduction.md).

## How correctness was established

Without the original x-vectors, agreement with the published curves can only be
checked for the Original Linkability condition. The rest is pinned by properties
and by independent oracles:

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
