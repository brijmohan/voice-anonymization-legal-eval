# Published results from the paper

Linkability for the **Original** (non-anonymized) condition, as produced by the
original experiment code.

| File | `L` | Points | Runs |
|---|---|---|---|
| `linkability_original_L1.json` | 1 | 220 | 5 |
| `linkability_original_L3.json` | 3 | 220 | 5 |
| `linkability_original_L10.json` | 10 | 220 | 5 |
| `linkability_original_L30.json` | 30 | 220 | 5 |

Each covers enrollment population sizes `N'` from 20 to 21,920 in steps of 100,
on Common Voice 11.0: set A (22,024 speakers) as enrollment, set B (4,949) as
test.

## Schema

```json
{
  "metric": "linkability",
  "conversation_length": 1,
  "metadata": { "attacker": "original", "n_runs": 5, ... },
  "values": { "20": [0.8232, 0.8232, 0.8228, 0.8173, 0.8216], ... }
}
```

Load with `legal_eval.sweeps.SweepResult.from_dict(read_results(path))`.

## Checking them

`python examples/02_reproduce_paper_figures.py` verifies these against the values
quoted in the paper's Section 5. All four reproduce: `L=1` gives 0.822 at `N'=20`
and 0.349 at `N'≈10,000` against the paper's "82%" and "35%"; `L=3` and `L=30`
give 0.950 and 0.944 at `N'=20` against "94-95%".

## Two caveats

**The error bars understate the true spread.** These were produced with a
run-to-run RNG bug: the five "independent" runs shared random state across
parallel workers. At `N'=20` in `L1` two runs are identical to 16 digits. The
means are unaffected. See [`docs/differences.md`](../../../docs/differences.md).

**The anonymized conditions are not here.** Results for the Informed,
Semi-Informed and Ignorant attackers, and all Singling Out and EER results, were
never committed to version control and exist only on the machines the experiments
ran on. Regenerating them needs the cosine score matrices; see
[`docs/reproduction.md`](../../../docs/reproduction.md).

## Provenance

Converted without changing any number from `cnil_linkability/plot1_scores_step100.json`
and `cnil_linkability_plot2/plot2_scores_step100.json` in Nijta's internal GitLab
repository at commit `3af1d02`.
