# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check legal_eval tests examples
```

## What this repository is for

It implements the metrics from one paper. The bar for changes is whether they
make the evaluation more correct, more reproducible, or easier to apply to a new
system — not whether they add features.

Out of scope: anonymization systems, embedding extractors, and model training.
Those belong upstream, in the Voice Privacy Challenge and Sidekit repositories.
See [`docs/reproduction.md`](docs/reproduction.md).

## Changing a metric

Anything that changes a computed number needs:

1. A test pinning the new behaviour, ideally against an independent oracle or a
   theoretical baseline rather than against the current output.
2. An entry in [`docs/differences.md`](docs/differences.md) if it diverges from
   the original experiment code, with the original behaviour reachable by a flag
   when it affects published results.
3. A note in [`CHANGELOG.md`](CHANGELOG.md).

The existing tests are the model: Singling Out is pinned to the `exp(-1)` PSO
baseline, Linkability to a literal sampling implementation, ROCCH-EER to a
brute-force oracle. A test that only asserts the code returns what it currently
returns is not much of a test.

## Style

- Google-style docstrings on anything public, stating what a number *means*, not
  just its type.
- Comments explain why, not what. If a line encodes a decision from the paper or
  a deliberate departure from it, say so there.
- `ruff check` must pass. Line length 100.
- Keep the core dependency set to NumPy and SciPy; anything else goes in an
  optional extra.

## Reporting a discrepancy

If a number here disagrees with the paper, that is worth an issue even without a
fix. Please include the metric, `L`, the population size, the attacker, and what
you expected. One such discrepancy is already known and documented: the number of
enrollment utterances averaged for Singling Out, where the paper says 30 and the
final internal script used 10.
