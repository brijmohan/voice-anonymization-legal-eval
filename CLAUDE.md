# Working on this repository

Reference implementation of the Singling Out and Linkability metrics from
Vauquier et al., Interspeech 2025. Read this before changing anything.

## What this project is

A published paper's numbers depend on this code. People will cite results
produced by it, and some of those results are meant to support claims about
whether data is legally anonymous. **The public contract is the numbers the
metrics produce**, not only the Python API.

That single fact drives most of the rules below.

## Invariants

**Anything that moves a computed value is a breaking change.** It needs a major
version bump, a `CHANGELOG.md` entry, and an entry in
[`docs/differences.md`](docs/differences.md) so a reader comparing against the
paper knows which version produced which number. Things that move values, and
are easy to change by accident:

- the order of averaging and L2 normalisation, and whether a single embedding is
  normalised again
- the rank at which the Singling Out threshold is placed, and how ties are
  broken in Linkability (`argmax` semantics mean ties favour the true speaker,
  so only *strictly* higher scores count as beaters)
- RNG seeding, and whether repeated runs are genuinely independent
- which utterances form a conversation at each conversation length

**Deviations from the original experiment code are deliberate and documented.**
Three defects in the internal scripts are corrected here, and each one is
reachable in its original form behind a flag where it affects published results.
Do not silently "fix" something else. Document it.

**The core must work with numpy alone.** `scipy`, `h5py`, `torch` and
`matplotlib` are imported at their point of use, behind an error message that
names the extra to install and the alternative that avoids it. Do not move any
of them to a module-level import.

**The published results live inside the package**, at
`legal_eval/data/paper_results/`. They were outside it once, and
`load_paper_results()` raised `FileNotFoundError` for every pip-installed user
because the path resolved to a directory that only exists in a checkout. Keep
them in the package and keep them in `package-data`.

## Tests

A test that asserts the code returns what it currently returns is not a test. It
locks in whatever the behaviour happens to be, including the bugs.

Pin behaviour against something independent:

- **Theoretical baselines.** An attacker whose predicate carries no information
  singles out at `exp(-1)`. Uninformative scores give Linkability of `1/N'`.
  Equal-variance Gaussians separated by `d` have EER `Phi(-d/2)`.
- **Independent oracles.** The fast hypergeometric Linkability estimator is
  checked against a literal candidate-set-sampling implementation. ROCCH-EER is
  checked against a brute-force search over randomised decision rules.
- **Qualitative findings from the paper.** Both legal metrics rise with
  conversation length while the EER stays nearly flat.

`examples/02_reproduce_paper_figures.py` checks the shipped data against values
quoted in the paper's text, and is a regression test on the data rather than the
code. `examples/03_verify_against_score_matrices.py` recomputes 2,640 published
points from the archived score matrices.

## Writing

**Never use em dashes or en dashes**, in code, comments, documentation or commit
messages. Rewrite the sentence: a full stop, a colon, a semicolon, parentheses,
or a plain conjunction. A spaced hyphen is the same habit in disguise and is not
a substitute. Check with:

```sh
git ls-files -z | xargs -0 grep -nP '[\x{2014}\x{2013}]'
```

Docstrings are Google style and say what a number *means*, not what type it is.
`Returns: The ROCCH-EER, in [0, 0.5].` is useful. `Returns: float` is not.

Comments explain why. If a line encodes a decision from the paper, or a
deliberate departure from the original code, the comment says which.

## Commands

```sh
pip install -e ".[dev]"
pytest -q
ruff check legal_eval tests examples scripts
python examples/02_reproduce_paper_figures.py    # paper values still reproduce
legal-eval demo --output-dir /tmp/demo           # whole pipeline, synthetic data
```

## Scope

In scope: the metrics, the evaluation framework, I/O, figures, and adapters that
feed embeddings in, such as `legal_eval/vpc.py`.

Out of scope: anonymization systems and embedding extractors. Those are external
and already maintained elsewhere. See
[`docs/reproduction.md`](docs/reproduction.md).

## Naming hazard

The VoicePrivacy Challenge repository already ships a `linkability.py` computing
the Gomez-Barrero `D_sys`, a different quantity from this project's Linkability.
Anything VPC-facing must disambiguate the two or it will cause a bad comparison.
