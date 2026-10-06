# Changelog

## 2.1.0

**Fix: VPC scenario labelling.** A VoicePrivacy run holds both original and
anonymized copies of every dataset, so pairing enrollment with trial sets yields
four genuinely different attacks, which VPC calls `oo`, `oa`, `ao` and `aa`. The
adapter emitted all of them labelled only by the trial dataset, so rows for the
same trial set appeared twice with different values and no way to tell which
enrollment produced them. Surfaced on the first real run against VPC 2026 B5
output, not by the test suite.

Rows now carry `base`, `scenario` and `enrollment`, `pair_datasets` returns a
`DatasetPair`, and the CLI groups by scenario rather than printing a dataset
directory name too long for its column. Two tests pin it, including that each
`(scenario, metric, L, speakers)` appears exactly once.

Anyone who ran `legal-eval vpc` on 2.0.0 against a VPC run should re-run: the
numbers were correct but not attributable to a condition.

## 2.0.0

Complete replacement of the implementation.

### The problem with 1.0.0

The previously published code did not implement the paper. It could not have
reproduced any result in it, and it should not have been used to evaluate
anything. Concretely:

- `anonymization.py` contained a `PlaceholderExtractor` and a
  `PlaceholderSynthesizer` that did no anonymization.
- `speaker_embedding.py` and `attack_models.py` defined a `SimplifiedECAPA` with
  randomly initialised weights that was never trained, so the "embeddings" it
  produced carried no speaker information.
- `SinglingOutMetric` calibrated its threshold on a flat, speaker-shuffled pool
  with a hardcoded 9th/10th rank, rather than on the selected speakers'
  calibration conversations with the rank set by the predicate speaker's own
  conversation count. It also never swept the population size `N`, which is the
  axis the metric is defined along.
- `LinkabilityMetric.compute_with_speaker_counts` drew `N` enrollment speakers at
  random and then matched test speaker `i` to the `i`-th *drawn* speaker. The
  true speaker was usually not in the candidate set at all, so the quantity
  computed was not linkability.

It has been removed rather than patched. This changelog entry stays so the
published record is clear.

### 2.0.0

**Metrics**, faithful to the paper and pinned by tests:

- Singling Out (PSO), with the `1/N` threshold calibration, cross-validation
  folds, and the `exp(-1)` trivial baseline verified to fall out of the
  calibration rule.
- Linkability as the closed-set identification rate the paper defines, *not* the
  Gomez-Barrero `Dsys` metric its footnote disavows.
- ROCCH-EER, implemented independently of the non-commercially-licensed BOSARIS
  toolkit and checked against a brute-force oracle.

**Framework**: population-size and conversation-length sweeps, cosine score
matrices as a reusable intermediate, Kaldi-style I/O, a CLI, and paper-style
figures with a colour-vision-validated palette.

**Performance**: Linkability's full sweep runs in seconds instead of hours, via a
hypergeometric draw that is equivalent in distribution to explicit candidate-set
sampling.

**Correctness fixes** relative to the internal scripts that produced the paper,
each documented in [`docs/differences.md`](docs/differences.md) with the original
behaviour reachable by a flag where it affects results:

- Parallel workers shared one RNG state, so the five "independent" runs were not
  independent.
- The Singling Out fold loop seeded the conversation split on speaker position
  only, so all ten folds reused the same split.
- `random.sample` was called on a `set`, which raises `TypeError` on Python 3.11
  and later.

**Data**: `legal_eval/data/paper_results/` ships the original Linkability results for the
Original condition at `L ∈ {1, 3, 10, 30}`, reproducing every Linkability value
quoted in the paper.

## 1.0.0

Withdrawn. See above.
