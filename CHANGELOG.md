# Changelog

## Unreleased

**New: `legal-eval preflight`, the hazards as checks.** Every failure recorded
in [`docs/operational-hazards.md`](docs/operational-hazards.md) produced a
plausible number rather than an error, and three of them were written down as
habits and then recurred anyway. `legal_eval.preflight` runs the subset that
can be judged from a run's own output and names the failure each check guards
against. Running it over this project's own results corrected the checks twice:
one threshold failed on correct data, and a second check conflated a legitimate
zero standard error at a metric's ceiling with a broken one.

**New: `legal_eval.audio_integrity`.** A system that emits silence instead of
speech is rewarded for it: the embedding sits at chance distance from every
enrolled speaker, and chance is the best privacy score attainable. The 2026
McAdams baseline did exactly this on one Common Voice utterance, producing a
file of the right length that was 99.93% exact zeros. No evaluation in the
field checks for it. `check_anonymized` judges whether output is speech at all,
with thresholds loose enough that heavy but legitimate processing passes.

**New: `legal_eval.vpc.unpaired_datasets`.** `pair_datasets` recognises only
VPC's naming, so a differently named embedding directory was discovered and
then silently dropped, which would have produced a comparison covering no
Common Voice at all with nothing failing.

**New: `scripts/check_scp_readers.py` and `scripts/merge_embedding_shards.py`.**
The first finds audio readers that cannot open a Kaldi extended filename, a
defect that appeared at four sites in one pipeline and twice survived a grep.
The second merges sharded embedding directories, which was needed because the
challenge extractor's own parallel path fails for this attacker; sharded output
is verified bit-identical to a single pass.

**New: [`docs/cv27-datasheet.md`](docs/cv27-datasheet.md).** Including the
figure to read before using the gender labels: gender is absent for 41% of
speakers and about 72% male among those who report it.


**New: `legal_eval.comparison`, for putting the recent privacy metrics on one
footing.** Four groups have proposed replacing the EER and none had been run
against the others on the same data, because published values come from
different corpora, attackers and splits. This module computes Local Information
Disclosure (Sterns, Drossos, Fernandes, Backstrom and Palamidessi,
arXiv:2607.06259) and Similarity Rank Disclosure
(Chandra et al., arXiv:2605.07291) alongside the metrics already here, so a
comparison can be run from one artifact.

The LID and SRD implementations are **our reading of the published
definitions**, not authoritative. They are small and commented so that their
authors can check them quickly rather than reimplementing. See
[`docs/metric-comparison.md`](docs/metric-comparison.md) for the findings, which
include: SRD's identification rate is the Linkability metric; LID's Platt
intercept and prior term cancel in the softmax, leaving the slope as the only
calibration parameter that reaches the result; the probability metrics fall with
population size while the bit metrics rise; and ALID moves by tens of bits, and
changes sign, when only the calibration slope changes.

No existing value moves. Nothing in this entry affects the published metrics.

**New: `scripts/check_no_dashes.py`, checked in CI.** The writing rule against
em and en dashes was documented as a `grep -nP` one-liner. BSD grep has no `-P`
and exits with a usage error that reads as a clean result, so the check failed
open on macOS and a dash reached `CONTRIBUTING.md` and survived several passes.
Fixed, and the check no longer depends on which grep is installed.

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
