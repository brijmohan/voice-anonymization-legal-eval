# Choosing the subset thresholds

The paper defines its two Common Voice subsets by duration: enrollment is
speakers with at least 2 minutes of speech, test is speakers with a further
3 minutes on disjoint utterances. Rebuilding on a current release is a chance to
ask whether those are the right criteria. They are defensible, but three things
about them are worth changing, and one of them the paper itself works around.

## The duration threshold measures the wrong quantity

Both legal metrics are defined over **utterance counts**, not minutes.
Linkability at conversation length `L` needs `L` utterances per test speaker.
Singling Out needs `2L`, because the test and calibration conversations must be
disjoint.

A duration threshold does not guarantee either. Three minutes is thirty
six-second clips or three sixty-second ones, and only the first can support
`L = 30`. The paper runs into this directly and patches it afterwards:

> When `L = 30`, test speakers with less than `2 x L` utterances are excluded to
> ensure `K >= 2`, hence the maximum `N` is below 22,024.

So the population at `L = 30` is set by an utterance-count filter applied *after*
a duration filter that was supposed to select it. The two criteria disagree, and
the one that governs is implicit.

**Change: threshold on utterance count, derived from the largest `L` you intend
to sweep.** A test speaker needs `2 * L_max` utterances. State `L_max` up front
and the population is then stable across the whole sweep rather than shrinking
at the top end for a reason the reader has to hunt for in a footnote.

Keep a minimum duration as a secondary floor if you want to exclude speakers
whose clips are all under a second, but let the utterance count lead.

## Enrollment conflates two different roles

Enrollment speakers do two unrelated jobs.

A few of them are **targets**: the attacker holds their reference embedding and
tries to match it to test data. These need enough speech for a reliable
reference.

The overwhelming majority are **distractors**: they exist only to make the
search harder, and the metric asks whether any of them outranks the true
speaker. A distractor needs only to be a plausible competitor. Thirty seconds is
ample.

Requiring two minutes of every enrollment speaker therefore buys reference
quality the experiment does not use, and pays for it in population size. That is
a poor trade, because **population size is the axis the whole argument rests
on**. Our VPC runs measure Linkability pinned at 1.000 across every population
up to 29 speakers: with too few distractors the metric cannot discriminate at
all.

**Change: decouple the two thresholds.** Apply a high bar to speakers who also
serve as test speakers, and a low one to the distractor population. On the same
corpus this buys a substantially larger `N` for the same anonymization compute,
since distractors need few utterances each.

## A fixed threshold leaves the attacker uncontrolled

With a floor rather than a budget, enrollment speakers arrive with anywhere from
two minutes to several hours of speech. The attacker's reference is therefore
much stronger for some speakers than others, and that variation is baked into
the reported number with no way to separate it from the effect being measured.

This matters more than it sounds. Recent work finds that per-speaker
re-identification risk is not an intrinsic property of a speaker but emerges
from the interaction between attacker, system, and the amount of speech
available.[^dufour] If the amount of speech varies uncontrolled across the
enrollment set, some of the per-speaker spread being reported is an artefact of
the sampling rather than a property of the system.

**Change: fix the enrollment budget.** Give every enrollment speaker the same
number of utterances. The attacker is then uniform across the population and the
measurement has one less confound.

## The enrollment budget deserves to be swept, not fixed

The paper sweeps conversation length `L`, which is how much data the attacker
has about the *target*. It fixes the enrollment side, which is how much the
attacker has about the *reference*. Both are attacker capabilities and there is
no principled reason to sweep one and fix the other.

Sweeping both, even coarsely, turns a curve into a surface and answers a
question a deployer actually has: for a given amount of speech per enrollee, how
much target data does it take before the risk becomes unacceptable.

**Change: treat the enrollment budget as a second axis**, with a small grid such
as 1, 5 and 20 utterances per enrollment speaker.

## What this suggests concretely

| | Paper | Suggested |
|---|---|---|
| Test speaker criterion | at least 3 minutes | at least `2 * L_max` utterances |
| Enrollment criterion | at least 2 minutes | fixed budget of `E` utterances |
| Distractor criterion | same as enrollment | at least 1 utterance |
| Enrollment budget | implicit, varies per speaker | swept: 1, 5, 20 |
| Population at `L = 30` | shrinks, explained in a footnote | stable by construction |

`scripts/construct_common_voice_subsets.py` implements the duration thresholds
today, because they reproduce the paper's construction. The caps
`--max-enroll-utterances` and `--max-test-utterances` already give a fixed
budget. The remaining work is to make the test criterion count utterances rather
than minutes, and to allow a separate low threshold for distractors.

## What not to change

The **disjointness** of enrollment and test utterances is not negotiable. If an
utterance appears on both sides the metrics measure recording identity rather
than speaker identity, and every number is inflated. That one is right.

The **ratio** of thresholds also encodes something sensible: the test side needs
more speech than the enrollment side, because `L` can be large. Keeping the test
bar above the enrollment bar is worth preserving in whatever units.

[^dufour]: O. Dufour, P. Magron, M. Rouvier, E. Vincent, "A Large-Scale
    Per-Speaker Analysis of Re-identification Risk in Speech Anonymization",
    Interspeech 2026, arXiv:2606.07210.

## Related work: the community is already working on this

Our subset design was arrived at from first principles, which was a mistake in
sequence: several groups have published on exactly these questions and some of
it supersedes or sharpens what is above. Checked against arXiv metadata rather
than recalled.

### Evaluation protocol design for anonymization

**Champion, "Anonymizing Speech: Evaluating and Designing Speaker Anonymization
Techniques", arXiv:2308.04455 (2023).** A doctoral thesis whose subject is the
evaluation design itself, not a new anonymizer. This is the closest prior work
to the present note and should be read before any further protocol change here.

**Tomashenko, Srivastava, Wang, Vincent, Nautsch et al., "Introducing the
VoicePrivacy Initiative", arXiv:2005.01387 (2020).** The origin of the attacker
taxonomy (ignorant, lazy-informed, semi-informed) and of the enrollment and
trial split this project inherits. Worth re-reading as a design document rather
than as a challenge announcement: the choices it makes are the ones we are now
questioning.

**Tomashenko, Miao, Champion, Meyer, Wang et al., "The VoicePrivacy 2024
Challenge Evaluation Plan", arXiv:2404.02677**, and **Tomashenko, Miao,
Champion, Meyer, Panariello et al., "The Third VoicePrivacy Challenge",
arXiv:2601.11846 (2026).** The evaluation plans are where the dataset decisions
are actually recorded. The 2026 edition adds emotion and content preservation,
which widens the utility side without widening the population.

### The target speaker is an evaluation confound, and this is measured

**Franzreb, Das, Polzehl and Möller, "Improving the Speaker Anonymization
Evaluation's Robustness to Target Speakers with Adversarial Learning",
arXiv:2508.09803 (2025).** Directly relevant and it predates our own work on
this. Their finding: the standard evaluation **overestimates privacy** when a
same-gender target selection algorithm is used, because anonymized speech
carries information from both the source and the target speaker and the
evaluation does not separate them. They add a target classifier to measure the
target's contribution and remove it with adversarial learning.

This matters for two things here. It is prior art for
[`research-ideas.md`](research-ideas.md) idea 7, pseudo-speaker selection as an
evaluation confound, which must now be positioned against it rather than
presented as new. And it is the natural companion to the CNIL G3 audit in
[`../legal_eval/pseudo_speaker.py`](../legal_eval/pseudo_speaker.py): they ask
how much target information leaks into the evaluation, we ask whether a target
is close enough to a real person to be accepted as them. Related, not the same,
and the pair is stronger than either.

### Per-speaker risk, and why averages hide it

**Dufour, Magron, Rouvier and Vincent, "A Large-Scale Per-Speaker Analysis of
Re-identification Risk in Speech Anonymization", arXiv:2606.07210 (2026).**
Average-case metrics such as the EER hide large disparities between
individuals, and they measure per-speaker linkability under a worst case at
large scale. This is why that direction is **dropped** from our own list rather
than pursued: it is done, and done at a population we do not have.

It also bears on the sizing argument above. A design that fixes the enrollment
budget so every attacker reference is equally strong, as ours does, makes the
population more uniform and therefore makes per-speaker disparity easier to
attribute to the speaker rather than to how much audio they happened to
contribute.

### Benchmark construction

**Miao, Tao, Zeng and Wang, "A Benchmark for Multi-speaker Anonymization",
arXiv:2407.05608 (2024).** Benchmark design for the multi-speaker case, which
every corpus used here avoids by construction: LibriSpeech and Common Voice are
both single speaker per recording. Our subsets inherit that limitation and
should say so.

### Dataset documentation and demographic composition

**Papakyriakopoulos, Choi, Andrews, Bourke, Thong et al., "Augmented Datasheets
for Speech Datasets and Ethical Decision-Making", arXiv:2305.04672 (2023).**
Extends datasheets for datasets to speech, and applies it to five widely used
corpora including Common Voice, finding systematic gaps in self-disclosed
documentation. Directly applicable: our subsets are a derived dataset and
currently ship a filelist and a construction script, not a datasheet.

This is the clearest gap in what we do. Two concrete consequences for the
subsets built from Common Voice 27:

* **Selecting the largest speakers first is a demographic choice, not a neutral
  one.** `--max-speakers` takes speakers in descending utterance count, so the
  subset over-represents prolific contributors. Common Voice's own demographic
  metadata is self-reported and optional, with age in ten-year bands, so the
  composition of the result is partly unknowable and should be reported as
  measured rather than assumed. The construction script already records the
  gender split it achieved, which is a start and not a datasheet.
* **A derived subset needs its own documentation.** Not the corpus's. What was
  filtered, on what criterion, what the resulting composition is, and what the
  subset cannot support.

### What this changes about the design above

Nothing in the three corrections (threshold on utterance counts, fixed
enrollment budget, separate distractor bar) is contradicted by this literature.
Two things are added:

1. **Report the target speaker's contribution, not only the source's.**
   Following arXiv:2508.09803, an evaluation that does not separate source from
   target information overestimates privacy. Our subsets do not currently
   support that measurement, because they define no target pool.
2. **Ship a datasheet with the subsets.** Following arXiv:2305.04672.
   Written: [`cv27-datasheet.md`](cv27-datasheet.md).

Both are now in [`research-ideas.md`](research-ideas.md).
