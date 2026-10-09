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
