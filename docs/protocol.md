# The evaluation protocol

For somebody running this on their own anonymization system, including
VoicePrivacy Challenge participants. It states what to compute, what to report
alongside each number, and what makes a reported result uninterpretable.

The metrics are in [`metrics.md`](metrics.md) and the reasoning behind the
choices is in [`constitution.md`](constitution.md). This document is only the
procedure.

## Why a protocol rather than a metric

A privacy number is meaningless without three things beside it: the population
it was computed over, the attacker that computed it, and the chance level for
that population. The field reports the first two inconsistently and the third
almost never.

That is not pedantry. Singling Out has a chance level of `exp(-1) ≈ 0.368`, so
a reported 0.37 is indistinguishable from an attacker who learned nothing.
Linkability has a chance level of `1/N`, so the same 0.37 is near-perfect
identification at `N = 3` and useless at `N = 10000`. Numbers of this kind have
been compared across papers as though they were commensurable.

## Step 1: build the evaluation sets

Two sets of utterances from the same speakers:

- **enrollment**, what the attacker holds about each candidate
- **test**, what the attacker is trying to attribute

**Utterance disjointness between them is not negotiable.** An utterance on both
sides measures memorisation rather than linkage. Speaker overlap, by contrast,
is required: the attacker has to have something to match against.

Give every enrollment speaker the **same number of utterances**. With a floor
rather than a fixed budget, speakers arrive with anything from the floor to
hundreds, and part of the per-speaker spread you then report is an artefact of
the sampling. [`subset-design.md`](subset-design.md) has the argument;
[`cv27-datasheet.md`](cv27-datasheet.md) is a worked example.

**Report `N`.** Everything downstream is a function of it.

## Step 2: state the attacker

An attacker is a concrete system with stated knowledge, not an abstraction.
CNIL's correspondence on this work requires risk quantified against a concrete
attacker with stated sample diversity and capability, which is a regulator
asking for exactly the two axes to report.

At minimum, say:

- what model produces the embeddings, and whether it saw anonymized speech
- whether the attacker knows the anonymization system, and whether it
  anonymized its own enrollment data: ignorant does neither, lazy-informed
  does both without retraining, semi-informed also retrains on anonymized
  data. VPC 2026 specifies only the last two, and ranks on semi-informed
- the operating point, where the metric needs one

## Step 3: check the anonymized audio is speech

Before any metric. A system that emits silence instead of speech is
**rewarded** for it: the embedding sits at chance distance from every candidate,
and chance is the best privacy score attainable.

```python
from legal_eval.audio_integrity import check_anonymized
result = check_anonymized(anonymized, original)
if not result.usable:
    print(result.reasons)
```

This is not a hypothetical. A published 2026 baseline produced, for one Common
Voice utterance, a file of the correct length that was 99.93% digital silence
with 67 full-scale impulses. See [`constitution.md`](constitution.md) section
3.9. No evaluation in the field checks for it, so a system that fails on a
fraction of real crowdsourced audio scores as more private for failing.

**Report the count of unusable utterances, even when it is zero.** The
direction of the error is not neutral, so
`audio_integrity.optimistic_bias_note` writes the sentence to carry.

## Step 4: compute the metrics

```sh
legal-eval score-matrix --enroll-dir enroll --test-dir test --output-dir scores
legal-eval linkability   --score-matrix scores/*.npy --output results/link.json
legal-eval singling-out  --enroll-dir test --test-dir enroll --output results/sing.json
legal-eval eer           --score-matrix scores/*.npy --output results/eer.json
```

Note the **swap** in `singling-out`: the roles of the two sets reverse, because
Singling Out asks about isolating a target from a population rather than
matching a recording to a speaker.

For the newer proposals on the same artifact, see
[`metric-comparison.md`](metric-comparison.md) and
`examples/04_compare_privacy_metrics.py`.

## Step 5: report chance levels and bounds

Every metric, next to its value:

| Metric | Chance | Ceiling | Direction |
|---|---|---|---|
| Singling Out | `exp(-1) ≈ 0.368` | 1 | lower is more private |
| Linkability | `1/N` | 1 | lower is more private |
| SRD IdR | `1/N` | 1 | lower is more private |
| SRD, LID (bits) | 0 | `log2(N)` | lower is more private |
| ROCCH-EER | 0.5 | 0.5 | higher is more private |

```python
from legal_eval.preflight import chance_levels, ceilings
chance_levels(9779)   # per metric
ceilings(9779)
```

**A value at its ceiling must be declared.** A saturated metric cannot
distinguish two systems, so any claim that two metrics agree is weak evidence
when drawn from saturated conditions. At `N = 29`, four of six conditions we
measured sat against the bound.

## Step 6: error bars that resample the right thing

Resample **trials**, not only the candidate draw. At `N` equal to the
enrollment count the closed-set draw is deterministic, so spread taken from it
alone is exactly zero. That reads as precision and it is a bug. It produced a
false claim in this project before it was caught.

## Step 7: run the checks

```sh
legal-eval preflight results.json
```

Seven checks, each naming the failure it guards against. None of those failures
raised an error when they happened: they all produced a plausible number. Full
list in [`operational-hazards.md`](operational-hazards.md).

Exit status is 1 if anything failed.

## Step 8: if you report LID

LID is a **joint property of the system and its calibration**, not of the
system. Only the fitted Platt slope reaches the result, because the intercept
and the prior cancel in the softmax. On real data one condition read 1.561 bits
under its own calibration and **-207.3 bits** under another's, against a 4.858
ceiling.

So report the fitted slope with the value, and say whether the calibration
scores were separable. On separable scores the slope has no finite maximum
likelihood estimate; ours ran to `w = 199.4` and LID then equals `log2(N)`
whatever the system does.

## What makes a result uninterpretable

Any one of these, on its own:

1. No `N`.
2. No chance level, so a reader cannot tell the number from guessing.
3. A metric at its ceiling, not declared.
4. A standard error of exactly zero.
5. Silence on whether any utterance was excluded.
6. A LID figure with no calibration recorded.
7. SRD's identification rate reported as corroborating Linkability. They are
   the same quantity, and we measure them identical to 0.0000 where the draw
   is deterministic.

`preflight` checks all seven.

## Known limits of this protocol

Stated so that nobody has to discover them by trusting it:

- It measures two of the three Article 29 criteria well and the third,
  inference, recently and with less validation. See `constitution.md` 3.5
  and 3.10.
- It says nothing about metadata or verbal content. CNIL calls the assumption
  that those are handled separately a strong one, and it is ours too.
- The attacker is whatever you supply. A weak attacker produces a flattering
  number and the protocol cannot tell.
- Nothing here establishes a shelf life. A demonstration valid against today's
  attackers is not valid indefinitely, and CNIL requires it kept at the state
  of the art.

## If you find something wrong

That is the most useful thing that can happen to this protocol. Three of its
checks were corrected by running them against real output, twice against our
own. Open an issue.
