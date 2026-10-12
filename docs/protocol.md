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

**Also evaluate the original, unprotected audio at the same `N`, and report it
beside every result.** This is not a formality and it is the step most likely to
be skipped.

The attacker's score on the original audio is its actual ceiling on that task.
Without it a reader cannot tell protection from difficulty. Measured on Common
Voice, the same attacker scores 0.9131 on original audio at `N = 29` and 0.4515
at `N = 9,779` **[M]**. At the first population it is pinned near the bound, so
no system can be shown to differ from another. At the second it already fails on
more than half the unprotected audio, so more than half of any privacy measured
there is the task being hard.

So the population is bounded on both sides, and a result is interpretable only
in the window between. Reporting the original's value at each `N` makes that
window visible without anyone having to agree a threshold for it. On this corpus
and attacker the window is roughly `N = 50` to `250`; on yours it will differ,
because it is a property of the attacker and the corpus rather than of the
metric.

This is the methodology point in Hanisch, Todt, Patino, Evans and Strufe,
*A False Sense of Privacy* (PoPETs 2024), which should be read alongside this
document.

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

## Cost, before you pick a population

**The expensive part is not the metric you would guess, and it is worth saying
how that was established: by attaching `py-spy` to the running process rather
than by reasoning about the code** **[M]**. A cost model inferred from reading
`singling_out_sweep` was wrong, and wrong by enough to have mispriced a run by
a factor of ten.

The dominant cost at a large population is the **EER standard error
bootstrap**. It resamples trials 20 times, and each replicate copies the score
matrix, builds a boolean mask of the same shape, extracts the non-mated scores
into another array of nearly that size, then sorts them for the convex hull. On
a 9,779 by 9,779 matrix that is roughly 1.6 GB of allocation and 95 million
values sorted, per replicate, 20 replicates, **per system** **[M]**.

Two consequences:

- It is **O(N^2) in the population** and **independent of how many values of
  `N` you sweep**. It is paid once per system whatever ladder you ask for.
- It is single-threaded. The observed process sat at 101% CPU on a 12 core
  machine **[M]**, so more cores do not help.

Singling Out is the other cost, and that one does scale with the ladder:

    draws = enrollment speakers x runs x folds x number of N values

which at 9,779 speakers with three runs and three folds over nine values of
`N` is 792,099 draws **[M]**.

The practical consequences for anyone planning a run:

1. **Sweep the whole ladder in one invocation.** Splitting it into stages pays
   the EER bootstrap again in each stage, and that is the dominant term. A
   two-stage plan written here paid it twice for nothing.
2. **Price the population before the ladder.** Doubling `N` quadruples the
   bootstrap. Adding four more values of `N` costs only more draws.
3. **Measure, do not model.** Attach a sampling profiler to the running job.
   The stack says in one line what an afternoon of reasoning got wrong.

Reducing folds or runs cuts the Singling Out cost proportionally but removes
the spread, and a standard error of exactly zero is one of the things
`preflight` refuses.

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

## Step 6b: produce the three-point report

```sh
python scripts/report_three_point.py results.json --original "my corpus original"
```

It prints, per population: the chance level, the attacker's score on the
original audio, the anonymized system's score, the **room** between chance and
the original, and the fraction of that room the system **closed**.

`closed` is the figure to compare across metrics, because it is the only one
normalised by what was achievable rather than by a bound nothing can reach. It
also flags the rows where the population rather than the system is deciding the
number.

## Step 7: run the checks

```sh
legal-eval preflight results.json
```

Nine checks, each naming the failure it guards against. None of those failures
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
