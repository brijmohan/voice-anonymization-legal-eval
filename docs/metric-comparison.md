# Comparing the recent privacy metrics

Several groups have proposed replacing the equal error rate as the measure of
residual re-identification risk in anonymized speech, and none has been run
against the others on the same data.

**Correction, 12 October 2026.** This note previously said "four groups", and
listed the four below. That undercounts, and it undercounts in a way that
matters for anyone deciding what to adopt. Reading the VoicePrivacy 2026
checkout shows that the challenge's own evaluation code already ships four
further privacy quantities **[M]**:

| Already in the challenge's code | What it is | Origin |
|---|---|---|
| `metrics/cllr.py` | Cllr, min and actual | Brummer et al., via Nautsch |
| `metrics/linkability.py` | `D_sys` linkability | Gomez-Barrero et al. |
| `metrics/utils/zebra_plots.py` | ZEBRA: population and worst-case disclosure | Nautsch et al., Interspeech 2020 |
| `metrics/utils/plo_plots.py` | privacy log-odds profiles | Nautsch et al., Interspeech 2020 |

So the field has roughly **nine** privacy quantities, not five, and four of them
are already installed alongside the systems being evaluated. **This comparison
covers five of the nine.** Extending it to the rest is the obvious next step
and is not done.

Two consequences worth stating plainly rather than leaving implicit.

**ZEBRA got to standardisation first.** It was offered in 2020 as "a candidate
for incorporation into biometric information protection standards" **[P]**. Any
proposal of ours in that direction completes that path rather than opening one,
and should say so.

**ZEBRA reports a population figure and a worst-case individual figure
separately** **[P]**. An average that looks acceptable can conceal one fully
exposed person. None of the metrics in the table below does this, ours included,
and on reflection the split looks like a property a procedure should require
rather than one metric's design choice.

| Proposal | Unit | What it asks |
|---|---|---|
| Singling Out, `pi_sing` | probability | does a calibrated predicate isolate exactly one record from a set of `N` |
| Linkability, `pi_link` | probability | does the true identity rank first among `N` |
| Similarity Rank Disclosure (SRD) | bits | how concentrated is the rank of the true identity |
| Local Information Disclosure (LID) | bits | how much of `log2(N)` does the calibrated posterior recover |

This note records what we find when all of them, plus the EER, are computed
from one set of embeddings and one enrollment and test split.

**The LID and SRD implementations in `legal_eval/comparison.py` are our reading
of the published definitions.** They exist so that a comparison can be run
without asking their authors to do the work first. They are not authoritative.
They are deliberately small and heavily commented so that checking them is a
short task rather than a reimplementation, and anywhere our reading is wrong,
the conclusions below are wrong with it.

Reproduce everything here with:

```sh
python examples/04_compare_privacy_metrics.py --demo
python examples/04_compare_privacy_metrics.py --results-dir exp/asv_ssl
```

## Four axes, not nine metrics

The most useful thing to come out of running these together is that the field's
privacy quantities do not measure nine things. They sort by **what each one is
averaged over**, and within a row they largely agree:

| averaged over | metrics | status here |
|---|---|---|
| trials, as a distribution | ROCCH-EER, Cllr, Gomez-Barrero `D_sys`, ZEBRA `D_ECE` | implemented |
| **the worst individual** | ZEBRA `max |LLR|` | implemented |
| a candidate population of size `N`, by **rank** | Linkability, SRD, LID | implemented |
| a candidate population of size `N`, by **threshold** | Singling Out | implemented |

**Within the rank row the metrics cluster numerically but are not
interchangeable.** On Common Voice, asked what fraction of the available room
McAdams closed, Linkability, SRD MeanD and LID ALID fall within 0.02 to 0.06 of
each other while Singling Out says several times as much **[M]**.

**That clustering is not agreement, and an earlier version of this note said it
was.** With the standard errors propagated, the rank metrics differ from each
other by 10 to 40 standard errors almost everywhere **[M]**. Their differences
are small in absolute terms and far outside the noise, so they are measurably
different quantities. The estimate is conservative: all four come from the same
draws, so their errors are positively correlated and treating them as
independent overstates the error of a difference.

So this comparison does **not** support telling the field that three of its
metrics are interchangeable. What it supports is that Singling Out is a
different kind of measurement from the other three, separating from them by 6
to 32 standard errors as well as by a large absolute margin.

The one redundancy actually demonstrated is definitional. SRD's identification
rate **is** the Linkability metric: identical where the candidate draw is
deterministic, and differing by at most 0.0022 elsewhere, which is one quantity
sampled twice rather than two quantities agreeing **[M]**.

The mechanism is that a rank survives any monotone squeeze of the scores and a
threshold does not, so the compression that barely disturbs an ordering pushes
many candidates across a fixed decision boundary.

**The consequence for a procedure is to specify a population per metric**, not
one population for all of them, because each metric's usable range differs and
Singling Out's room collapses by a factor of 21 across the range measured here
**[M]**. An earlier version of this note recommended one metric per axis on the
grounds that the rank row was redundant; the standard errors do not support
that and the recommendation has been withdrawn. The four axes do map onto
distinctions that already exist in law: Article 29 separates singling out from
linkability, which is the threshold and rank rows, and the worst-individual row
is the one a data protection authority asks about when it asks whether any
individual is exposed.

**The worst-individual axis has exactly one member, and it is not ours.** ZEBRA
is the only quantity in the field that reports an individual rather than an
average, and an average that looks acceptable can conceal one person about whom
the evidence is overwhelming. That is the strongest single reason to build on
ZEBRA rather than around it.

## These three are the authors' metrics, not our reading of them

Cllr, the Gomez-Barrero linkability and ZEBRA differ from SRD and LID in an
important way: **reference implementations ship with the VoicePrivacy
Challenge**, so they did not have to be taken from the papers.
`legal_eval/trial_metrics.py` is implemented against that code and
`scripts/verify_trial_metrics.py` checks it over 30 score sets spanning
separable, overlapping, tied, bounded and skewed shapes **[M]**:

| quantity | worst absolute difference |
|---|---|
| PAV | **0** |
| optimal LLR | 1.776e-15 |
| Cllr | 1.110e-16 |
| minimum Cllr | 2.220e-16 |
| Gomez-Barrero `D_sys` | **0** |
| ZEBRA `D_ECE` | 3.247e-15 |
| ZEBRA worst case | 4.441e-16 |
| ZEBRA category tag | 30 of 30 agree |

Two details worth carrying into any report.

**ZEBRA's category tag is the answer to legibility.** It maps the worst-case
figure onto the European Network of Forensic Science Institutes' scale for
strength of evidence, so a result arrives as a category rather than only a
decimal, and that scale already has forensic currency in Europe. Across three
synthetic systems of increasing protection the tag moves C, B, A **[M]**. No
new scale should be invented for this.

**Cllr must be reported actual and minimum.** The gap between them is
calibration loss, which belongs to the attacker rather than to the
anonymization, so the actual figure alone flatters a system whose scores happen
to be poorly calibrated. The actual figure can exceed 1 bit, which means
treating raw cosine scores as log-likelihood ratios is worse than learning
nothing **[M]**.

## Why published numbers cannot be compared directly

The four proposals were evaluated on different corpora, with different attacker
models, different enrollment and test splits, and different population sizes.
Those differences are larger than the differences between the metrics, so a
table assembled from published values measures the pipelines, not the metrics.
Everything below therefore comes from a single artifact per system.

## What is already shared

Three of the metrics are functionals of the same object: the attacker's
probability `q` that a trial belongs to the true identity, inside a closed set
of `N` enrolled speakers. Each reports a form of `log2(N q)`, which is zero when
`q = 1/N`, that is when the attacker has learned nothing. They differ in how `q`
is estimated and how it is summarised.

Two consequences follow, and neither appears in the papers.

**SRD's identification rate is the Linkability metric.** SRD reports `IdR`, the
probability that the true reference ranks first in the closed set. That is
Linkability's definition. The two families already overlap in a published
statistic, so they are not alternatives at that point, they are the same number
under two names. `tests/test_comparison.py` pins this.

**LID's intercept and prior term cancel.** LID maps z-normalised scores to
log-likelihood ratios by Platt scaling, `LLR = w * s_hat + b - ln(N_mated /
N_nonmated)`, and then takes a softmax. A softmax is shift invariant, so `b` and
the prior ratio drop out exactly. The fitted slope `w` is the only part of the
calibration that reaches the reported bits, which makes LID far more sensitive
to that one number than the three-term expression suggests.

Singling Out is not of this family. It asks whether a calibrated predicate
isolates exactly one record, a different legal criterion, and its uninformative
baseline is `exp(-1)` rather than `1/N`. It also needs an artifact the others do
not: calibration conversations per speaker. A score matrix alone is insufficient
for it.

## How uncertainty is estimated, and a trap in it

Every value reported here carries a standard error, because a comparison without
one invites conclusions the data does not support. Two systems whose metric
values differ by less than the combined standard error have not been shown to
differ, and when a metric is near its ceiling that is the usual case.

The spread comes from resampling trials with replacement and then drawing the
closed candidate sets, ten replicates. The EER is bootstrapped over trials the
same way, so that the incumbent metric can be compared on the same footing
rather than quoted as a single number.

**The trap.** Estimating the spread from the closed-set draw alone is not
enough. When `N` equals the number of enrollment speakers, every speaker is in
every closed set, the draw is deterministic, and the spread is exactly zero.
That is the largest `N` available and the most interesting one, so a spread that
vanishes there makes every difference look resolved. We reported one inversion on
that basis before noticing, and the standard errors in question were exactly
`0.0000`. A standard error of exactly zero is a signal that the resampling is
not capturing the right source of variation.

## Finding 1: the metrics agree on the ranking

On three systems of differing strength, all five order them identically, least
disclosure first. This holds on synthetic data and on the VoicePrivacy
baselines.

That matters, because the natural reading of four competing proposals is that
they disagree about which system is more private. They do not. The disagreement
is about the margin, and about what the number means once you have it.

## Finding 2: the two families move in opposite directions with `N`

| Metric | As `N` grows |
|---|---|
| Singling Out | falls |
| Linkability | falls |
| SRD MeanD | rises |
| LID ALID | rises |

This is structural rather than a quirk of the data. A larger search space makes
a successful identification less likely, which is what the probability metrics
report, and makes each success more informative, which is what the bit metrics
report. Both are correct. They answer different questions, so neither is a
substitute for the other, and a network that adopts one of them loses the other
axis.

The practical consequence is that "how much privacy does this system give" has
no single answer without first fixing `N`, and the two families then disagree
about which direction is better.

## Finding 3: LID is a joint property of the system and the calibration

This is the result with the most consequence for a shared evaluation platform.

Hold the data and the split fixed and vary only the calibration slope. The rank
based metrics cannot move, because they use the ordering alone. ALID moves by
tens of bits, and changes sign:

```
--- calibration sensitivity, weak anonymization, N = 50 ---
Linkability 0.0513 and SRD MeanD 0.2753 use the ranking only, so no
choice of calibration can move them. LID is a function of the slope:
  slope w   LID ALID   LID PDR   LID max
    0.100     0.0603    0.6067    0.4860
    0.466     0.1530    0.5433    2.1141   <- fitted
    1.000    -0.0728    0.4533    3.9139
    4.000    -5.6808    0.2167    5.6300
   16.000   -35.9841    0.0867    5.6439
```

A reported ALID is therefore not interpretable without the calibration protocol
that produced it, and no such protocol is specified in the proposal.

It gets sharper. The calibration is normally fit on the system being evaluated,
which makes the slope a function of that system. A system whose scores carry
less information fits a flatter slope, and a flatter slope pushes the posterior
towards uniform, which pushes ALID towards zero whatever the system actually
leaked. The metric partly cancels the effect it is measuring.

Evaluating every system under every system's fitted calibration shows the size
of this:

```
LID ALID in bits, N = 50
                            w=2.11      w=0.47      w=0.09
                        (original)      (weak)    (strong)
original                     2.381       1.039       0.222
weak anonymization          -1.607       0.153       0.054
strong anonymization        -2.789      -0.114       0.004
```

The diagonal is what a paper reports. Read down the first column, under one
shared calibration, and the three systems span 5.2 bits. Read the diagonal and
they span 2.4 bits. Same systems, same data, same split.

The top right entry is the one to dwell on. Under a flat calibration, LID
reports that **original, un-anonymized speech discloses 0.222 bits** out of the
`log2(50) = 5.64` available. On that same data Linkability is 0.359 against a
chance level of 0.02, so the attacker identifies the correct speaker eighteen
times more often than guessing. A mis-specified calibration turns that into a
near-zero disclosure figure. Linkability and SRD cannot produce this failure,
because they never fit anything.

None of this says LID is the wrong instrument. A calibrated posterior is the
right object if you want disclosure in bits, and the quantity is well defined
once the calibration is. It says the calibration protocol is load bearing and
currently unspecified, which is a gap a shared platform can close and an
individual paper cannot.

## What this implies for a shared platform

1. **Report a probability metric and a bit metric, not one or the other.** They
   move in opposite directions with `N` because they answer different questions.
2. **Fix `N` explicitly, and report the curve rather than a point.** Every
   metric here except the EER depends on it, and the EER's independence from it
   is the reason the EER was found inadequate.
3. **Specify LID's calibration protocol, including whether the calibration set
   is the system under evaluation.** Without it, ALID values from two papers are
   not comparable, and a flat calibration can report near-zero disclosure for
   un-anonymized speech.
4. **Do not treat SRD's `IdR` and Linkability as independent evidence.** They
   are the same statistic.

## What would falsify the above

Finding 3 rests on our reading of the Platt scaling step. If LID's calibration
is specified on a fixed external development set rather than the evaluation
system, the feedback effect disappears, though the slope sensitivity remains.
If the intended normalisation differs from the per-trial z-norm we implement,
the fitted slopes change and the magnitudes with them.

Both are cheap for the authors to check against our implementation, which is why
it is written to be read. See `legal_eval/comparison.py`.

## Verifying our reading, in an afternoon

The findings above depend on our reading of two other groups' definitions, so
the implementations are built to be checked rather than trusted.
`scripts/make_verification_kit.py` writes a self-contained directory holding one
score matrix, the true identity per trial, the values our implementation
produces across a grid of population sizes and calibration slopes, and the
implementation itself:

```sh
python scripts/make_verification_kit.py --output-dir verification_kit
```

The matrix is synthetic by default, so the kit carries no data licence and can
be sent to anyone. `--from-vpc` builds it from a VoicePrivacy run instead.

Reporting ALID across a grid of slopes rather than at one fitted value is
deliberate: it separates "do we agree on the metric" from "do we agree on the
calibration", and those are different questions with different answers.
