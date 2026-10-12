# The research questions, and why each one is worth asking

Six questions. This document says, for each, what the question is, why it is a
scientific question rather than an engineering chore, what would count as an
answer, and where we actually stand. Claims carry [M] measured, [P] published,
[I] inferred or [?] open, as everywhere in this project.

The dependency structure matters as much as the questions, so it is stated
first.

## How they depend on each other

**RQ3 gates RQ1.** Every metric here is a function of the candidate population
`N`. At the VoicePrivacy population of 29 speakers, four of six conditions we
measured sit against their metric's ceiling **[M]**. A saturated metric cannot
disagree with another one, so any finding that the metrics agree is weak
evidence by construction until it is retested at a population with range. You
cannot answer the commensurability question without first answering the
population question.

**RQ4 explains RQ2.** Where identity survives tells you why a criterion
survives. A system that reduces verification while leaving attribute inference
untouched is not anomalous once you know the two draw on different channels.

**RQ6 bounds all of them.** If the protocol is biased, every number computed
through it inherits the bias, ours included. It is the reflexive question, and
it is the one that makes this project about measurement rather than about
anonymization systems.

**RQ5 stands alone** and is the only purely forward-looking one.

---

## RQ1: Do the proposed privacy metrics measure the same thing?

**The question.** Between 2025 and 2026 at least four groups proposed
replacing the equal error rate for voice anonymization evaluation: Singling Out
and Linkability (Vauquier et al., Interspeech 2025), Similarity Rank Disclosure
(Chandra, Pettenò, Evans, Panariello, Todisco, arXiv:2605.07291), and Local
Information Disclosure (Šterns, Drossos, Fernandes, Bäckström, Palamidessi,
arXiv:2607.06259). Each argues the EER is inadequate. **None had been run
against the others on the same data** **[M]**, because published values come
from different corpora, attackers and splits, so the numbers are not
comparable even in principle.

**Why it is a scientific question.** Metric proliferation without
commensurability is a specific failure mode. It allows each paper to report the
metric that flatters its system, and it leaves a reader unable to tell a
disagreement between systems from a disagreement between measuring
instruments. Resolving it has two possible outcomes and both are useful:

- If the metrics agree in their ordering of systems, **the field needs one, not
  four**, and the choice becomes a question of interpretability and legal
  defensibility rather than of discriminative power. That is a convergence the
  field can act on.
- If they disagree, **at least one is wrong, or they measure different
  threats**, and the difference has to be characterised before any of them is
  used to support a claim. A regulator acting on one number would then reach a
  different conclusion than on another.

The second outcome is not hypothetical. The CNIL correspondence shows a
regulator engaging directly with numbers of this kind, which makes
commensurability a practical question about whose answer someone will rely on.

**What would answer it.** All metrics computed from one embedding set, one
split and one attacker, across several systems and a range of populations, with
error bars that permit a difference to be called real. The partial order of
systems under each metric, with ties where the error bars overlap.

**Where we stand.** Partly answered, and the strongest results are structural
rather than numerical:

- **SRD's identification rate is the Linkability metric.** Not approximately:
  the same closed-set rank-1 quantity, measured identical to 0.0000 where the
  candidate draw is deterministic **[M]**. They must not be reported as two
  metrics corroborating each other.
- **LID is a joint property of the system and its calibration, not of the
  system.** The Platt intercept and the prior term cancel in the softmax,
  leaving the fitted slope as the only calibration parameter that reaches the
  result **[M]**. One condition reads 1.561 bits under its own calibration and
  -207.3 bits under another's, against a 4.858 ceiling **[M]**. A published LID
  figure without its slope cannot be reproduced.
- **No inversions across six conditions** **[M]**, but four of the six are
  saturated, so this is weak evidence pending RQ3.

---

## RQ2: Can all three Article 29 criteria be measured on speech?

**The question.** Article 29 Working Party Opinion 05/2014 defines
anonymisation by three tests: singling out, linkability, and **inference**. The
published framework implemented two **[M]**.

**Why it is a scientific question, not a box to tick.** The legal standard is
**conjunctive**. Failing any one criterion means the data is not anonymous. So
measuring two of three and reporting favourably is not a partial result that
will be completed later: it is potentially a **wrong conclusion**, presented
with the authority of the two criteria that were measured.

There is also a genuine conceptual problem underneath, which is why this is
research rather than implementation. The machine learning privacy literature
reaches for **membership inference** when it hears "inference". The Opinion is
asking something else: whether an *attribute value* of a person can be derived.
Membership inference, whether a given record was in the set, is nearer to
singling out than to inference. Conflating the two is a category error that
produces a number answering the wrong question, and the resolution had to be
argued from the Opinion's text rather than from practice.

**What would answer it.** An attribute inference measurement computed from the
released artifact, not from an internal representation, reported against the
population prior rather than as raw accuracy, and grouped so that a classifier
recognising the speaker cannot score on the attribute for the wrong reason.

**Where we stand.** Implemented and now measured on real data for the first
time **[M]**. Predicting reported gender from the released ASV embeddings of
the Common Voice subsets, one row per speaker, on the 5,879 of 10,000
enrollment speakers who report one:

| condition | prior | accuracy | advantage | female recall |
|---|---|---|---|---|
| original, 6 utterances per speaker | 0.7195 | 0.9119 | +0.6859 | 0.8363 |
| McAdams, 6 utterances per speaker | 0.7195 | 0.9136 | +0.6919 | 0.8423 |
| original, 20 utterances per speaker | 0.7220 | 0.9334 | +0.7604 | 0.8771 |

**McAdams moved the advantage by +0.006**, which is not a reduction and is
inside the noise. The same system moves Linkability from 0.955 to 0.914 **[M]**.

So the criteria are **not redundant**, which is the finding. A controller
reporting Singling Out and Linkability alone would report that this system
reduces risk, while 84% of the women in the population remain correctly
identified as women from the anonymized embedding.

Caution, because the number is strong and will be quoted: the features are
embeddings trained for speaker discrimination, so this establishes that such an
embedding carries gender. It does **not** establish that a system which
defeated verification would also defeat gender inference **[?]**. ASR-BN, which
moves Linkability to 0.386, is the experiment that would settle it.

---

## RQ3: What population and attacker does a defensible claim require?

**The question.** Every metric here is a function of two things the reporter
chooses: the candidate population `N`, and the attacker's capability. What do
those have to be for a claim to mean anything?

**Why it is a scientific question.** Because the metrics are not
scale-invariant and the field reports them as though they were.

Linkability's chance level is `1/N`. Singling Out's is `exp(-1)`, about 0.368,
so a reported 0.37 is indistinguishable from an attacker who learned nothing. A
bit metric's ceiling is `log2(N)`. The consequence is that **the same number
means opposite things at different populations**: Linkability of 0.37 is
near-perfect identification at `N = 3` and useless at `N = 10000`. Values of
this kind have been compared across papers as if commensurable.

The saturation problem is the sharp form of this. We measured that at `N = 29`,
four of six conditions sit against their bound, and Linkability for original
`libri_test` is exactly 1.000 at every `N` **[M]**. In that regime no metric can
separate two systems, so a comparison can look perfectly consistent while
carrying no information. **This is why RQ3 gates RQ1.**

**Why it is also a legal question.** CNIL obligation G2 requires risk
quantified against a concrete attacker with stated sample diversity and
capability. That is a regulator asking for precisely these two axes. So this is
not methodological fussiness; it is the form a compliance demonstration has to
take, and a framework that omits it cannot support the claim it exists to
support.

**What would answer it.** Metrics measured across a population spanning orders
of magnitude, with the saturation point identified, and with the attacker's
capability stated in terms a third party can reproduce. Then: the smallest `N`
at which the metrics still discriminate, and the attacker strength below which
a favourable number is an artefact.

**Where we stand.** In progress, and this is the current bottleneck. Saturation
is measured. Common Voice 27 subsets are built at 10,000 enrollment speakers
with exactly zero utterance overlap **[M]**, embeddings are extracted, and the
comparison at populations up to 9,779 is running. One honest limitation: Common
Voice carries only two systems, original and McAdams, so **breadth exists at
`N = 29` and scale exists for two systems, and they are currently separate
axes** **[M]**.

A labelling error found while auditing against the VPC 2026 plan belongs here,
because it is exactly the kind of thing G2 is about: the attacker we described
as *ignorant* is what the plan calls **lazy-informed** **[M]**. Lazy-informed is
the stronger of the two, so the numbers were conservative, but the capability
was misstated.

---

## RQ4: Where does identity survive anonymization?

**The question.** Which part of the signal carries the identity that survives
an anonymization pipeline?

**Why it is a scientific question.** Because design discussions keep reaching
for the spectral envelope, and if that is the wrong knob then the field is
optimising the wrong thing. This is the question with the most direct influence
on what a system builder does next, and it is answerable by measurement.

**What would answer it.** A decomposition: for each place identity could
survive, a measurement of how much the metrics move when only that place is
modified.

**Where we stand.** Partly answered, and the headline is a negative result:

- **The spectral envelope is not where identity lives.** McAdams warping
  modifies the envelope and nothing else, and moves Linkability from 0.955 to
  0.914. ASR-BN moves it to 0.386 **[M]**. The sharpest single piece of
  evidence we have, and it points away from the knob the field reaches for.
- **Vocoder drift is learnable and reversible** **[P]**, so a vocoder is not a
  privacy mechanism.
- **Prosody is identity-linked** and the mitigation is random offsets **[P]**.
- **Content leaks identity through vocabulary** in LibriSpeech **[P]**, which
  bounds what any acoustic anonymization can achieve on that corpus.

This also closes a loop with RQ2. Attribute inference survives anonymization
that reduces verification because the two draw on different channels, so the
inference result is not a surprise once RQ4 is taken seriously.

Open: the accent channel is unmeasured **[?]**, and pseudo-voice
identifiability is measured but uninterpretable at `N = 29`, where chance is
0.928 **[M]**.

---

## RQ5: How long does a privacy demonstration stay valid?

**The question.** A demonstration is made against the attackers of the day.
When does it expire, and what obliges a re-run?

**Why it is a scientific question.** Because anonymisation is not a property of
data alone; it is a property of data **relative to attacker capability**, and
that capability improves. GDPR assesses anonymisation against "means reasonably
likely to be used", which is a standard that is **itself time-indexed**. No
technical framework reflects that. The consequence is that "this data is
anonymous" is a claim with a date on it, and nobody states the date.

CNIL obligation G4 requires the demonstration kept at the state of the art,
which turns this from an interesting observation into a standing obligation with
no defined trigger. The VoicePrivacy Attacker Challenge exists precisely because
attacks improve, so the raw material for an answer is available **[P]**.

**What would answer it.** An empirical decay curve: the same anonymized data
re-evaluated against successive generations of attacker, giving the rate at
which a demonstration degrades, and from that a defensible re-assessment
interval. Plus a rule for what event obliges a re-run ahead of schedule.

**Where we stand.** Not started **[M]**. The right move is almost certainly to
contribute to the Attacker Challenge rather than build a parallel registry, and
`preflight` is the natural carrier for a re-assessment rule once there is one.

---

## RQ6: Does the evaluation protocol itself bias the result?

**The question.** Does the way we measure systematically favour or disfavour
some systems, independently of their merit?

**Why it is a scientific question, and the most uncomfortable one.** Because if
the protocol is biased, every number computed through it inherits the bias,
**including ours**. It cannot be deferred to future work without qualifying
every other answer in this document. It is also the question that justifies the
harness: a protocol that cannot be audited cannot be trusted, by us least of
all.

**What would answer it.** For each protocol choice, a measurement of how much
the result moves when the choice is made differently.

**Where we stand.** Partly answered, and the clearest instance was found by
accident this week, which is itself evidence for how exposed this question is:

- **A destroyed utterance scores as perfect privacy.** The 2026 McAdams
  baseline emitted, for one Common Voice utterance, a file of the correct length
  that is 99.93% exact zeros with 67 near-full-scale impulses, against 93,487
  audible samples in the original **[M]**. The attacker raised on it, which is
  the visible outcome. The ordinary outcome is that a more forgiving extractor
  embeds six seconds of near-silence, that embedding sits at chance distance
  from every candidate, and the utterance counts as a privacy success. Chance is
  the best privacy score attainable, so **a system is rewarded for destroying
  speech by exactly as much as for anonymizing it perfectly**, and **no
  evaluation in the field checks for it** **[I]**. Nobody is cheating; the
  protocol cannot tell the two apart.
- **Target speaker leakage overestimates privacy** under same-gender selection
  **[P]**. The VPC 2026 plan moving to mixed, cross-gender trials is the
  challenge correcting for the same effect **[M]**.
- **Subset design is a demographic choice.** Selecting speakers by prolificacy,
  as we do, produces a population that is not representative, and part of the
  gender skew in our subsets is an artefact of that **[M]**.
- **The official attacker preprocesses one utterance two ways**: the filterbank
  stream receives loudness-normalised audio and the WavLM stream re-reads the
  raw file **[M]**. It does not bias a comparison, since the same attacker is
  applied to every condition, but it is a caution for corpora whose level varies
  more than read audiobook speech.

Open, and squarely our own exposure: whether the legal metrics inherit the
target-leakage bias is untested **[?]**. Singling Out calibrates per enrollment
speaker so it may absorb it; Linkability is a rank statistic so uniform leakage
should cancel. Neither has been checked.

---

## Review, 12 October 2026: do these questions need revising?

The first full comparison has run. This is the review that follows it, kept
here rather than in a separate note so the questions and their revisions stay
together.

**Two revisions, one addition, one warning, three unchanged.**

### RQ1 succeeded, and its successor is a different question

RQ1 asked whether the metrics measure the same thing. It got a real answer, in
three parts **[M]**: they agree on the **ordering** with zero inversions, their
**magnitudes** differ by 10 to 40 standard errors so they are not
interchangeable, and their **resolving power** differs, with Linkability and
SRD separating 6 of 6 system pairs where the EER and Singling Out separate 4.

So the question worked and is largely closed on this corpus. But notice that
the binary framing, "the same thing or not", is not where the answer lives. The
question a standard actually needs is:

> **RQ1'. Which metrics must a procedure require, and over what population
> range is each one valid?**

That is not a restatement. RQ1 could be answered by a table of correlations;
RQ1' cannot, because it needs the usable range per metric, which is what 3.14
showed is the thing that bites. Treat RQ1 as answered and RQ1' as the live
question.

### RQ3 is answered further than expected, and should narrow

RQ3 asked what population and attacker a defensible claim requires. It
corrected us twice: the population is bounded **below** by saturation and
**above** by task-difficulty confounding, and the bounds **differ per metric**,
with Singling Out's room collapsing 21-fold across the range while
Linkability's grows **[M]**.

What remains open is one axis only, and we already have the models for it:

> **RQ3'. How does the usable population window move with attacker strength?**

Everything measured so far is the lazy-informed attacker. The semi-informed
models are trained for all three LibriSpeech systems and have never been used
for the comparison. A stronger attacker raises the original's score, which
pushes the confounding boundary outward, so the window should widen. That is a
cheap and falsifiable prediction **[?]**.

### A new question the work produced

> **RQ7. Does a defensible anonymity claim require an individual-level
> disclosure figure, or does a population average suffice?**

Article 29's three criteria are all about what can be done to *a* person, but
every metric this project implemented reports a population average. ZEBRA's
worst-case figure is the only quantity in the field that reports an individual
**[M]**, and CNIL obligation G3, that pseudo-identities must not identify a
real person, is adjacent to it without naming it.

This matters because an average that looks acceptable can conceal one fully
exposed person, and because the answer determines whether a standard must
require two figures per metric rather than one. It is also the axis we do not
occupy, which is the strongest reason to approach ZEBRA's authors as
collaborators.

### A warning, not a revision

**RQ5, the shelf life of a demonstration, has had no work at all.** It is the
only one of the six in that position, and it is the one with a regulator
explicitly asking for it, through CNIL G4's requirement to keep the
demonstration at the state of the art.

It is also the question most likely to be deferred indefinitely, because it
needs a longitudinal measurement that no single run produces. Recording that
here so the deferral is a decision rather than a drift.

### Unchanged: RQ2, RQ4, RQ6

RQ2 and RQ4 stand as written and both advanced.

RQ6 stands as written and deserves a note. **It was the most productive
question of the programme**, and it is the one that looked most like
methodological housekeeping. Most of what was learned came from it: a destroyed
utterance scoring as perfect privacy, our own implementation not scaling to the
population the question required, extraction shards becoming twelve fictitious
systems, and a metric read outside its usable range certifying a system as
anonymous.

The lesson generalises past this project. **The question about whether the
instrument works outproduced the questions about what the instrument measures**,
and it would have been the easiest one to postpone.

## What would falsify the programme

Stated here because a set of research questions with no failure condition is a
research agenda rather than a scientific one.

If the metrics turn out to agree everywhere, at every population and against
every attacker, then the commensurability work has a short answer and the
interesting question moves entirely to RQ4 and RQ6. That would be a real result
and it would shrink this project considerably.

If attribute inference turns out to be an artefact of using speaker-trained
embeddings, and vanishes against features chosen for the attribute, then RQ2's
headline weakens to a statement about ASV embeddings rather than about
anonymization.

And if the protocol biases found under RQ6 turn out to dominate the differences
between systems, then no current number in the field means what it claims,
ours included, and the honest output of this project is a correction rather
than a framework.
