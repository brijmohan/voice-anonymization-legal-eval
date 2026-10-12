# Constitution

What this project is for, what we have actually established, and the tests a
proposed piece of work has to pass before it earns time. `CLAUDE.md` holds the
engineering invariants that serve this document.

Status markers used throughout, and they are load bearing:

- **[M]** measured in this project, with the number and where it came from.
- **[P]** published by someone, cited.
- **[I]** inference from [M] or [P]. Reasoning given. Not a result.
- **[?]** open. Needs evidence we do not have.

A claim with no marker is a mistake. If you cannot mark it, you do not know it.

---

## 1. The objective

**Make "is this speech data personal data?" answerable with evidence a data
protection authority will accept, and make the answer reproducible by someone
who does not trust us.**

Everything else is instrumental. Three consequences:

1. **The numbers are the public contract**, not the API. A change that moves a
   computed value is a breaking change, even if no signature changes.
2. **Legal defensibility beats metric elegance.** An information-theoretic
   quantity cannot go in a DPIA. A validated criterion can. This is a claim
   about regulatory form, not about which is better science, and the
   distinction must be stated every time it is used.
3. **Reproducibility is a deliverable, not hygiene.** If a result cannot be
   recomputed from archived artifacts, it does not count as established.

### What the CNIL actually said, and what it obliges us to do

[`cnil-opinion.md`](cnil-opinion.md) is authoritative. In summary **[P]**: in
August 2024 the CNIL's Technological Expertise Department reviewed the **Voice
Harbor solution**, not this framework, and identified no obstacle to that
solution demonstrating compliance with the three Opinion 05/2014 criteria. It is
a technical opinion, **not a certification**, and it does **not** transfer to a
deployer. This framework was the instrument of the demonstration. Describing it
as "validated" overstates the document, and this repository did so until it was
read.

The letter places four obligations on the demonstration, and each names a gap
we are accountable for closing. They are cited throughout as **G1** to **G4**:

| | Obligation | Status |
|---|---|---|
| **G1** | Identification from **metadata and verbal content** is a "strong assumption" the CNIL says must not be taken for granted | **[?]** Nothing here measures it. Idea 13. |
| **G2** | Re-identification risk must be quantified against a **concrete attacker**, accounting for the diversity of samples available to it and its capability | Partly met. We report population and attacker model, and saturation at `N = 29` is why idea 11 exists. |
| **G3** | **Pseudo-voices** must not make it possible to identify an individual | **[?]** Nothing here measures it. Idea 14. |
| **G4** | The demonstration must be kept **at the state of the art** as attacks evolve | Partly met. The commensurability programme is this, but there is no standing protocol. Idea 15. |

G2 is worth dwelling on: it is a data protection authority **requiring** the two
axes this project reports. It is the regulatory basis for refusing to quote a
privacy number without a population size and an attacker model, and it turns
Finding 3.1's saturation result from an inconvenience into a compliance problem.

### Our VPC setup, audited against the plan

Checked against the VoicePrivacy 2026 evaluation plan, version 2.0:
[`vpc-2026-conformance.md`](vpc-2026-conformance.md). The attacker
architecture, the semi-informed condition and the mixed trial lists conform.
Three things do not, and each bounds a claim:

- Our attacker was **labelled ignorant and is lazy-informed** **[M]**. The
  numbers are unchanged; what changes is that they come from a stronger
  attacker than the label implied, and that the plan collects lazy-informed EER
  for analysis rather than ranking. The ignorant condition has never been
  computed here.
- **B4, the neural audio codec baseline, was never run**, dropped at 188 days
  projected **[M]**. "Across the VPC 2026 baselines" means three of four.
- **IEMOCAP is excluded** and the two utility metrics are not run, so nothing
  here speaks to the privacy and utility trade-off.

### What would falsify the project's premise

If the EDPB criteria turned out to be measurable only under assumptions no real
deployment satisfies, or if a regulator rejected the operationalisation, the
framework would be a modelling exercise. The CNIL opinion **[P]** is what
currently stands against that, and it is the single most load-bearing external
fact we rely on, which is exactly why its scope is stated precisely rather than
favourably.

---

## 2. Non-negotiables

| Rule | Why | Enforced by |
|---|---|---|
| Every value carries a spread | Two systems differing by less than the combined standard error have not been shown to differ | `examples/04_*`, tie-aware ranking |
| No em or en dashes | House style | `scripts/check_no_dashes.py`, CI |
| No PII, ever, in the public repo | A privacy paper leaking personal data is self-refuting | `scripts/check_no_private_material.py`, CI |
| Tests pin against oracles, not current behaviour | A test asserting what the code does today locks in its bugs | theoretical baselines, brute-force oracles |
| Core works with numpy alone | Anyone can reproduce without a GPU stack | imports behind error messages |
| Deviations from the original experiment are documented | A reader comparing to the paper must know which version produced which number | `docs/differences.md` |
| Expensive intermediates are cached and never deleted casually | Recomputation is the largest avoidable cost in this project | section 7 |

### Traceability

Every number in a document must be reachable: a script that produces it, a log
that contains it, or an archived artifact that holds it. The chain for the
published results is: code on PyPI, score matrices under
[10.5281/zenodo.23142030](https://doi.org/10.5281/zenodo.23142030), and
`examples/03_verify_against_score_matrices.py` recomputing 2,640 published
points from them. Mean absolute difference 0.0007, worst 0.00509 **[M]**.

---

## 3. What we have established

### 3.1 The metrics

**SRD's identification rate is the Linkability metric.** Not approximately. At
`N = 29`, where the closed-set draw is deterministic, they agree to `0.0000` in
all six VPC conditions **[M]**. It is a definitional identity, pinned by
`tests/test_comparison.py`. A comparison reporting both as independent evidence
double counts.

**LID's Platt intercept and prior term cancel in the softmax**, leaving the
fitted slope as the only calibration parameter that reaches the reported bits
**[M]**, so LID is far more sensitive to one number than its three-term
definition suggests.

**Probability metrics fall with population size, bit metrics rise** **[M]**.
Structural: a larger search space makes success rarer and each success more
informative. Neither substitutes for the other, so report one of each.

**No metric inverts against another.** Across six dataset and attacker
combinations, where two metrics both separate a pair they order it the same way
**[M]**. The metrics are not rivals about ranking.

**They differ in resolution.** Of 10 system pairs: Linkability 9, SRD 9,
1-EER 9, LID 8, Singling Out 7 **[M]**. Our own Singling Out resolves fewest,
because its range is compressed between `exp(-1)` and 1 **[I]**.

**LID is a joint property of the system and its calibration.** On `libri_test`
the original-speech condition is perfectly separable, so the logistic fit has no
finite maximiser; ours stopped at `w = 199.4` and ALID then equals `log2 N`
exactly at every population size **[M]**. Evaluate B5 against that calibration
and it reports -207.3 bits where its own gives 1.561, against a 4.858 ceiling
**[M]**. This is a specification gap, not a defect in the idea.

**The evaluation set is the binding constraint.** At `N = 29`, four of six VPC
conditions sit against the bound; Linkability for original `libri_test` speech
is exactly 1.000 at every `N` **[M]**. A saturated metric cannot invert against
anything, so the "no inversions" finding is weak evidence until it is retested
on a population where every metric is off its ceiling **[I]**.

### 3.2 A correction we made, and why it is in this document

The resolution finding above was initially the opposite. With the spread taken
from the closed-set draw alone, LID appeared to be the only metric separating B2
from original, and that was written up. At `N = 29` the closed-set draw is
deterministic, so that spread was exactly `0.0000` and every difference looked
resolved. The same flaw manufactured an EER-versus-LID inversion. A trial
bootstrap removed both **[M]**.

**A standard error of exactly zero is a signal that the resampling is not
capturing the right source of variation.** This is now a standing check.

### 3.3 Our LibriSpeech numbers are bounded by content leakage

Every Linkability, Singling Out, SRD and LID value we report on `libri_dev` and
`libri_test` is measured on a corpus where **speakers are identifiable from
their vocabulary alone**. Franzreb, Das, Polzehl and Möller, arXiv:2601.13107
(2026) **[P]**:

> the books read by the Librispeech speakers are so distinct, that speakers can
> be identified by their vocabularies. Even perfect anonymizers cannot prevent
> this identity leakage.

The metrics are not wrong: they measure the voice signal and that is what they
claim to measure. What does not follow is the further claim that a system
*anonymizes* those speakers, because an attacker with the transcript does not
need the voice. This is CNIL obligation **G1** demonstrated on the exact corpus
the whole VoicePrivacy line uses, and it is the strongest external evidence
that G1 is a real gap rather than regulatory caution **[I]**.

It also supplies an argument for Common Voice we had not made. LibriSpeech
leaks because each speaker reads a distinct book; Common Voice speakers read
short sentences from a shared pool, so per-speaker vocabulary should be far less
distinctive **[?]**. That is testable with their method, and if it holds it
justifies the CV27 subsets on firmer ground than population size.

### 3.5 The third criterion, and how it was missing

Opinion 05/2014 sets three tests: singling out, linkability and **inference**.
This project implements the first two and has never implemented the third
**[M]**, by inspection of its own code. The CNIL letter concerns all three, so
the demonstration it reviewed rests on inference being handled somewhere, and
nothing here handles it.

This is a larger gap than any of G1 to G4 and it sat unstated while the
repository was named for the criteria. It was found by a venue scan, not by
reading our own scope.

**Membership inference is the wrong instrument**, and this was settled by
reading the Opinion rather than by experiment. It asks about deducing *the value
of an attribute*; membership inference asks whether a record was present, which
is nearer to singling out. A system can resist one and leak the other in either
direction **[I]**.

`legal_eval.inference` now implements **attribute inference** instead: a linear
probe for a sensitive attribute on features from the released artifact, reported
as the share of available headroom above the population prior, so 0 is no
disclosure and 1 is full determination **[M]**.

Prior art, and the method follows it: Noe, Mohammadamini, Matrouf, Parcollet,
Nautsch and Bonastre, Interspeech 2021, introduce attribute-driven privacy and
measure concealment with a classifier **[P]**; Vali and Backstrom, Interspeech
2024, show vector-quantized bottlenecks leak through codebook occupancy, which
independently supports section 4.3 **[P]**; Feng, Peri and Narayanan,
Interspeech 2022, name the attack **[P]**.

Two things are ours rather than theirs. The measurement is on the **released
artifact** rather than the internal representation, which matters because
vocoder drift means those differ. And it is reported **against the prior**,
because a raw accuracy is not interpretable: 71% on an attribute whose majority
class is 70% is nothing.

**Maturity, stated plainly.** This is the least validated thing in the package.
A linear probe gives a lower bound, so a null result is weak evidence of safety,
and demographic labels are self-reported and noisy. It should not carry the same
weight as Singling Out or Linkability until it has been run on real data and
compared against a stronger attacker.

### 3.6 Released artifacts are more revealing than they look

The released score matrices are **exactly rank 256**, the x-vector dimension:
the singular spectrum collapses by four orders of magnitude immediately after
index 256, and a rank-256 reconstruction reproduces the released scores to
within 2e-07 **[M]**. So a score matrix is an inner-product representation of
the embeddings, not a summary of them. "Similarities are safer than vectors" is
false at this shape **[I]**.

### 3.7 The attacker we measure preprocesses one utterance two ways

The 2026 challenge's lazy-informed attacker is `ecapa_ssl`: an ECAPA head over two
feature streams, a filterbank and WavLM-Large. Reading its extractor to drive
it over Common Voice turned up something about how it is fed **[M]**.

The filterbank stream is computed from the signal the caller decoded and passed
through `normalize_wave`, which loudness-normalises it. The WavLM stream is not.
`extract_vector` receives that same decoded audio and ignores it, calling
`extract_features([wav_path])`, which re-opens the file from disk and applies
its own layer norm. So the two streams that the ECAPA head combines see
differently preprocessed versions of one utterance, and every clip is decoded
twice.

Two consequences, and they pull in opposite directions.

It does not bias the comparison. The same attacker is applied to every
condition, so a difference between systems is still a difference between
systems. This is not a reason to discount any number already reported.

It is a caution for Common Voice specifically **[?]**. Loudness normalisation
is doing real work on one stream and none on the other, and Common Voice clips
are far more varied in level and channel than LibriSpeech read speech. If the
CV27 numbers come out oddly, this asymmetry is the first thing to check, and
the check is cheap: extract once more with the WavLM stream fed the normalised
audio and see whether the ranking moves.

What we must not do is quietly fix it. Changing which audio reaches which
stream changes every embedding, so the libri numbers already reported would no
longer be comparable to anything computed afterwards. If it is worth changing
it is worth re-running everything and saying so, which is the rule in
[`../CLAUDE.md`](../CLAUDE.md) about anything that moves a computed value.

### 3.8 The CV27 embeddings are not an artefact of how the work was divided

Common Voice is large enough that extracting its embeddings had to be
parallelised, and the challenge extractor's own parallel path does not run for
this attacker: it pickles each worker's result and torch refuses a non-leaf
tensor that requires grad. So extraction was sharded at the process level
instead, by crc32 of the utterance id, with each shard its own single-process
job and the results merged.

That is a change to how a published number is produced, so it was checked
rather than assumed. The same 1200 utterances were extracted twice, once in a
single pass and once as six shards merged back together. The embeddings are
**bit-identical: worst absolute difference 0.000e+00 across all 1200, and zero
speaker label mismatches** **[M]**. The check was repeated after limiting torch
to two threads per shard, which doubled throughput, with the same result
**[M]**.

This matters because the alternative is invisible. If sharding perturbed the
embeddings, every CV27 metric would shift by an unknown amount and nothing in
the output would indicate it. The numbers would simply be wrong, and wrong in a
way that looks like a finding about Common Voice.

The merge refuses rather than guesses, for the same reason: an identifier in
two shards, a total below what was expected, an incomplete shard, two
embedding dimensions. Each of those changes the population, and every metric's
chance level is a function of the population, so a short merge does not look
like an error either.

### 3.9 A destroyed utterance looks exactly like perfect privacy

Running the 2026 McAdams baseline over Common Voice 27 produced, for
`common_voice_en_25475281`, a file of the correct length that is not speech
**[M]**:

| | samples | rms | audible samples |
|---|---|---|---|
| original | 96768 | 0.0376 | 93487 |
| anonymized | 96768 | 0.0096 | 67 |

The output is 99.93% exact zeros with 67 near-full-scale impulses. The attacker
raised on it, which stopped a shard after 18063 utterances and is how we found
out.

**The crash is the fortunate outcome.** Consider the ordinary one. A more
forgiving extractor embeds six seconds of near-silence, that embedding sits at
chance distance from every enrolled speaker, and the utterance is counted as a
privacy success. Chance is the best privacy score attainable, so **a system
that destroys an utterance is rewarded for it**, by exactly as much as a system
that anonymized it perfectly.

This has three consequences, in increasing order of importance.

First, our own CV27 numbers need the exclusion declared. One utterance in
195580 is immaterial to any estimate, but the principle is not about this
magnitude.

Second, it bounds what a privacy figure means. A reported privacy level is an
upper bound on what the system achieves on audio it processes correctly, and
the gap is unmeasured unless somebody looks.

Third, and this is the part worth publishing: **no evaluation in this field
checks it.** The VoicePrivacy protocol scores whatever the anonymization stage
emits. A baseline that fails on a fraction of real crowdsourced audio, as
distinct from read audiobook speech, would be scored as more private for doing
so, and the leaderboard would not show it. Nobody is cheating; the protocol
simply cannot tell the two apart. That is a defect in the measurement, which
is this project's subject.

`legal_eval/audio_integrity.py` implements the check, with thresholds set
loosely enough that heavy but legitimate processing passes: a system is allowed
to sound bad and several good ones do. `preflight`'s `exclusions_declared`
requires a report to answer the question one way or the other, because silence
on it is the only answer that cannot be audited.

We must not fix the attacker's crash. VPC's model calls `layer(x,
lengths=lengths)`, the installed speechbrain's `TDNNBlock` does not accept
`lengths`, and a fallback path then fails on a one-frame feature tensor.
Repairing that would change what the attacker computes for every utterance and
would move every number already reported. The utterance is skipped and written
down instead.

### 3.10 The third criterion is untouched by a system that moves the second

The inference metric has now been run on real data for the first time, on the
CV27 subsets, predicting reported gender from the released ASV embeddings with
one row per speaker **[M]**:

| condition | prior | accuracy | advantage | female recall |
|---|---|---|---|---|
| original, 6 utterances per speaker | 0.7195 | 0.9119 | +0.6859 | 0.8363 |
| McAdams, 6 utterances per speaker | 0.7195 | 0.9136 | +0.6919 | 0.8423 |
| original, 20 utterances per speaker | 0.7220 | 0.9334 | +0.7604 | 0.8771 |

5879 of 10000 enrollment speakers report a gender, so the measurement runs on
those and the prior is 0.72. Advantage is headroom above that prior, so 0 means
the attacker did no better than naming the majority class.

**McAdams moved the advantage by +0.006.** That is not a reduction, and it is
inside the noise of a five-fold estimate. The same system moves Linkability on
LibriSpeech from 0.955 to 0.914 **[M]**. So it does something, modest, to
identification, and nothing whatever to attribute disclosure.

This is the paper's own thesis extended, and it is the argument for why the
third criterion had to be implemented rather than noted. Article 29 asks three
questions, and a system can answer one of them partially while leaving another
entirely open. A controller reporting Singling Out and Linkability alone would
report that McAdams reduces risk. On inference it reduces nothing, and 84% of
the women in the population are still correctly identified as women from the
anonymized embedding.

Two cautions, because the number is strong and will be quoted.

The label is self-reported and binary, and absent for 41% of speakers. The
excluded speakers are excluded because "not reported" is a property of an
optional form field rather than an attribute of a person; this is argued in
[`cv27-datasheet.md`](cv27-datasheet.md). Treating it as a third class would
have raised accuracy while measuring something else.

The features are the attacker's embeddings, which are trained for speaker
discrimination. Gender is heavily entangled with speaker identity, so this
result says that an embedding useful for verification carries gender. It does
not establish that a system which defeated verification would also defeat
gender inference. That is the interesting open question and the right next
experiment is the ASR-BN condition, which moves Linkability to 0.386 **[?]**.

---

## 4. Where identity actually leaks

This section exists because design discussions keep reaching for the wrong knob.
It is the pedagogical core and it is deliberately opinionated about what the
measurements support.

### 4.1 The pipeline, and the three places identity survives

A modern anonymizer is three stages:

```
speech -> [encoder] -> content representation
                       + speaker representation -> [pseudo-speaker] -> target
                       + prosody (F0, duration, energy)
                                                   -> [vocoder/TTS] -> speech
```

Identity can survive in all three. The usual assumption is that replacing the
speaker representation is sufficient. It is not.

### 4.2 Spectral envelope is not where identity lives

The sharpest evidence we have. McAdams coefficient warping modifies the spectral
envelope and nothing else. Against the lazy-informed attacker at `N = 29`
**[M]**:

| | Linkability | SRD MeanD | 1-EER |
|---|---|---|---|
| original | 0.9552 | 4.6018 | 0.9655 |
| B2 McAdams | 0.9138 | 4.3894 | 0.9665 |
| B5 ASR-BN | 0.3862 | 2.3338 | 0.7670 |

McAdams barely moves any metric, and `1-EER` even ranks it marginally *worse*
than original. A signal-processing transform of the envelope leaves the speaker
recoverable. ASR-BN, which replaces the representation, is the only baseline
with real range **[M]**.

**Pedagogically:** if a method's story is "we changed how the voice sounds", it
is probably not an anonymizer. The attacker does not listen.

### 4.3 Encoder and content tokens

Content representations are trained to predict text, not to discard speakers,
and the two objectives are not opposed enough for the second to happen for free.

- **ASR bottleneck features** retain speaker information in coarticulation,
  speaking rate, and phone duration patterns. Our B5 numbers are the evidence:
  EER 23.3%, not 50% **[M]**, so a third of the identity signal survives a
  representation explicitly designed to carry phonetics.
- **Vector quantisation** reduces capacity but does not target speaker
  information. A codebook large enough for intelligibility has room for
  speaker-correlated allocation **[I]**.
- **SSL representations** (wav2vec 2.0, HuBERT, WavLM) are known to be strongly
  speaker-discriminative in their upper layers, which is why speaker
  verification systems use them **[P]**. Using them as "content" requires an
  argument about which layer and what was removed, not an assumption.

**[?]** We have not measured how much identity survives quantisation as a
function of codebook size. This is research idea territory, and it is the kind
of question where a measurement would settle a lot of hand-waving.

### 4.4 Pseudo-speaker selection

This is the part most often treated as a detail and is in fact an evaluation
confound. **The CNIL flagged it independently** (**G3**): a pseudo-identity is
understood to be a "mix" of several people, and the letter asks us to ensure it
"does not make it possible to identify an individual".

**Assignment granularity decides whether linkability survives.** If a
pseudo-speaker is assigned per source speaker, every utterance of that speaker
maps to the same target, so the utterances remain **mutually linkable** even
though the identity is concealed. That is precisely the EDPB linkability
criterion, and the EER cannot see it, because the EER is computed over trials
rather than over the structure of the assignment **[I]**. VPC's configs expose
both granularities (`anon_level_spk` and `anon_level_utt`), so the experiment is
cheap.

Other failure modes worth naming:

- **Pool collision.** Drawing targets from a finite pool means two source
  speakers can receive near-identical targets, which creates false links.
- **Proximity to a real speaker.** A GAN-sampled embedding is not guaranteed to
  be far from every real speaker. `num_sampled: 5000` with `sim_threshold: 0.7`
  in the sttts config is an explicit attempt to manage this, which tells you the
  designers considered it a live risk **[P]**.
- **Consistency versus privacy.** Per-utterance assignment reduces linkability
  and damages downstream usability for anything that needs speaker continuity.
  There is a real trade-off here and it should be stated, not resolved by
  default.

### 4.5 Vocoder and prosody

The vocoder is the last place identity can re-enter, and the literature is
ahead of us on both routes. Both points below were written here as inference and
are now replaced by citations.

**1. Prosody is identity-bearing, and the fix is to perturb it, not to pass it
through.** Meyer, Lux, Koch, Denisov, Tilli and Vu, "Prosody Is Not Identity: A
Speaker Anonymization Approach Using Prosody Cloning", ICASSP 2023,
doi:10.1109/icassp49357.2023.10096607, open with the premise **[P]**:

> Prosody is closely linked to the identity of a speaker, leading to individual
> pitch and intonation patterns.

The title is a claim about what their *method* achieves, not a denial of that
link. Their contribution is to clone the prosodic structure while breaking
identity, by multiplying pitch and energy sequences by random offsets, reporting
high privacy together with high pitch correlation.

So the assertion here was right and its framing was wrong: prosody is not a
channel you can leave open, and it is also not one you have to discard. The
VoicePrivacy `sttts` baseline already implements their method, which is what the
`anon_prosody/random_offsets` directory in its intermediates is **[M]**. We have
been running their fix without citing it.

**2. The vocoder does not faithfully render the embedding it is given, and the
error is reversible.** Panariello, Todisco and Evans, "Vocoder drift in
x-vector-based speaker anonymization", Interspeech 2023,
doi:10.21437/interspeech.2023-448 **[P]**. Vocoder drift is the difference
between the x-vector handed to the vocoder and the one recoverable from its
output. Their findings: the drift is **learnable and therefore reversible by an
attacker**, its impact is "substantial and sometimes dominant" relative to the
anonymization function itself, and privacy may therefore be "weaker than
previously thought".

This is a sharper and more consequential mechanism than the one asserted here
before, which was about the vocoder's training distribution narrowing output
diversity. The real risk is not that the vocoder is biased, it is that the map
from intended target to rendered target is systematic, so an attacker who learns
it can undo part of the anonymization. It also means a pipeline that selects a
pseudo-speaker carefully is not thereby delivering it.

**Consequence for our own work.** Our G3 audit in
[`../legal_eval/pseudo_speaker.py`](../legal_eval/pseudo_speaker.py) compares
pseudo-speaker embeddings against real speakers. Vocoder drift says the
embedding the pipeline *chose* is not the one the audio *carries*, so the audit
should run on x-vectors extracted from the rendered audio, not on the intended
targets. The run reported in section 3 did use attacker-space x-vectors from
rendered audio, so it is on the right side of this, but by accident rather than
by design **[M]**.

**[?]** We have not isolated prosody's contribution by ablation (anonymise
embedding only, versus embedding plus prosody, on identical content). That is a
clean, cheap experiment and we should run it.

---

## 5. Research directions, and the alignment test

The twelve ideas live in [`docs/research-ideas.md`](research-ideas.md). This
section is the filter, not the list.

**A proposal earns time only if it passes all four:**

1. **Does it change what a controller would conclude?** If the answer is a
   number no decision depends on, it is a paper, not a contribution.
2. **Is it measurable on an artifact we have or can build?** If it needs data we
   cannot get, it is a proposal for someone else.
3. **Can it conclude against us?** A study designed so our metric wins is
   advocacy. Findings 3.1 and 3.3 reduce our own standing, which is what makes
   the rest credible.
4. **Does it avoid competing with a PSST fellow who owns it?** Convening beats
   duplicating. We dropped per-speaker risk analysis for exactly this reason.

### Current priorities, in order

Ordered by whether a regulator has already asked for the answer. A **G**
reference means the CNIL named it.

1. **Pseudo-voice identifiability** (idea 14, **G3**). The CNIL asked for this
   by name and we do not measure it. It is also cheap. Highest priority.
2. **Metric commensurability on a population with range** (idea 11, **G2**).
   Blocked on the evaluation set, which is why the Common Voice work sits in
   front of it.
3. **Content and metadata leakage** (idea 13, **G1**). The CNIL calls the
   current position a "strong assumption". Larger effort, and it needs scoping
   before it needs resourcing.
4. **A standing re-assessment protocol** (idea 15, **G4**). Cheap to specify,
   and it converts a one-off demonstration into something with a shelf life.
5. **The LID calibration specification.** Needs its authors. Our role is the
   measurement and the question, not the answer.
6. **Subset design correction.** Implemented; needs writing up.

### Diversion warnings

Named explicitly, because each has already tempted us:

- **Building anonymization systems.** Out of scope. We evaluate.
- **Chasing a thirteenth metric.** Section 3.1 says the field does not need one.
- **Optimising infrastructure past the point of need.** See section 7: the test
  is whether a measurement is blocked, not whether a number could be larger.
- **Reproducing numbers nobody disputes.** Verification is worth doing once.

---

## 6. Scientific correctness discipline

Applied at every step, not at review time.

1. **State the marker.** [M], [P], [I] or [?]. An [I] presented as [M] is the
   most common serious error.
2. **Quote the uncertainty.** And if it is exactly zero, find out why before
   believing it.
3. **Prefer an oracle to a snapshot.** Hypergeometric Linkability is checked
   against literal candidate-set sampling; ROCCH-EER against brute-force search
   over randomised rules.
4. **Name what would falsify the claim.** Every finding in section 3 has this.
5. **Report the correction, not the corrected version only.** Section 3.2 exists
   because a reader who cannot see our errors cannot calibrate our claims.
6. **Distinguish coarsening from contradiction.** Two metrics disagreeing
   because one has not resolved a pair is not a disagreement.

---

## 7. Compute, cost and caching

**The test for optimisation work: is a measurement blocked or delayed past
usefulness?** If not, stop.

### What we know about this machine **[M]**

| | |
|---|---|
| GPU | A100-SXM4-40GB |
| Observed during sttts | 0% util, 3.5/40 GB, 56/400 W |
| CPU | 12 cores, 0.73 busy |
| iowait | 0% |
| Filestore throughput | ~18 MiB/s sequential read |
| Per-worker cost | 3.5 GB GPU, 0.73 cores |

Sharding to 8 workers fits in both budgets **[I]**, and the rule is implemented
and tested (`legal_eval/sharding.py`, `docs/gpu-utilisation.md`).

### Caching is a first-class concern

The single largest avoidable cost here is recomputing intermediates. The VPC
pipeline caches per utterance and per dataset, and `save_intermediate: true`
with every `force_compute_*: false` is the configuration that makes a 16 hour
run resumable rather than restartable.

**Rules:**

- Never delete `exp/anon_pipeline_*` to "clean up". It is 31 GB of
  `anon_speech` and hours of GPU time.
- Precomputed intermediates that upstream distributes are a gift. Prefer
  `download_precomputed_intermediate_repr: true`.
- Before any long run, confirm it will resume rather than restart.
- Archive embeddings and score matrices, not audio. A score matrix is the
  sufficient statistic for EER, Linkability, LID and SRD. Singling Out
  additionally needs calibration conversations, which is why a score matrix
  alone cannot produce it.

### Decisions already taken on cost grounds **[M]**

- **nac dropped.** 496 of 104,014 utterances in 21.5 hours, about 188 days.
  VPC's own code calls the loop barbaric.
- **float32 for released matrices.** Exactly lossless here, max error 0.000e+00,
  10.5 GB to 5.2 GB.
- **Hypergeometric Linkability.** Identical distribution to candidate-set
  sampling, hours to seconds.

---

## 8. What this document needs from Brij

Marked **[?]** because it is not inferable from the repository, and writing it
from general knowledge of the field would be exactly the fabrication this
document forbids.

1. **The thesis, and the ICASSP 2020 pseudo-speaker paper.** Section 4.4 is
   currently reasoned from the VPC configs and our own measurements. The actual
   trajectory, what was tried, what failed, and which conclusions have since
   been overturned, should come from the sources, not from me.
2. **Which claims from 2020-2021 you now consider wrong.** The most valuable
   content in a document like this is the superseded belief, and only you know
   which ones they were.
3. **The 2026 v3 system paper.** It is referenced as the companion to the VPC
   baseline ranking work, and I have not read it.
4. **Any CNIL correspondence that bounds the validation.** Section 1 leans on
   it, and the precise scope of what was validated matters.

Until those arrive, sections 4.3 to 4.5 should be read as the literature plus
our measurements, not as a settled account.
