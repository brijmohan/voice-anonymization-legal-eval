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

### 3.3 Released artifacts are more revealing than they look

The released score matrices are **exactly rank 256**, the x-vector dimension:
the singular spectrum collapses by four orders of magnitude immediately after
index 256, and a rank-256 reconstruction reproduces the released scores to
within 2e-07 **[M]**. So a score matrix is an inner-product representation of
the embeddings, not a summary of them. "Similarities are safer than vectors" is
false at this shape **[I]**.

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
envelope and nothing else. Against the ignorant attacker at `N = 29` **[M]**:

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

The vocoder is the last place identity can re-enter, by two routes.

1. **Prosody passed through unmodified.** F0 contour, duration and energy are
   identity-bearing. If the pipeline anonymises the speaker embedding and feeds
   the original prosody to the vocoder, it has left a channel open. The sttts
   pipeline has an explicit `prosody_anonymization` stage, which is the right
   shape.
2. **The vocoder's own training distribution.** A vocoder trained on a
   particular speaker set imposes that set's characteristics, which can both
   leak the training speakers and reduce the effective diversity of outputs
   **[I]**.

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
