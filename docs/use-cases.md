# Use cases, and why the evaluation depends on them

> Claims carry the markers used throughout this repository: [M] measured here,
> [P] published elsewhere, [I] inferred, [?] open.

A privacy evaluation that does not say what the data is for has made an
assumption, and the assumption is almost always wrong. It has assumed that
every attribute of the speaker is a leak and that nothing about the speech
needs to survive. No deployment looks like that. Under that assumption a system
that outputs silence scores perfectly, which is the same failure as rewarding a
destroyed recording, recorded as finding 3.9.

This document is the other half of the framework. The metrics say how much
survives. The use case says which of it was supposed to.

## Why utility is not the secondary axis

The usual framing treats privacy as the objective and utility as a constraint
to be traded against it. The deployment incentive runs the other way, and the
mechanism is worth stating because it decides whether any of this gets used
**[I]**, from commercial experience rather than from measurement.

A data controller holding sensitive speech has three options: keep it and carry
the liability, anonymize it and reuse it, or delete it. Anonymization competes
with deletion, not with keeping the original. So a method that protects well
and leaves the data unusable does not get deployed badly. **It does not get
deployed at all, and the data is deleted instead.** The controller loses
nothing they were going to use, and the research that produced the method
protects nobody, because the recordings it would have protected no longer
exist.

That reverses the priority order. A privacy method's utility cost is not a
concession to industry. It is the condition under which the privacy is ever
realised. It also explains why heavy noise addition, which is strong on paper,
has not been adopted for speech: the data it protects is data nobody can use.

The consequence for this framework is that a privacy number quoted without its
utility cost is not conservative. It is uninformative about whether the system
would ever be used, and a system that is never used provides no protection at
all.

## The point that reorganises everything

**In many real deployments, an attribute that the third Article 29 criterion
treats as disclosure is the reason the data was collected.**

A call centre analysing why customers are angry needs emotion to survive
contact with the anonymizer. A fairness audit of a speech recogniser needs
accent and age to survive, or it cannot measure the disparity it exists to
measure. A pathology screening corpus needs the pathology.

The measurement does not change. It is the attribute inference advantage from
`legal_eval.inference`, the share of available headroom an attacker captures.
What changes is the sign of its interpretation, and nothing in the number tells
you which sign to use. So `legal_eval.tradeoff` refuses to score a system
without a declared use case, and the declaration has to carry a justification
per attribute, because an unjustified role is how a convenient one gets
assigned.

This is why "anonymize the speech" is not an instruction that can be checked.
"Remove the speaker's identity while keeping emotion recoverable, and show both
with their chance levels" is.

## The second point: linkability is not monotonically bad either

A voice assistant that adapts to its user needs the same speaker to map to the
same pseudo-identity every time, or adaptation has nothing to adapt to. That is
a requirement for **within-speaker linkability** held simultaneously with a
prohibition on **linkability to the real person**. The two are different
questions and the field's single Linkability number does not separate them.

The VoicePrivacy pipelines expose exactly this knob, as `anon_level_constant`,
`anon_level_random_per_spk` and `anon_level_random_per_utt` [M]. A deployment
that needs adaptation wants per-speaker assignment. An evaluation that wants
the hardest case for an attacker wants per-utterance, which is what this
project's ASR-BN run on Common Voice used [M]. Quoting a privacy number without
saying which was used is quoting a number from a different experiment.

## The matrix

What each column means. **Must survive** is the utility requirement, and a
system that removes it has failed the deployment however private it is. **Must
go** is prohibited under the criteria. **Binds hardest** names the criterion
that is the real constraint, which is rarely the one the field measures.

| Use case | Must survive | Must go | Binds hardest | Measure |
|---|---|---|---|---|
| Call centre sentiment analytics | Emotion, prosody, content | Caller and agent identity | Linkability, against a strong attacker: the employer holds enrolment data | Linkability at the real population, plus emotion advantage retained |
| Agent quality monitoring | Content, speaking rate | Caller identity; agent identity is often required by the employer | Two different subjects in one recording, with different roles | Separate measurement per speaker role |
| Clinical dictation and archives | Verbatim content, including drug and condition names | Patient identity | **G1, verbal content.** The patient's name is often spoken aloud, so the identifier is inside the thing that must survive | Content-based re-identification, not voice-based |
| Pathology and screening corpora | The pathology itself, which is Article 9 special-category data | Patient identity | Inference, inverted: the required attribute is the most sensitive thing present | Pathology advantage retained, identity at chance, and a stricter release control |
| Fairness audit of a speech system | Accent, age, gender, the protected attributes | Individual identity | Inference, inverted. Removing the attribute destroys the audit | Attribute advantage retained per group, identity at chance |
| Public research corpus release | Content, acoustic diversity | Identity, and attributes not needed by the stated research | Singling out and linkability against an **unbounded** attacker: anyone may attack, forever | The full population sweep, and the worst individual, not the mean |
| Voice assistant training | Content, acoustic diversity | Identity | Linkability, with the within-speaker requirement above | Both linkability questions, separately |
| Assistive and accessibility personalisation | Speaker-specific acoustics, by design | Linkage to the real person outside the device | Pseudo-speaker identifiability, obligation G3 | `legal_eval.pseudo_speaker`, and whether the pseudo-voice resembles a real one |
| Market and audience research | Demographics for segmentation | Identity | Inference, inverted, and the attribute is demographic | Segment-level attribute retention, identity at chance |
| Forensic voice comparison | Identity, which is the entire purpose | Nothing | None. Anonymization does not apply | Included to mark the boundary: not every voice task is a privacy task |

## What this changes about the numbers

**An equal error rate cannot answer any row of that table.** It is a single
average over trials from one attacker, and every row needs either a population
it did not fix, a worst case it averaged away, an attribute it did not look at,
or a channel it does not cover.

**Two systems are comparable only when they are on one row.** A system tuned
for call centre analytics and a system tuned for corpus release are solving
different problems, and the privacy number that favours one is measured against
a different attacker.

**The three-point rule applies on both axes.** Report the chance level, the
anonymized value and the same measurement on the original recordings, for
privacy and for utility alike. Without the original, a reader cannot tell
protection from difficulty on one axis, or preservation from the attribute
never having been there on the other.

## Reading the two axes together

`legal_eval.tradeoff` will assert one ordering and refuse the rest. A system is
better than another only if it is at least as private and at least as useful,
with one strict. Anything else is a trade, and a trade is reported as a trade:

```python
from legal_eval.tradeoff import SystemPoint, report

points = [
    SystemPoint("mcadams", exposure=0.28, utility=0.94, utility_metric="1-WER"),
    SystemPoint("asrbn",   exposure=0.07, utility=0.88, utility_metric="1-WER"),
]
print(report(points).summary())
```

Those two are incomparable, and the summary says so rather than ranking them.
Ordering them is a judgement about how much utility a unit of privacy is worth,
which belongs to the controller and the regulator.

When a deployment states the utility it needs, the frontier can be interpolated
to that level and the systems compared at equal utility, which is the
comparison two separate tables cannot support. The interpolation assumes a
system can be tuned between its measured operating points, and that assumption
is about the systems rather than the arithmetic, so it is stated when the
result is quoted.

## Open questions this exposes

Each is a piece of work somebody could take.

- **Nobody has measured attribute retention and identity removal on one
  artifact across the challenge baselines** [?]. This repository measures the
  second and, since the Common Voice run, part of the first.
- **The within-speaker linkability requirement has no metric** [?]. It is a
  deployment requirement with no number, and the pipelines already have the
  knob.
- **Content-based re-identification is outside every current evaluation** [M,
  obligation G1], and for clinical speech it is the binding constraint rather
  than a secondary one.
- **The utility side has no commensurability work at all** [I]. This project
  found that nine privacy quantities were not comparable. Word error rate,
  mean opinion score, emotion recognition accuracy and speaker similarity are
  in the same position, and nobody has done for them what section B of the
  reference library covers for privacy.
- **Dual-role recordings are unaddressed** [?]. A call recording has two
  speakers with different legal positions, and every corpus in use here is one
  speaker per recording by construction.
