# Datasheet: the Common Voice 27 evaluation subsets

Following the structure of datasheets for datasets, and the speech-specific
argument in arXiv:2305.04672 that the field ships filelists and construction
scripts where it should ship datasheets.

This exists because [`subset-design.md`](subset-design.md) says to ship one,
and because a subset is a design decision. Choosing which speakers constitute
the population fixes every chance level in the evaluation, so the choices have
to be legible to somebody who did not make them.

All counts here are measured from the built subsets, not from the construction
script's intent **[M]**.

## Motivation

**Why were these subsets created?** The published Interspeech 2025 evaluation
used two Common Voice subsets defined by duration, on a release that is now
several versions old. Rebuilding on a current release is a chance to correct
three design defects, which `subset-design.md` states and implements. The
larger reason is that at the VoicePrivacy populations of 29 speakers, four of
six measured conditions sit against their metric's ceiling, so no metric can
separate systems there. A population with range is the only way to find out
whether the metrics agree.

**Who created them, and who funded it?** Nijta SAS, as part of the work behind
`doi:10.21437/Interspeech.2025-1699`, on its own compute budget.

## Composition

**What is an instance?** One utterance of read English speech, recorded by a
Common Voice contributor, with its contributor's pseudonymised identifier and
self-reported gender.

**How many instances?**

| Subset | Speakers | Utterances per speaker | Utterances |
|---|---|---|---|
| `cv27_A_enroll` | 10,000 | exactly 6 | 60,000 |
| `cv27_B_test` | 9,779 | exactly 20 | 195,580 |

Drawn from 2,591,456 clips in `cv-corpus-27.0-2026-09-11`, English.

Every speaker has exactly the stated number of utterances. This is the fixed
enrollment budget from `subset-design.md`: with a floor rather than a budget,
speakers arrive with anywhere from the floor to hundreds of utterances, and
part of the per-speaker spread being reported is then an artefact of the
sampling rather than a property of the system.

**How do the two subsets relate?** All 9,779 `B_test` speakers are among the
10,000 in `A_enroll` **[M]**. The remaining 221 had fewer than 20 utterances
left after the enrollment draw and so appear only as distractors. Utterance
overlap between the two is **exactly zero** **[M]**, which is the one property
of the design that is not negotiable: an utterance on both sides makes the
evaluation measure memorisation rather than linkage.

**What does each instance consist of?** Audio only, plus the speaker
identifier and reported gender. No transcript is used by the privacy metrics,
although Common Voice supplies one.

**Is any information missing?** Yes, and it matters:

| Reported gender | `A_enroll` speakers | share | `B_test` speakers | share |
|---|---|---|---|---|
| male | 4,230 | 42.30% | 4,185 | 42.80% |
| female | 1,649 | 16.49% | 1,611 | 16.47% |
| not reported | 4,121 | 41.21% | 3,983 | 40.73% |

**Read this before using gender as a label.** Gender is absent for about 41% of
speakers, and among those who do report it the split is roughly 72% male to 28%
female. An attribute inference attack scoring 72% on reporting speakers has
learned nothing: that is the majority class. This is why
`legal_eval.inference` reports headroom above the population prior rather than
accuracy, and why it reports the most exposed minority value rather than the
best overall recall.

"Not reported" is **not** an attribute value. The Article 29 inference
criterion asks whether an attribute of a person can be inferred, not whether
they completed an optional form field. The primary analysis therefore runs on
the 5,879 reporting speakers in `A_enroll` and 5,796 in `B_test`. Whether
declining to report is itself inferable is a separate and narrower question,
and conflating the two would inflate any accuracy figure with a label that is
about the platform rather than the person.

Age and accent are also self-reported in Common Voice and also sparse. They are
not yet used here.

**Does it contain confidential or sensitive data?** It contains voice, which is
biometric data under GDPR Article 9 when processed to identify someone, and
self-reported gender, which is personal data. The whole purpose of the
collection is to measure whether anonymization defeats identification, so it is
handled as personal data throughout.

**Are there identifiable individuals?** Yes. Speakers are real contributors and
the audio is their voice. Pseudonymisation does not change that, and
[`data-release.md`](data-release.md) states why the pseudonyms are not a
safeguard.

## Collection

**How was the data acquired?** Downloaded from the Common Voice 27.0 English
release. Contributors donated recordings to Mozilla's public corpus under CC0.
Nijta collected nothing.

**Was consent obtained?** Contributors consented to Mozilla's terms, which
place the recordings in the public domain. They did not consent to a privacy
evaluation specifically, because no such mechanism exists. This is a real
limitation and worth stating rather than glossing: a CC0 dedication is not
informed consent to every downstream use.

**Sampling strategy.** Speakers were ordered by available utterance count and
taken from the top until the enrollment target was met. This is a demographic
choice, not a neutral one: prolific Common Voice contributors are not a random
sample of speakers, and the resulting gender skew above is partly a
consequence. A random draw over eligible speakers would trade population size
for representativeness. The choice is recorded here so that a reader can
discount it.

## Preprocessing

**What was done?** Clips were selected by utterance count, speaker identifiers
were replaced by pseudonyms of the form `spk-<g>-<n>` where `<g>` is the
reported gender marker, and the 64-hex Common Voice `client_id` was discarded
and does not appear anywhere in the subsets **[M]**.

Audio was transcoded from mp3 to FLAC at 16 kHz mono. FLAC is lossless and the
transcode was verified sample-identical to decoding the mp3 in the pipeline, to
within half a 16-bit quantisation step, on sampled clips **[M]**. The reason
was throughput: the evaluation pipeline decodes each clip twice, and mp3 decode
dominated at 2.25 utterances per second against 11.88 for a plain file.

**Is the raw data available?** Yes, from Mozilla. Everything here is derivable
from the public release plus the construction script.

## Anonymized copies

McAdams anonymized copies of both subsets exist, produced by the VoicePrivacy
2026 baseline, with the same utterance counts.

**One utterance is excluded.** For `common_voice_en_25475281`, McAdams emitted
a file of the correct length that is 99.93% exact zeros with 67 near-full-scale
impulses, against 93,487 audible samples in the original **[M]**. The attacker
raises on it. Constitution section 3.9 explains why this matters far more than
one utterance in 195,580: a destroyed utterance scores at chance, chance is the
best privacy score attainable, so a system is rewarded for destroying speech by
exactly as much as for anonymizing it perfectly. The exclusion is recorded with
the embeddings and declared in every report, and `legal_eval.audio_integrity`
checks for it.

## Uses

**What is it for?** Measuring Singling Out, Linkability, attribute inference,
SRD and LID at a population large enough that the metrics are not pinned
against their bounds.

**What should it not be used for?** Training. The subsets are an evaluation
set and the enrollment and test sides are deliberately utterance-disjoint;
training on either destroys that. It is also unsuited to any claim about
population demographics, given the sampling strategy above.

**Is there anything that makes a use inappropriate?** Using the gender labels
as ground truth about people. They are self-reported, optional, absent for 41%
of speakers, and recorded in a binary scheme that does not describe everyone.

## Distribution

**How will it be distributed?** As a filelist of utterance identifiers plus the
construction script, not as audio. The audio is Mozilla's to distribute and the
identifiers are enough to rebuild the subsets exactly.

**Licence.** The construction script and filelists under the repository's MIT
licence. The audio remains CC0 from Mozilla.

**Any IP or restrictions?** None asserted by Nijta.

## Maintenance

**Who maintains it?** Nijta SAS, brij@nijta.com.

**Will it be updated?** Common Voice releases roughly twice a year and speaker
counts grow. A rebuild on a later release gives a different population, and so
different chance levels and different numbers. Any result has to name the
release it used, which is why the corpus directory name appears in the
provenance record rather than only "Common Voice".

**How will errors be communicated?** Through the repository's issues.

## Known limitations, collected

1. Speakers are selected by prolificacy, so the population is not
   representative and the gender skew is partly an artefact of that.
2. Gender is missing for 41% of speakers and binary where present.
3. Contributors consented to CC0, not to a privacy evaluation.
4. Read speech from volunteers, so accent, channel and recording conditions
   are broad but not controlled, and none of that variation is annotated.
5. English only.
6. One utterance is unusable after McAdams anonymization, and nothing
   establishes that other systems fail on the same one or on none.
