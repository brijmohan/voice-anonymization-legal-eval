# Research ideas and venues

Follow-on work this codebase makes cheap, roughly in order of how much is
already in hand. Each entry records the claim, what evidence exists today, what
is still missing, and where it might go.

Deadlines below are the usual windows for each venue, not verified dates.
Check the call before planning around any of them.

---

## 1. Do the VPC baselines rank differently under legal metrics?

**Claim.** The EER ranking of anonymization systems does not match the ranking
under Singling Out and Linkability, so challenge leaderboards are ordering
systems by a measure that does not track the legal criterion.

**Why it is the strongest idea.** It tests the Interspeech 2025 paper's argument
on systems the authors did not design, needs no new data collection, and the
answer is interesting either way. A ranking change is a direct result. No
ranking change but a much wider spread still says the EER compresses differences
that matter legally.

**Already in hand.** `legal_eval.vpc` reads the embeddings VPC's ASV step
already caches, so the metrics cost seconds per baseline. Four Track 1 baselines
have public configs and published results. Runbook in
[`vpc-benchmark.md`](vpc-benchmark.md).

**Partly done.** asrbn and mcadams are complete, both attacker conditions.
Early signal: mcadams leaves Linkability at 0.966 against a semi-informed
attacker where asrbn reaches 0.793, while `1 - EER` separates them by only 8
points against Linkability's 17 and Singling Out's 32. Our mcadams EER
reproduces VPC's published 5.617 exactly, so the pipeline matches theirs.

**Still needed.** sttts. nac was dropped: it anonymizes at 240 utterances an
hour, making `train-clean-360` an 18 day job, and VPC's own code calls the loop
"barbaric" and unbatched. It ranks third of four on EER, so the span from
strongest to weakest survives without it.

**Venue.** Interspeech (deadline usually February or March) or the ISCA SPSC
Symposium, which is the natural home for privacy-metric work and a friendlier
audience for a methodological result. ICASSP if the timing fits.

**Risk.** If the organizers adopt the metrics first, this becomes a challenge
report rather than an independent paper. That is a good outcome, just a
different one, so decide early which you want.

---

## 2. Released score matrices leak more than they appear to

**Claim.** Publishing an enrollment-by-test cosine score matrix is close to
publishing the embeddings themselves. The matrix is exactly rank `d`, the
embedding dimension, so it is an inner-product representation rather than a
summary, and similarity structure that was never released can be recovered from
it.

**Why it matters.** The speech privacy community routinely releases score files
as the safe alternative to embeddings. If that intuition is wrong, several
existing releases are more disclosive than their authors believe. This is a
short, sharp result with immediate practical consequences.

**Already in hand.** Measured on the paper's own 22,024 x 4,949 matrices: the
singular value spectrum collapses by four orders of magnitude immediately after
index 256, which is the x-vector dimension, and a rank-256 reconstruction
reproduces the released scores to 2e-07. Written up in
[`data-release.md`](data-release.md).

**Still needed.** The honest gap: a plain SVD does not complete the attack,
because the scale splits arbitrarily between the two factors. Recovering the
true geometry means solving for the transform that restores unit norms on both
factor sets, which is over-determined by the norm constraints but not yet
demonstrated. Doing that properly, and quantifying what an adversary gains at
each step, is the paper. Then survey what the community has already released and
check which releases are affected.

**Venue.** PoPETs has rolling quarterly deadlines and is the right audience for
a disclosure-mechanics result. IEEE TIFS for a longer treatment. SPSC for the
speech-specific framing.

**Note.** Do this carefully and responsibly. If existing public releases are
affected, contact those authors before publishing.

---

## 3. What reproducing a privacy evaluation actually costs

**Claim.** Privacy metrics are reported to three decimal places from pipelines
whose published error bars understate their true variance, and the field has no
convention for reporting which version of a metric produced a number.

**Already in hand.** Three concrete defects found while reimplementing one
paper's own code, all documented in [`differences.md`](differences.md): parallel
workers sharing one RNG state so five "independent" runs were not independent
(visible in the published JSON, where two runs agree to 16 digits); a
cross-validation loop seeded on speaker position only, so all ten folds reused
one split; and `random.sample` on a set, which has raised `TypeError` since
Python 3.11, meaning the original scripts no longer run at all.

Also in hand: a closed-form replacement for the Linkability sampling loop that
is identical in distribution but collapses a multi-hour sweep to under a second.

**Still needed.** Either broaden it into a survey of reproducibility in the
voice privacy literature, which is a lot of work and makes enemies, or keep it
narrow and constructive as a short methods paper about what a reproducible
privacy evaluation needs: versioned metrics, seeded independence, and published
intermediate artifacts.

**Venue.** The narrow version fits an Interspeech short paper or SPSC. The broad
version fits Computer Speech and Language. A reproducibility track, if one is
running, is the best fit of all.

**Caution.** The narrow, constructive framing is the one to write. Tone matters
here: the point is what the field should standardise, not whose code had a bug.

---

## 4. The inference criterion

**Claim.** Opinion 05/2014 names three criteria. The Interspeech 2025 paper
operationalises two and explicitly sets inference aside, arguing that building
an inference attack on speech would need an attacker holding an attribute table
for all enrollment speakers, which it calls unrealistic today.

**Why revisit it.** That argument is about four years old in a fast-moving area.
Speech foundation models now predict age, accent, health markers and emotional
state from short utterances with no speaker enrollment at all. The premise may
no longer hold, and the paper's own conclusion lists this as future work.

**Still needed.** Everything. This is the largest and most original of these
ideas: a quantitative inference metric for speech that fits the same legal
framework as the other two, plus an argument for which attributes count.

**Venue.** Interspeech or ICASSP for the metric, a journal for the full
framework. Worth coordinating with CNIL again given the legal-validation angle
that made the original paper distinctive.

---

## 5. Legal metrics on linguistic content

**Claim.** The current framework assumes the transcript carries no identifiers,
so all risk is paralinguistic. Real recordings contain names, addresses and
dates, and the same three legal criteria apply to them.

**Already in hand.** Nothing in this repository, but it is the paper's own
stated future work and it is directly adjacent to Nijta's text anonymization.

**Still needed.** A joint metric over content and voice. The interesting
question is how the two combine: a system that scrubs the transcript perfectly
but leaves the voice intact is not anonymous, and vice versa, so a single
defensible number has to account for both.

**Venue.** Interspeech, or a privacy venue given the regulatory framing.

---

## 6. VPC 2026 submission, with v3

Once v3 is open-sourced, run it through the same harness and report Singling Out
and Linkability beside the EER. Independent of idea 1, but shares all its
infrastructure, and a system paper that reports legally grounded numbers that
nobody else reports is differentiated on its own.

---

---

## 7. Pseudo-speaker selection as an evaluation confound

**Claim.** The choice of pseudo-speaker is treated as a design parameter of an
anonymization system. It is also a parameter of the *evaluation*, and the legal
metrics are far more sensitive to it than the EER is. A system's measured
privacy therefore depends on a choice that is usually reported in one line of a
system description, if at all.

**The trajectory this sits on.** Pseudo-speaker selection has moved a long way
since the 2020 taxonomy:

| Period | Idea |
|---|---|
| 2020 | **Select** from a pool. Srivastava et al., "Design Choices for X-Vector Based Speaker Anonymization", Interspeech 2020: a spectrum over distance metric, region of x-vector space, gender, and whether assignment is per utterance or per speaker. |
| 2020 to 2022 | **Generate** rather than select. Distribution-preserving and GAN-based x-vector generation, on the argument that an averaged selection lands in a low-density region an attacker can detect. |
| 2022 to 2023 | **Guarantee** rather than generate. Differentially private speaker anonymization, PoPETs 2023, replacing heuristics with a privacy budget. |
| 2023 | **The vocoder breaks the link.** Vocoder drift work shows synthesised speech does not land where the chosen embedding says, so careful selection is partly undone downstream, prompting alignment-based compensation. |
| 2024 to 2025 | **Abandon the explicit embedding.** Adversarial perturbation of the speaker embedding rather than replacement; kNN-VC with no pseudo-speaker vector at all; neural audio codecs choosing a discrete voice id from a library, which is what VPC 2026 B4 does; diffusion-based control. |
| 2025 to 2026 | **The choice contaminates the measurement.** Work on making evaluation robust to target-speaker choice, and a finding that per-speaker risk is not intrinsic but emerges from attacker, system and available speech interacting. |

The arc runs: pick a point, then generate a plausible point, then guarantee a
property of the point, then discover the vocoder moves the point, then stop
using points, then notice the evaluation depended on the point all along.

**Why the legal metrics are the right instrument.** Assignment granularity is
the cleanest case. Per-speaker assignment gives every utterance of a speaker the
same pseudo-voice, so those utterances remain mutually linkable even though the
identity is hidden. Per-utterance assignment breaks that. Linkability should
separate these sharply while the EER barely moves, because the EER is computed
over trials that do not probe cross-utterance consistency. If that holds, it is
a direct demonstration that the field has been choosing a privacy-relevant
parameter using a metric that cannot see it.

Note the prior finding that speaker-level assignment does **not** give better
empirical protection than utterance-level under conventional metrics. That is
the result to re-examine with Linkability and Singling Out: a null under the EER
is exactly where a more sensitive metric earns its place.

**Already in hand.** The metrics, the adapter, and `legal-eval vpc`. VPC's own
configs expose `anon_level_random_per_utt` and `anon_level_constant`, so both
granularities are a config change on an existing baseline rather than new code.

**Still needed.** Run one baseline under both assignment granularities and
compare. That is a one-baseline experiment, cheap given the pipeline exists.

**Venue.** Interspeech or SPSC. The framing connects the 2020 design-choice
taxonomy to a measurement critique, which is a natural arc for the same author.

---

## 8. The enrollment budget as a second attacker axis

**Claim.** The published framework sweeps conversation length, which is how much
data the attacker holds about the target, and fixes the enrollment side, which
is how much it holds about the reference. Both are attacker capabilities. There
is no principled reason to sweep one and fix the other.

**Why it matters.** Sweeping both turns a curve into a surface and answers the
question a deployer actually asks: given the reference material an adversary
plausibly has, how much target speech makes the risk unacceptable. A single
curve cannot answer that.

**Already in hand.** The analysis in [`subset-design.md`](subset-design.md), and
`--max-enroll-utterances` in the subset constructor, which already fixes the
budget.

**Still needed.** A coarse grid over the enrollment budget, and the argument for
which point on the surface a compliance claim should be made at.

**Venue.** Fits naturally inside idea 1 or a journal version rather than
standing alone.

---

## 9. Reproducibility versus the right to erasure

**Claim.** Mozilla restricted access to older Common Voice releases in order to
honour deletion requests, because an old snapshot cannot. This is a concrete,
citable instance of a tension the speech community has not confronted:
reproducing published results requires frozen data, and data protection requires
that nothing stays frozen.

**Already in hand.** The case itself, lived rather than hypothesised. Derived
artifacts published under a DOI that reference clips whose speakers may since
have withdrawn, and the reasoning in [`data-release.md`](data-release.md) about
why derivability rather than pseudonymity is what justifies that release.

**Still needed.** The constructive part. Options worth working out: publishing
clip identifier lists rather than audio so withdrawals can be applied after the
fact, versioned artifacts that can be reissued against a withdrawal list, and
reporting coverage against the release actually used rather than claiming exact
reproduction.

**Venue.** SPSC, or a privacy venue. Could also be a position piece.

**Note.** This is the idea with the clearest claim to being needed rather than
merely interesting, and the authors of a CNIL-validated framework are
well-placed to make it.

## 10. The calibration is the metric: reproducibility of information-theoretic privacy measures

**The claim.** Local Information Disclosure reports privacy loss in bits via a
calibrated posterior, and the calibration is load bearing and unspecified. The
Platt intercept and the prior-ratio term cancel exactly in the softmax, so the
fitted slope is the only calibration parameter that reaches the reported number.
Varying it alone moves ALID by several bits on fixed data and can change its
sign. Fitting it per system, which is the natural thing to do, feeds the
measured effect back into the measurement: a weaker system fits a flatter slope,
a flatter slope flattens the posterior, and ALID drifts towards zero whatever
the system leaked.

Worse, on real VoicePrivacy data the fit is degenerate. The `libri_test`
enrollment against original trials is perfectly separable, so the logistic
regression has no finite maximum likelihood. The slope diverges and the reported
value is the optimiser's stopping point. In that limit ALID equals `log2 N`
exactly, so the metric collapses to a restatement of "Linkability is 1".

**Why it is worth writing.** This is not a defect in the idea. A calibrated
posterior is the right object if you want disclosure in bits. It is a
specification gap, and specification gaps in evaluation metrics are exactly what
a shared benchmark has to close. The paper writes the specification: which
development set, fit per system or shared, what regularisation, what behaviour
under separability. Rank-based metrics need none of it, which is a real argument
in their favour and one their own authors have not made.

**What makes it credible rather than adversarial.** Do it with the metric's
authors, and publish the specification jointly rather than the criticism alone.
Our implementation is public and labelled as our reading, and the verification
kit exists so they can correct it before anything is claimed.

**Status.** Measured, implemented, tested. The one thing outstanding is the
authors' confirmation of the reading, which is a one-paragraph answer.

**Venue.** Computer Speech and Language or Speech Communication as a joint
methodological paper. Or the SPSC Symposium for the short version first, since
the community that cares is small and concentrated there. The journal route is
better because a specification needs to be citable.

## 11. Metric saturation: we have been evaluating on populations too small to measure

**The claim.** Every privacy metric currently proposed, except the EER, is
bounded by the population: Linkability cannot exceed 1 and the bit metrics
cannot exceed `log2 N`. On the VoicePrivacy `libri` sets the largest population
available is 29 speakers, and at that size most conditions sit against the
bound. Linkability for original speech is exactly 1.000 at every `N` up to 29.
Four of six measured conditions have SRD and LID within ten percent of their
ceiling.

Where a metric is saturated it cannot separate systems, and the apparent
disagreements between metrics turn out to be sort-order tie-breaks on identical
values. The first version of our own comparison reported "two different
orderings" for exactly this reason, which is how the problem was found.

**Why it is worth writing.** It reframes the whole metric debate. The field has
spent eighteen months proposing replacements for the EER while evaluating them
on populations where none of the replacements has range. The binding constraint
is the evaluation set. A paper that measures the saturation point of each metric
as a function of population size, and states the minimum population at which
each becomes informative, is more useful than another metric.

It also tells a challenge organiser something actionable: how many speakers the
evaluation set needs for the metric to do any work.

**The experiment.** Sweep population size on a corpus large enough to exhaust
it, which means Common Voice rather than LibriSpeech. For each metric, find the
population at which it stops being pinned, and the population at which its
standard error over runs drops below the gap between two baseline systems. That
second quantity is the one a protocol designer needs and nobody reports.

**Status.** The saturation is measured on VPC. The sweep needs the larger
evaluation set, which is the Common Voice 27 work already in progress.

**Venue.** Interspeech, because it is a measurement result with a clear
practical consequence and it fits the length. Pair it with the subset design
correction, which is the same argument from a different direction.

## 12. Resolution, not ranking: what a privacy metric needs to be able to tell apart

**The claim.** The useful comparison between privacy metrics is not whether they
rank systems in the same order. On real data they do: where two metrics both
separate a pair of systems, they order it the same way, with no inversions. What
differs is **resolution**, how many pairs a metric separates at all, given the
spread over evaluation runs.

This suggests a figure of merit nobody computes: for a metric, a population size
and a pair of systems, is the gap between the two systems larger than the
combined standard error? Call it the metric's discriminative power at that
population. It is cheap to compute, it is what a protocol designer actually
needs, and it ranks metrics on something other than taste.

**Why it is worth writing.** The current literature argues about which metric is
conceptually right. That argument has no empirical content as long as the
metrics agree on ordering. Resolution has empirical content, it is measurable,
and it can conclude against the author's own metric, which ours does: Singling
Out resolves fewer pairs than Linkability or SRD on the same data, because its
range is compressed between `exp(-1)` and 1.

**Relation to existing work.** This is the privacy analogue of what the speaker
recognition community does with error bars on the EER and what the MT community
does with significance testing on BLEU. The machinery is standard. It simply has
not been applied here, which is why "the metrics disagree" is repeated without
anyone checking whether the disagreements survive the noise.

**Status.** Implemented. The tie-aware ranking and the coarsening-versus-
inversion distinction are in `examples/04_compare_privacy_metrics.py`. Needs the
larger population to be worth a paper.

**Venue.** Interspeech or ICASSP. Could also be the methodological core of the
joint commensurability paper rather than a separate submission, and that is
probably the better use of it.

## 13. Re-identification from verbal content and metadata (closes CNIL G1)

**The regulatory hook.** The CNIL's letter states that the Voice Harbor
demonstration "rests on the strong assumption that the data subject cannot be
identified, directly or indirectly, from the metadata and the verbal content of
their voice data", and that this "should not be taken for granted in the actual
context of use" (**G1**, see [`cnil-opinion.md`](cnil-opinion.md)). Everything
this project measures is the voice signal. The regulator has named the gap.

**The claim to test.** Anonymizing the voice does not anonymize the record. A
transcript carries named entities, places, employers, rare medical terms,
idiolect and topic. A call centre recording with the voice replaced still says
"this is Brij from Lille about account 4471". Voice Harbor masks 15+ categories
of spoken identifiers, so the interesting question is what survives **after**
that masking: writing style, vocabulary, topic, and the combination of
quasi-identifiers that no entity recogniser flags.

**The experiment.** Run the same Singling Out and Linkability predicates on
**text** rather than embeddings: author identification over transcripts, using a
text encoder in place of the x-vector. The framework is agnostic to where the
similarity comes from, which is a real strength and has not been exploited. Then
report the joint risk: a speaker who is unlinkable by voice and linkable by text
is not anonymous, and the current pipeline would score them as safe.

**Why it is publishable.** It makes the framework multimodal without changing
its definitions, and it answers a question a data protection authority asked in
writing. The negative result is also useful: if masking already suppresses
textual linkability on realistic data, that is worth establishing rather than
assuming.

**Venue.** Interspeech or a privacy venue (PETS, or the SPSC Symposium for the
short version). Potentially strong for a legal audience because it is the gap
between "anonymized audio" and "anonymized record".

**Status.** Not started. Needs a transcript corpus with speaker labels, which
Common Voice does not provide usefully (read sentences, not spontaneous speech).
Scoping required before resourcing.

## 14. Do pseudo-voices identify real people? (closes CNIL G3)

**The regulatory hook.** "The 'pseudo-identities' generated to anonymize a voice
dataset warrant vigilance. While it is understood that a pseudo-identity does not
correspond to a specific real person but to a 'mix' of several people, it should
be ensured that it does not make it possible to identify an individual"
(**G3**). The CNIL identified this independently of us, and nothing in this
repository measures it.

**The claim to test.** A GAN-sampled pseudo-speaker embedding is not guaranteed
to be far from every real speaker. The sttts baseline's own configuration is
evidence that its designers considered this live: `num_sampled: 5000` with
`sim_threshold: 0.7` is an explicit nearest-neighbour rejection step. Nobody has
reported what that threshold achieves, nor what happens without it.

**The experiment, which is cheap.** For every generated pseudo-speaker, compute
the similarity to every real speaker in the generator's training pool and report
the distribution of the maximum. Three quantities:

1. **Nearest-real-speaker similarity.** How close does the closest pseudo-voice
   get to a real person, and does any exceed the verification threshold a
   speaker verification system would use to accept?
2. **Collision rate between pseudo-voices.** Two source speakers assigned
   near-identical targets create a false link, which is a linkability failure
   that no current metric reports.
3. **Sensitivity to `sim_threshold`.** Sweep it, including removing it, and show
   what it buys.

All of it runs on embeddings we already have. No anonymization, no GPU hours
beyond extraction.

**Why it matters beyond compliance.** This is the first measurement that treats
the pseudo-speaker pool itself as an attack surface. If a pseudo-voice can be
matched to a real speaker, the anonymization has substituted one identity for
another rather than removing one.

**Venue.** Interspeech or ICASSP. Also directly usable in the Nijta POC report,
which is the rare case where a research result has an immediate commercial use.

**Status.** Not started, highest priority in
[`constitution.md`](constitution.md). The GAN model and the embedding pools are
on the workstation.

## 15. A standing re-assessment protocol (closes CNIL G4)

**The regulatory hook.** "We invite NIJTA to keep this demonstration up to date
in light of developments in anonymization and re-identification techniques, so as
to remain at the state of the art. This recommendation extends to the
demonstration of anonymity carried out by the entities implementing the
solution" (**G4**).

**The observation.** A privacy demonstration has a shelf life, and nobody has
said how long. An attacker model that was realistic in 2024 is not the frontier
in 2026: SSL encoders improved, and the semi-informed attacker in VPC 2026 is
stronger than its predecessor. A number computed once is a claim about a past
state of the art, and the metric does not record which.

**The contribution.** A protocol, not a metric:

1. **Version the attacker, not just the system.** Every reported privacy value
   carries the attacker model and its training data, the way an EER is
   meaningless without them. This repository already records attacker and
   population; the step is making it a required field rather than a convention.
2. **An attack registry with a re-run obligation.** A dated list of published
   attacks, each with the code to reproduce it, and a rule for when a
   demonstration must be recomputed. "Annually, or on publication of an attack
   that beats the current best by more than its error bar" is a defensible rule
   and nobody has proposed one.
3. **Report the decay.** Recompute an old demonstration under the current
   attacker and publish the delta. That number, how much a two-year-old
   anonymization claim has degraded, does not exist anywhere and would be
   genuinely new.

**Why it is the right kind of work for us.** It is convening and
infrastructural, it suits the PSST network, and it is what a standards body
needs and cannot produce. Nijta's CARE phase already promises annual
re-assessment commercially, so the protocol has a customer.

**Venue.** A journal, because a protocol needs to be citable and stable:
Computer Speech and Language, or Speech Communication. Possibly a standards
contribution instead of or alongside.

**Status.** Not started. Cheap to specify, and the measurement in point 3 is
already possible because the VPC 2024 and 2026 attackers both exist.

## Cross-cutting notes

**The name collision is a prerequisite for anything VPC-facing.** VPC 2026
already ships `evaluation/privacy/asv/metrics/linkability.py` computing the
Gomez-Barrero `D_sys`, a different quantity from this framework's Linkability.
Two metrics under one name in one repository will cause a bad comparison
eventually. Settle naming with the organizers before writing any integration.

**Always report four things with a legal metric.** Conversation length,
population size, attacker model, and whether distractor speakers were used. A
number without all four is not interpretable, and this should be stated in
whatever these ideas turn into.

**Report resolution, not just a value.** Every number from these metrics
should carry a spread over evaluation runs, and any claim that two systems
differ should be checked against it. Two systems whose metric values differ by
less than the combined standard error have not been shown to differ, and at the
ceiling that is the usual case. Our own first comparison claimed a ranking
disagreement that was a tie-break on saturated values.

**Check the ceiling before blaming the metric.** Linkability is bounded by 1 and
the bit metrics by `log2 N`. Before concluding that a metric fails to separate
two systems, check whether it had any range left at that population.

**Cite the CNIL obligation where one applies.** Ideas 13, 14 and 15 close
gaps a data protection authority named in writing (**G1**, **G3**, **G4**), and
idea 11 is the measurement **G2** requires. Saying so changes how a reviewer
reads the motivation, and it is true. See [`cnil-opinion.md`](cnil-opinion.md)
for the exact wording, and do not overstate it beyond that.

**Artifacts exist and should be cited.** Code on GitHub, data at
[10.5281/zenodo.23142030](https://doi.org/10.5281/zenodo.23142030). Any of these
papers can point at a reproducible baseline, which is unusual in this area and
worth making explicit in the submission.
