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

**Still needed.** Run the four baselines: roughly a day each on one GPU, from
VPC's own quoted anonymization costs (B2 2 h, B3 13 h, B5 1 h) plus a two-epoch
attacker finetune. Decide how to present the population-size caveat, since VPC's
evaluation sets hold about 40 speakers against the paper's 22,024.

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

**Artifacts exist and should be cited.** Code on GitHub, data at
[10.5281/zenodo.23142030](https://doi.org/10.5281/zenodo.23142030). Any of these
papers can point at a reproducible baseline, which is unusual in this area and
worth making explicit in the submission.
