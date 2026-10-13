# What the CNIL actually said

The single most load-bearing external fact this project relies on, stated at the
precision the source supports and no further. Read this before describing the
CNIL's involvement anywhere, in a paper, a README, a grant application or an
email.

## What is public here, and what is not

In August 2024 the CNIL's Technological Expertise Department reviewed Nijta's
Voice Harbor solution, at Nijta's request, and identified no obstacle to that
solution demonstrating compliance, as such, with the three criteria of the
Article 29 Working Party's Opinion 05/2014 on anonymization techniques
(ref. MAI/SA241024, 28 August 2024).

**The correspondence itself is not public**, and this file deliberately does not
reproduce it. The letter, its exact wording and its file references are held
with the rest of the regulatory correspondence outside this repository. What
follows is our own account of what the review obliges us to do, written so that
the engineering in this repository can cite it. Where a precise legal scope
matters, work from the letter, not from this summary.

## Three things it is not

**Not a certification.** It is a technical opinion from the authority's expert
department. Nijta holds no certification under Article 42 GDPR.

**Not about this framework.** The subject of the review was a solution. The
metrics in this repository were the instrument of the demonstration; they were
not themselves the object of it, and they were not certified.

**Not transferable.** An organisation deploying the solution must demonstrate
anonymity in its own context of use. A deployer may rely on the demonstration,
and must still satisfy the three criteria for its own processing.

## The four obligations the review placed on the demonstration

Numbered **G1** to **G4** so that research directions and code can cite them.
Each is our paraphrase of a requirement, paired with what this repository does
about it.

### G1. Metadata and verbal content are outside the demonstration

The demonstration rests on the assumption that a data subject cannot be
identified, directly or indirectly, from the metadata or the verbal content of
their voice data, and the review was explicit that this assumption should not be
taken for granted in an actual context of use.

Everything measured here is about the voice signal. Identification through what
was said, or through surrounding metadata, is out of scope and is flagged as not
safe to assume. See the content-leakage bound in
[`constitution.md`](constitution.md).

### G2. Re-identification risk must be quantified against a concrete attacker

The factors used to quantify re-identification risk must be chosen concretely,
accounting for the diversity of voice samples an attacker can reach and that
attacker's capabilities.

This is the regulatory basis for the two axes this project always reports:
population size and attacker model. It is why a privacy number quoted without
both is treated here as unusable, and why metric saturation at a small
population is a finding rather than a nuisance.
[`protocol.md`](protocol.md) requires both, and
`legal_eval.preflight` fails a report that omits the population.

### G3. Pseudo-voices must not identify a real person

A pseudo-identity is understood to correspond to a mix of several speakers
rather than to one real person, and it still warrants vigilance that it does not
make an individual identifiable.

The risk was identified independently of us. `legal_eval.pseudo_speaker`
measures it, and nothing else in this package does.

### G4. The demonstration must be kept at the state of the art

Keeping the demonstration current as anonymization and re-identification
techniques develop is a standing obligation rather than a one-off, and it
extends to the organisations implementing the solution.

A demonstration therefore has a shelf life. This is the mandate for the metric
commensurability work, and the reason
[`metric-comparison.md`](metric-comparison.md) exists.

## How to describe this, and how not to

| Do not write | Write |
|---|---|
| "validated by the CNIL" | "reviewed by the CNIL's Technological Expertise Department, which identified no obstacle to the solution demonstrating compliance" |
| "CNIL-certified" | "the subject of a CNIL technical opinion; not a certification" |
| "legally validated metrics" | "metrics used in a compliance demonstration reviewed by the CNIL" |
| "our framework is CNIL approved" | "the Voice Harbor solution was reviewed; the framework was the instrument of that review" |

The paper's published title, *Legally validated evaluation framework for voice
anonymization*, cannot be changed and should be cited as the title it is. It is
not a licence to describe the metrics themselves as legally validated.
