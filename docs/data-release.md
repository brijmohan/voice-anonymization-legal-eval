# Releasing the score matrices

> **Published.** [10.5281/zenodo.23142030](https://doi.org/10.5281/zenodo.23142030), version 2.0.0, CC-BY-4.0,
> 27 files, 5.34 GB. Concept DOI [10.5281/zenodo.14976868](https://doi.org/10.5281/zenodo.14976868).
> All 25 uploaded files verified byte-identical to the local copies.

What the cosine score matrices contain, what they let someone do, and where they
should live.

## What they are

Twelve matrices, one per attacker model and conversation length. Each is
22,024 x 4,949: every enrollment speaker in Common Voice 11.0 subset A scored
against every test speaker in subset B.

| | |
|---|---|
| Shape | 22,024 x 4,949 |
| Stored as | float32, 436 MB each, 5.2 GB total |
| Rows | enrollment speakers (subset A) |
| Columns | test speakers (subset B) |
| Values | cosine similarity between averaged x-vectors |
| Labels | internal pseudonyms, e.g. `spk-f-20851` |

### float32 is exactly lossless here

The scores were computed from float32 x-vectors, so every value is already
exactly representable in float32. Converting is not an approximation:

```
max |float64 - float32|  =  0.000e+00     for all 12 matrices
```

Beaters counts are identical, and every Linkability value matches to the last
digit. The release is half the size with literally no change in any output.
`scripts/prepare_score_matrices.py` verifies this per matrix and refuses to
write a lossy conversion unless explicitly told to.

Further compressing to `.npz` saves another 17% (5.2 GB to 4.3 GB) but costs a
decompression step on every read. Not worth it at this size.

## What they let someone do

**They are not "much less identity-bearing than embeddings".** It is tempting to
assume that releasing similarities rather than vectors is inherently safer. For a
matrix this large relative to the embedding dimension, that is not true, and the
structure says so plainly.

The matrix is the product of two sets of unit-norm embeddings, `S = X·Yᵀ`, so its
rank is bounded by the embedding dimension rather than by its shape. Measured on
a random 3,000 x 3,000 block, the singular value spectrum collapses by four
orders of magnitude immediately after index **256** (exactly the x-vector
dimension), and a rank-256 reconstruction reproduces the released scores to
within 2e-07, which is float32 rounding:

```
rank  192: sigma_k/sigma_1 = 6.1e-06
rank  256: sigma_k/sigma_1 = 1.3e-06     <- embedding dimension
rank  300: sigma_k/sigma_1 = 1.3e-09     <- numerical zero
```

So the matrix is an inner-product representation of the embeddings. Factorising
it recovers `X` and `Y` up to an invertible 256 x 256 transform; the unit-norm
constraints on both factor sets over-determine that transform, so recovering the
true geometry is an inverse problem rather than a barrier. (A plain SVD does
*not* do it, because the scale splits arbitrarily between the factors, but that
is an inconvenience rather than a protection.)

Concretely, a recipient can:

- link a speaker **across the released conditions**, matching their column in the
  *ignorant* matrix to their column in the *original* one, since both are
  similarity profiles against the same 22,024 references;
- cluster speakers and find unusually distinctive ones, which is exactly the
  worst-case analysis in the CNIL report;
- recover similarity structure that was never released, including
  enrollment-to-enrollment similarities.

All of this stays inside the pseudonymous label space.

## Do not treat the pseudonyms as a safeguard

It would be convenient to argue that `spk-f-20851` protects anyone. It does not,
and a paper about anonymization should not lean on an argument it would reject
from someone else.

The file lists defining subsets A and B are **already published**, CC-BY, at
[10.5281/zenodo.14976869](https://doi.org/10.5281/zenodo.14976869). Common Voice
11.0 and its per-clip `client_id` metadata are public. So anyone can reconstruct
the speaker grouping of both subsets with real Common Voice identifiers. And
because this repository deliberately makes the pipeline reproducible, they can
also compute their own *original*-condition score matrix and align it to the
released one by permutation, which recovers the pseudonym-to-client-id mapping.

The pseudonyms raise the cost of de-anonymisation. They do not prevent it.

## Why releasing them is still reasonable

The justification is **derivability, not pseudonymity**:

1. **Everything here follows from already-public inputs.** Common Voice 11.0 is
   openly downloadable, the subset definitions are already on Zenodo, and the
   anonymization systems are the public Voice Privacy Challenge baselines B1 and
   B1.a. A motivated party can compute equivalent matrices without this release.
   The marginal disclosure is the saved compute, not new information.
2. **What that party would learn is mild.** At worst they can say how
   re-identifiable a given Common Voice volunteer remains after anonymization.
   That is an attribute of a public-corpus contributor, derived from audio they
   chose to publish under CC-0.
3. **The field already does this.** Voice Privacy Challenge artifacts routinely
   include embeddings and score files.
4. **Reproducibility needs them.** Without the matrices, six of the nine panels
   of the paper's figure cannot be checked by anyone outside the authors.

Point 1 is the load-bearing one. If a future release covers a corpus that is
*not* public, none of this reasoning carries over and the decision has to be
made again from scratch.

## What must not be released

**The mapping from these pseudonyms to Common Voice client ids.** Publishing it
turns a reconstructible link into a handed-over one, and there is no
reproducibility argument for it: every curve in the paper can be recomputed
without it. It sits in the experiment archive beside everything else, so it has
to be excluded deliberately rather than by accident.

The same goes for the x-vectors themselves and for any utterance-level file that
carries Common Voice ids.

## Where to publish

**Zenodo** for the canonical copy, **Hugging Face** as an optional mirror.

| | Zenodo | Hugging Face | GCS |
|---|---|---|---|
| Durability | CERN-operated, long-term preservation commitment | depends on a company | depends on a billing account |
| DOI | yes, versioned, citable | available for datasets | no |
| Cost | free | free | storage and egress billed |
| Survives losing your cloud account | yes | yes | **no** |
| Discoverability in speech research | good | very good | none |

The decisive criterion is the one you named: it should stay in the public domain
even if the account that uploaded it goes away. That rules out GCS as the
canonical home. A bucket is a fine working mirror, but it is not an archive: it
disappears with the billing relationship, has no DOI, and bills whoever downloads
it.

Zenodo mints a DOI, versions the record, and is operated by CERN with a
preservation commitment measured in decades. 5.2 GB sits inside its normal
per-record quota. The DOI is also what makes the artifact citable from the paper
and from this repository, which matters more than download speed for something
that will be fetched once.

Hugging Face is worth adding as a mirror because it is where speech researchers
actually look, and `huggingface_hub` makes programmatic download a one-liner. It
should not be the only copy, since its retention is a company's policy rather
than an archival commitment.

### Checklist

1. `python scripts/prepare_score_matrices.py --root <archive> --output-dir release/`
2. Confirm every line reports `exact=True`.
3. Confirm the release directory contains **no** pseudonym-to-client-id mapping,
   no x-vectors, and no utterance-level Common Voice ids.
4. Strip macOS AppleDouble sidecars before uploading. Writing to an external
   volume creates a `._name` file beside every real one, and they will otherwise
   be published as 4 KB of noise per matrix:
   ```sh
   dot_clean release/ || find release -name '._*' -delete
   ```
5. Upload `release/` to Zenodo as a **new version of the existing record**,
   concept DOI [10.5281/zenodo.14976868](https://doi.org/10.5281/zenodo.14976868),
   which already holds the subset file lists. One versioned record keeps the
   subset definitions and the matrices under a single citable DOI.
6. Put this document's "What they let someone do" and "Do not treat the
   pseudonyms as a safeguard" sections in the record description, so recipients
   are not misled about what they are getting.
7. Record the version DOI and the `MANIFEST.json` checksums in
   [`reproduction.md`](reproduction.md).
8. Optionally mirror to Hugging Face, pointing back at the DOI.
