# Zenodo record: fields to enter

Draft metadata for the new version of concept DOI
[10.5281/zenodo.14976868](https://doi.org/10.5281/zenodo.14976868).

The current published version has an empty description, one creator, no
keywords and no link to the paper, so this replaces all of it. Upload through
**New version** rather than creating a separate record, so the subset
definitions and the score matrices stay under one citable DOI.

---

## Title

```
Score matrices and dataset definitions for "Legally validated evaluation framework for voice anonymization" (Interspeech 2025)
```

## Resource type

`Dataset`

## Version

`2.0.0`

## License

`Creative Commons Attribution 4.0 International` (CC-BY-4.0, unchanged)

The source corpus, Mozilla Common Voice 11.0, is released under CC-0, so CC-BY
on these derived artefacts is compatible.

## Creators

| Name | Affiliation |
|---|---|
| Vauquier, Nathalie | Nijta SAS, France |
| Srivastava, Brij Mohan Lal | Nijta SAS, France |
| Hosseini, Seyed Ahmad | Nijta SAS, France |
| Vincent, Emmanuel | Université de Lorraine, CNRS, Inria, LORIA, France |

Add ORCIDs where you have them; Zenodo uses them to link author profiles.

## Keywords

```
voice anonymization
speaker anonymization
privacy
GDPR
singling out
linkability
speaker verification
x-vectors
Common Voice
VoicePrivacy Challenge
```

## Related identifiers

| Relation | Identifier | Type |
|---|---|---|
| Is supplement to | `10.21437/Interspeech.2025-1699` | DOI |
| Is documented by | `https://github.com/brijmohan/voice-anonymization-legal-eval` | URL |

## Description

Zenodo's description box is a **rich text editor**, not an HTML source field.
Pasting HTML markup into it shows the tags literally instead of rendering them,
and its allowed tag set does not include tables. So paste the plain text below
and, if you want headings and emphasis, apply them with the editor's own toolbar
afterwards.

The text is written to read correctly with no formatting at all, so applying
the toolbar is optional.

```text
Data accompanying N. Vauquier, B. M. L. Srivastava, S. A. Hosseini and
E. Vincent, "Legally validated evaluation framework for voice anonymization",
Interspeech 2025, pages 3229 to 3233, doi:10.21437/Interspeech.2025-1699.

The paper introduces two metrics, Singling Out and Linkability, that translate
the singling out and linkability criteria of the Article 29 Working Party's
Opinion 05/2014 on Anonymization Techniques into quantities that can be
measured on speech. The framework was formally validated by the French Data
Protection Authority (CNIL). This record holds the data needed to reproduce the
paper's results. The reference implementation is at
https://github.com/brijmohan/voice-anonymization-legal-eval

CONTENTS

Dataset definitions, unchanged from version 1.0.0:

- cv11-A-filelist (19.5 MB): the Mozilla Common Voice 11.0 clips forming subset
  A, 22,024 speakers with at least 2 minutes of speech each.
- cv11-B-filelist (83.3 MB): the clips forming subset B, 4,949 speakers with at
  least 3 minutes each. Subset B's speakers are a subset of A's, with disjoint
  utterances.

Cosine score matrices, new in version 2.0.0, 12 files totalling 5.2 GB:

- scores_<attacker>_L<length>.npy: a 22,024 x 4,949 float32 array per attacker
  model and conversation length. Entry [i, j] is the cosine similarity between
  enrollment speaker i of subset A and test speaker j of subset B.
- scores_<attacker>_L<length>.npy.json: a sidecar per matrix giving the row and
  column speaker order, so each matrix is self describing.
- MANIFEST.json: SHA-256 checksums and shapes for every file.

The four attacker models differ in what their speaker embedding extractor was
trained on:

- original: original speech, evaluated on original speech.
- ignorant: original speech.
- semi_informed: speech anonymized with VoicePrivacy 2022 baseline B1.a.
- informed: speech anonymized with VoicePrivacy 2024 baseline B1.

Conversation length L is the number of utterances averaged per speaker, and
takes the values 1, 3 and 30. Test data for the three attackers is anonymized
with VoicePrivacy 2024 baseline B1.

USING THE MATRICES

    pip install git+https://github.com/brijmohan/voice-anonymization-legal-eval

    from legal_eval.io import load_score_matrix
    from legal_eval.sweeps import linkability_sweep
    m = load_score_matrix("scores_informed_L1.npy")
    print(linkability_sweep(m, speaker_counts=[20, 100, 1000, 10000]).mean())

To check a download against the published results:

    python examples/03_verify_against_score_matrices.py --release-dir .

That recomputes all 2,640 published Linkability points. Expect a worst case
difference of about 0.005, which is Monte Carlo noise from averaging five runs.

WHAT THE MATRICES CONTAIN

These are similarity scores rather than speaker embeddings, but the distinction
is smaller than it sounds and it is worth being explicit. Each matrix is the
product of two sets of unit norm x-vectors, so its rank is bounded by the
embedding dimension rather than by its shape. Measured on a random 3,000 x
3,000 block, the singular value spectrum collapses by four orders of magnitude
immediately after index 256, which is the x-vector dimension, and a rank 256
reconstruction reproduces the released scores to within 2e-07. A recipient can
therefore recover similarity structure that was never released, link a speaker
across the four released conditions, and cluster speakers by voice similarity.

ON THE SPEAKER LABELS

Row and column labels are internal pseudonyms of the form spk-f-20851, not
Mozilla Common Voice client identifiers. They should not be relied on as a
safeguard. The subset file lists in this record are public, Common Voice
publishes a client identifier for every clip, and the evaluation pipeline is
openly reproducible, so a determined party could recompute an original
condition matrix and align it to the released one to recover the mapping. The
pseudonyms raise the cost of doing so. They do not prevent it.

This release is published because every input it derives from is already
public: Common Voice 11.0 is openly available under CC-0, the subset
definitions are in this record, and the anonymization systems are the public
VoicePrivacy Challenge baselines B1 and B1.a. What a recipient gains is saved
computation rather than new information about any individual.

DELIBERATELY NOT INCLUDED

- The mapping from these pseudonyms to Common Voice client identifiers.
- The x-vectors themselves.
- Any utterance level file carrying Common Voice identifiers.

None of these is needed to reproduce any result in the paper.

CITATION

    @inproceedings{vauquier25_interspeech,
      title     = {Legally validated evaluation framework for voice anonymization},
      author    = {Nathalie Vauquier and Brij Mohan Lal Srivastava and
                   Seyed Ahmad Hosseini and Emmanuel Vincent},
      year      = {2025},
      booktitle = {Interspeech 2025},
      pages     = {3229--3233},
      doi       = {10.21437/Interspeech.2025-1699},
      issn      = {2958-1796},
    }

Derived from Mozilla Common Voice 11.0 (CC-0), which should be cited alongside
this record.
```

### Optional formatting once pasted

Using the editor toolbar, not by typing markup:

- Set each ALL CAPS line (CONTENTS, USING THE MATRICES, and so on) to a heading.
- Make the two indented code blocks monospace if the toolbar offers a code style.
- The `-` lists can be turned into real bullet lists.

None of this is required; the text is written to read correctly unformatted.

## After publishing

Send the new version DOI so it can be recorded, with the `MANIFEST.json`
checksums, in [`reproduction.md`](reproduction.md) and the README.
