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

Zenodo renders HTML. The block below is ready to paste.

```html
<p>Data accompanying <strong>N. Vauquier, B. M. L. Srivastava, S. A. Hosseini and
E. Vincent, "Legally validated evaluation framework for voice anonymization",
Interspeech 2025, pp. 3229-3233</strong>
(<a href="https://doi.org/10.21437/Interspeech.2025-1699">10.21437/Interspeech.2025-1699</a>).</p>

<p>The paper introduces two metrics, <em>Singling Out</em> and <em>Linkability</em>,
that translate the singling out and linkability criteria of the Article 29
Working Party's Opinion 05/2014 on Anonymization Techniques into quantities that
can be measured on speech. The framework was formally validated by the French
Data Protection Authority (CNIL). This record holds the data needed to reproduce
the paper's results, and the reference implementation lives at
<a href="https://github.com/brijmohan/voice-anonymization-legal-eval">github.com/brijmohan/voice-anonymization-legal-eval</a>.</p>

<h3>Contents</h3>

<p><strong>Dataset definitions</strong> (unchanged from version 1.0.0)</p>
<ul>
  <li><code>cv11-A-filelist</code> (19.5 MB): the Common Voice 11.0 clips forming
      subset A, 22,024 speakers with at least 2 minutes of speech each.</li>
  <li><code>cv11-B-filelist</code> (83.3 MB): the clips forming subset B, 4,949
      speakers with at least 3 minutes each. Subset B's speakers are a subset of
      A's, with disjoint utterances.</li>
</ul>

<p><strong>Cosine score matrices</strong> (new in 2.0.0, 12 files, 5.2 GB total)</p>
<p><code>scores_&lt;attacker&gt;_L&lt;length&gt;.npy</code>, one per attacker model
and conversation length, each a 22,024 x 4,949 float32 array. Entry
<code>[i, j]</code> is the cosine similarity between enrollment speaker
<code>i</code> of subset A and test speaker <code>j</code> of subset B. Each file
carries a <code>.npy.json</code> sidecar listing the row and column speaker
order, so a matrix is self-describing. <code>MANIFEST.json</code> gives SHA-256
checksums for every file.</p>

<table>
<tr><th>Attacker</th><th>Speaker embedding extractor trained on</th></tr>
<tr><td><code>original</code></td><td>original speech, evaluated on original speech</td></tr>
<tr><td><code>ignorant</code></td><td>original speech</td></tr>
<tr><td><code>semi_informed</code></td><td>speech anonymized with VoicePrivacy 2022 baseline B1.a</td></tr>
<tr><td><code>informed</code></td><td>speech anonymized with VoicePrivacy 2024 baseline B1</td></tr>
</table>

<p>Conversation length <code>L</code> is the number of utterances averaged per
speaker, and takes the values 1, 3 and 30. Test data for the three attackers is
anonymized with VoicePrivacy 2024 baseline B1.</p>

<h3>Using the matrices</h3>

<pre><code>pip install git+https://github.com/brijmohan/voice-anonymization-legal-eval

python -c "
from legal_eval.io import load_score_matrix
from legal_eval.sweeps import linkability_sweep
m = load_score_matrix('scores_informed_L1.npy')
print(linkability_sweep(m, speaker_counts=[20, 100, 1000, 10000]).mean())
"</code></pre>

<p>To check a download against the published results:</p>

<pre><code>python examples/03_verify_against_score_matrices.py --release-dir .</code></pre>

<p>That recomputes all 2,640 published Linkability points. Expect a worst-case
difference of about 0.005, which is Monte Carlo noise from averaging five runs.</p>

<h3>What the matrices contain</h3>

<p>These are similarity scores, not speaker embeddings, but the distinction is
smaller than it sounds and it is worth being explicit. Each matrix is the product
of two sets of unit-norm x-vectors, so its rank is bounded by the embedding
dimension rather than by its shape: measured on a random 3,000 x 3,000 block, the
singular value spectrum collapses by four orders of magnitude immediately after
index 256, which is the x-vector dimension, and a rank-256 reconstruction
reproduces the released scores to within 2e-07. A recipient can therefore
recover similarity structure that was never released, link a speaker across the
four released conditions, and cluster speakers by voice similarity.</p>

<h3>On the speaker labels</h3>

<p>Row and column labels are internal pseudonyms of the form
<code>spk-f-20851</code>, not Mozilla Common Voice client identifiers. They should
not be relied on as a safeguard. The subset file lists in this record are public,
Common Voice publishes a client identifier for every clip, and the evaluation
pipeline is openly reproducible, so a determined party could recompute an
original-condition matrix and align it to the released one to recover the
mapping. The pseudonyms raise the cost of doing so; they do not prevent it.</p>

<p>This release is published because every input it derives from is already
public: Common Voice 11.0 is openly available under CC-0, the subset definitions
are in this record, and the anonymization systems are the public VoicePrivacy
Challenge baselines B1 and B1.a. What a recipient gains is saved computation
rather than new information about any individual.</p>

<h3>Deliberately not included</h3>

<ul>
  <li>The mapping from these pseudonyms to Common Voice client identifiers.</li>
  <li>The x-vectors themselves.</li>
  <li>Any utterance-level file carrying Common Voice identifiers.</li>
</ul>

<p>None of these is needed to reproduce any result in the paper.</p>

<h3>Citation</h3>

<pre><code>@inproceedings{vauquier25_interspeech,
  title     = {{Legally validated evaluation framework for voice anonymization}},
  author    = {Nathalie Vauquier and Brij Mohan Lal Srivastava and
               Seyed Ahmad Hosseini and Emmanuel Vincent},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {3229--3233},
  doi       = {10.21437/Interspeech.2025-1699},
  issn      = {2958-1796},
}</code></pre>

<p>Derived from Mozilla Common Voice 11.0 (CC-0), which should be cited
alongside this record.</p>
```

---

## Plain-text fallback

If you would rather not paste HTML, this conveys the essentials:

```
Data accompanying N. Vauquier, B. M. L. Srivastava, S. A. Hosseini and
E. Vincent, "Legally validated evaluation framework for voice anonymization",
Interspeech 2025, pp. 3229-3233, doi:10.21437/Interspeech.2025-1699.

Reference implementation:
https://github.com/brijmohan/voice-anonymization-legal-eval

CONTENTS

cv11-A-filelist, cv11-B-filelist: the Common Voice 11.0 clips forming subsets A
(22,024 speakers, at least 2 minutes each) and B (4,949 speakers, at least 3
minutes each). B's speakers are a subset of A's, with disjoint utterances.

scores_<attacker>_L<length>.npy: twelve 22,024 x 4,949 float32 cosine score
matrices, one per attacker model (original, ignorant, semi_informed, informed)
and conversation length L (1, 3, 30). Rows are subset A enrollment speakers,
columns are subset B test speakers. Each has a .npy.json sidecar giving the row
and column speaker order. MANIFEST.json holds SHA-256 checksums.

WHAT THEY CONTAIN

Each matrix is the product of two sets of unit-norm x-vectors, so its rank is
bounded by the 256-dimensional embedding rather than by its shape. A recipient
can recover similarity structure that was never released, link a speaker across
the four conditions, and cluster speakers by voice similarity.

SPEAKER LABELS

Labels are internal pseudonyms (spk-f-20851), not Common Voice client
identifiers, and should not be relied on as a safeguard: the subset lists here
are public, Common Voice publishes a client identifier per clip, and the
pipeline is reproducible, so the mapping could be recovered by alignment. This
release is published because every input is already public, so what a recipient
gains is saved computation rather than new information about any individual.

NOT INCLUDED

The pseudonym-to-client-identifier mapping, the x-vectors, and any
utterance-level file carrying Common Voice identifiers. None is needed to
reproduce any result in the paper.
```

## After publishing

Send the new version DOI so it can be recorded, with the `MANIFEST.json`
checksums, in [`reproduction.md`](reproduction.md) and the README.
