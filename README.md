# Legally validated evaluation framework for voice anonymization

Reference implementation of the **Singling Out** and **Linkability** metrics from:

> N. Vauquier, B. M. L. Srivastava, S. A. Hosseini, E. Vincent.
> *Legally validated evaluation framework for voice anonymization.*
> Interspeech 2025.
> [[paper]](https://www.isca-archive.org/interspeech_2025/vauquier25_interspeech.html)

The two metrics turn the *singling out* and *linkability* criteria of the
Article 29 Working Party's [Opinion 05/2014 on Anonymization
Techniques](https://ec.europa.eu/justice/article-29/documentation/opinion-recommendation/files/2014/wp216_en.pdf),
endorsed by the European Data Protection Board, into quantities you can measure
on speech.

**All three criteria, at different levels of maturity.** Article 29 Working
Party Opinion 05/2014 sets three tests: **singling out**, **linkability** and
**inference**. The first two are the published, CNIL-reviewed metrics. The
third, `legal_eval.inference`, is **new and far less validated**: it measures
attribute inference, how much better than the population prior an attacker
predicts a sensitive attribute from the released artifact. Treat it as a lower
bound from a linear probe, not as a guarantee, and read its own limitations
before quoting it. A null result bounds one classifier on one set of labels.

**On the CNIL's involvement, stated precisely.** In August 2024 the CNIL's
Technological Expertise Department reviewed Nijta's Voice Harbor solution at
Nijta's request and concluded that it identified no obstacle to the solution
demonstrating compliance, as such, with the three anonymization criteria of
Article 29 Working Party Opinion 05/2014 (ref. MAI/SA241024, 28 August 2024).
That is a technical opinion on a solution. It is **not** a certification, Nijta
holds no Article 42 GDPR certification, and it does **not** transfer to anyone
deploying the solution: the CNIL is explicit that a deploying organisation must
demonstrate anonymity in its own context. The metrics in this repository were
the vehicle of that demonstration; they were not themselves certified. See
[`docs/cnil-opinion.md`](docs/cnil-opinion.md) for the full scope and the four
obligations the CNIL placed on deployers.

The motivating finding: **the equal error rate is close to blind to residual
re-identification risk.** Across attacker models and conversation lengths the EER
barely moves, while Linkability and Singling Out move a great deal.

> [!IMPORTANT]
> This repository was previously published with code that did not implement the
> paper. Version 2.0.0 replaces it in full. See [CHANGELOG.md](CHANGELOG.md).

## Install

```bash
pip install voice-anonymization-legal-eval          # core
pip install "voice-anonymization-legal-eval[all]"   # plus h5py and matplotlib
```

From a checkout, for development:

```bash
pip install -e ".[dev]"
```

Python 3.9 or newer. The core needs only NumPy and SciPy; `h5py` is needed to
read HDF5 x-vector files, `matplotlib` to draw figures. SciPy and h5py are
imported only where they are used, so most of the package works without them.

The scripts under `examples/` and `scripts/` also run straight from a clone
without installing anything, falling back to the repo root on `sys.path`. An
installed copy always takes priority, so this never shadows a real install.

## Try it in 30 seconds

```bash
legal-eval demo --output-dir demo_output
```

That builds a synthetic corpus, simulates two degrees of anonymization, computes
all three metrics across population sizes and conversation lengths, and writes a
figure. No data, no GPU, no model. For the same thing as readable code, see
[`examples/01_quickstart.py`](examples/01_quickstart.py).

To redraw the paper's published curves and check them against the values quoted
in its text:

```bash
python examples/02_reproduce_paper_figures.py --output-dir figures
```

## What this package does, and what it does not

This package evaluates **speaker embeddings**. It takes x-vectors (or any other
speaker embeddings) and computes the re-identification risk they carry.

It does **not** anonymize speech and does not train embedding extractors. Those
are upstream steps, done with the toolkits the paper used, and re-implementing
them here would only produce a worse copy. [`docs/reproduction.md`](docs/reproduction.md)
pins the exact external systems and versions.

```
 speech ──▶ anonymization ──▶ x-vector extraction ──▶  THIS PACKAGE
            (VPC B1/B1.a)      (Sidekit ECAPA-TDNN)    Singling Out
            external            external                Linkability
                                                        ROCCH-EER
```

To evaluate your own system: anonymize your audio, extract embeddings with your
attacker's extractor, and point this package at them. The metrics do not care
how the embeddings were produced.

## The metrics

| Metric | Question it answers | Chance level |
|---|---|---|
| **Singling Out** `π_sing` | Can an attacker isolate exactly one speaker out of `N`? | `exp(-1) ≈ 37%` |
| **Linkability** `π_link` | Can an attacker match a test recording to the right speaker among `N'`? | `1/N'` |
| **ROCCH-EER** | The conventional speaker-verification baseline, plotted as `1 - EER`. | `50%` |

Both legal metrics are reported against the number of speakers the attacker must
search and against the conversation length `L`, the number of utterances averaged
per speaker. [`docs/metrics.md`](docs/metrics.md) gives the definitions, the
calibration rule and the correspondence to the paper's equations.

### Comparing against the other recent proposals

Three other groups have proposed replacing the EER, in bits rather than
probabilities. `legal_eval.comparison` implements Local Information Disclosure
and Similarity Rank Disclosure so that all of them can be computed from one
artifact, which is the only way to compare the metrics rather than the
pipelines:

```bash
python examples/04_compare_privacy_metrics.py --demo
```

Those two implementations are **our reading of the published definitions**, not
authoritative. [`docs/metric-comparison.md`](docs/metric-comparison.md) records
what the comparison shows, including that SRD's identification rate is the
Linkability metric, that the probability and bit metrics move in opposite
directions as the population grows, and that LID's reported bits depend as much
on its calibration as on the system being measured.

## Why this exists, and what is established

[`docs/constitution.md`](docs/constitution.md) sets out the objective, the
findings and their evidence, and where identity survives anonymization. Every
claim carries a marker so a reader can tell a measurement from an inference:
[M] measured, [P] published, [I] inferred, [?] open.

[`docs/bibliography.md`](docs/bibliography.md) is the reference library: 262
works categorised by the question they answer rather than by topic, with
[`docs/references.bib`](docs/references.bib) as the machine-readable form. The
working set is marked separately from the background, because a library where
every entry looks equally load bearing says nothing about what to read first.

## Using it on your own data

Point it at Kaldi-style directories, each holding a `spk2utt` file and an
`xvector.h5` with one dataset per utterance id:

```bash
# Linkability and EER: enrollment set A, test set B
legal-eval score-matrix --enroll-dir data/cv11-A --test-dir data/cv11-B \
    --conversation-length 1 --attacker informed --output scores/informed_L1.npy

legal-eval linkability --score-matrix scores/informed_L1.npy \
    --output results/linkability_informed_L1.json
legal-eval eer --score-matrix scores/informed_L1.npy \
    --output results/eer_informed_L1.json

# Singling Out: the roles of A and B swap, per the paper
legal-eval singling-out --enroll-dir data/cv11-B --test-dir data/cv11-A \
    --conversation-length 1 --attacker informed \
    --output results/singling_out_informed_L1.json

legal-eval plot --results-dir results --output figures/paper_figure.pdf

# Check the result against failures we have already made
legal-eval preflight results.json
```

Or from Python:

```python
from legal_eval import (
    build_speaker_embeddings, build_test_embeddings,
    cosine_score_matrix, linkability_sweep,
)

enroll = build_speaker_embeddings(enroll_spk2utt, utt2xvector)
test, _ = build_test_embeddings(test_spk2utt, utt2xvector, conversation_length=3)
scores = cosine_score_matrix(enroll, test)

result = linkability_sweep(scores, speaker_counts=[20, 100, 1000, 10000])
print(result.mean())
```

## VoicePrivacy Challenge

The metrics run directly on a VPC 2026 evaluation, reusing the speaker
embeddings its ASV step already caches. No model, no GPU, no second pass over
audio:

```sh
pip install "voice-anonymization-legal-eval[vpc]"
legal-eval vpc --results-dir exp/asv_anon_mcadams \
    --distractors train-clean-360_mcadams --output results/legal_mcadams.csv
```

See [`docs/vpc-benchmark.md`](docs/vpc-benchmark.md), which also covers the two
things to get right before publishing a comparison: VPC's evaluation sets hold
far fewer speakers than the paper's, and the VPC repo already uses the name
"linkability" for a different quantity.

## Reproducing the paper

The full pipeline needs roughly 1,700 hours of anonymized Common Voice and three
trained x-vector extractors, so it is not something to re-run casually.
[`docs/reproduction.md`](docs/reproduction.md) sets out what is needed at each
stage and what this repository can verify on its own.

`legal_eval/data/paper_results/` ships the values behind all nine panels of the paper's
Figure 1: three metrics, three conversation lengths, four attacker models.
`python examples/02_reproduce_paper_figures.py` redraws the figure from them and
checks it against the values quoted in the paper's text.

The twelve cosine score matrices those results come from are on Zenodo at
**[10.5281/zenodo.23142030](https://doi.org/10.5281/zenodo.23142030)** (5.2 GB, CC-BY-4.0), together with the
Common Voice subset definitions. With them you can recompute every Linkability
and EER curve from scratch rather than taking the published numbers on trust:

```sh
pip install zenodo_get && zenodo_get 10.5281/zenodo.23142030 -o release/
python examples/03_verify_against_score_matrices.py --release-dir release/
```

Read [`docs/data-release.md`](docs/data-release.md) before redistributing them.

## Running it on your own system

[`docs/protocol.md`](docs/protocol.md) is the procedure, aimed at somebody
evaluating their own anonymization system, including VoicePrivacy Challenge
participants. It states what to compute, what to report beside each number, and
the seven things that make a reported result uninterpretable.

The short version of why a protocol and not just a metric: a privacy number
means nothing without the population it was computed over, the attacker that
computed it, and the chance level for that population. Singling Out's chance
level is `exp(-1)`, so a reported 0.37 is indistinguishable from an attacker
who learned nothing. Linkability's is `1/N`, so the same 0.37 is near-perfect
identification at `N = 3` and useless at `N = 10000`.

## Checking a result before you believe it

```sh
legal-eval preflight results.json
```

Every check in it exists because a specific wrong number got as far as a draft
or a launch, and each one prints the failure it guards against. The point is
that none of those failures raised an error. They all produced a plausible
number:

| Check | What it caught |
|---|---|
| `population_recorded` | Without `N` nothing reads against its own chance level |
| `exclusions_declared` | A destroyed utterance scores at chance, so dropping one quietly flatters the system |
| `no_zero_standard_error` | A standard error of exactly zero is a bug, not precision |
| `saturation_declared` | A metric at its ceiling cannot separate systems |
| `srd_idr_matches_linkability` | They are the same quantity, not two metrics agreeing |
| `lid_calibration_recorded` | Only the fitted slope reaches the result, so LID without it cannot be reproduced |
| `worst_case_reported` | An average disclosure can conceal one fully exposed individual |
| `cllr_calibration_reported` | The gap between actual and minimum Cllr is the attacker's calibration loss, not the system's privacy |
| `metric_directions` | How a coarsening gets misread as an inversion |

Two of these corrected themselves when first run over this project's own
output, which is the use we would most encourage: run it on numbers the
harness did not produce.

Related, and the one that is easiest to miss: a system that emits silence
instead of speech is rewarded for it, because silence scores at chance and
chance is the best privacy score attainable.
`legal_eval.audio_integrity.check_anonymized` tells the two apart, and
[`docs/constitution.md`](docs/constitution.md) section 3.9 has the case that
prompted it.

## How we know it is right

The implementation was checked against the original cosine score matrices the
paper's numbers came from. Every published Linkability point, across all four
attacker models and all three conversation lengths, was recomputed:

| | |
|---|---|
| Published points recomputed | **2,640** |
| Mean absolute difference | **0.0007** |
| Worst difference at any point | **0.005** |
| Correlation with published | **> 0.9987** |

The residual is Monte Carlo noise from averaging only five runs; it shrinks
further with `--estimator exact`. Reproduce it with
[`examples/03_verify_against_score_matrices.py`](examples/03_verify_against_score_matrices.py)
once you have the score matrices.

Each full condition takes **0.6 seconds**, against hours for the original code.

The metrics are additionally pinned by independent oracles and theoretical
baselines rather than by their own past output; see
[`docs/provenance.md`](docs/provenance.md#how-correctness-was-established).

## Differences from the original experiment code

The implementation here is faithful in its mathematics and corrects three
reproducibility defects in the internal scripts: an RNG-sharing bug across
parallel workers, a fold loop that silently reused one data split, and the use of
`random.sample` on a set, which has been an error since Python 3.11. Each is
documented, and the original behaviour remains reachable by a flag where it
affects results. See [`docs/differences.md`](docs/differences.md).

Linkability also runs far faster here. Explicit candidate-set sampling is
replaced by an equivalent hypergeometric draw, which collapses the paper's full
sweep from hours to seconds without changing the sampling distribution. The
derivation is in [`docs/metrics.md`](docs/metrics.md) and the equivalence is
asserted in the test suite.

## Tests

```bash
pytest -q
```

The suite pins behaviour rather than just exercising it:

- Singling Out: an uninformative attacker isolates at `exp(-1)`, the PSO baseline.
- Linkability: the fast estimator matches literal candidate-set sampling.
- ROCCH-EER: matches a brute-force search over randomised decision rules, and the
  analytic EER of separated Gaussians.
- Both legal metrics respond to conversation length while the EER stays flat.

## Releasing

See [`docs/releasing.md`](docs/releasing.md). Versioning treats **the numbers the
metrics produce** as the public contract, not just the Python API: anything that
moves a value is a major version and gets an entry in
[`docs/differences.md`](docs/differences.md).

## Citation

```bibtex
@inproceedings{vauquier25_interspeech,
  title     = {Legally validated evaluation framework for voice anonymization},
  author    = {Nathalie Vauquier and Brij Mohan Lal Srivastava and
               Seyed Ahmad Hosseini and Emmanuel Vincent},
  booktitle = {Interspeech 2025},
  year      = {2025},
}
```

## License

MIT, see [LICENSE](LICENSE). The ROCCH-EER here is an independent implementation;
the BOSARIS toolkit used in the original experiments is licensed for
non-commercial use only and is deliberately not vendored.
