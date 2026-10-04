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
on speech. The framework was formally validated by the French Data Protection
Authority (CNIL).

The motivating finding: **the equal error rate is close to blind to residual
re-identification risk.** Across attacker models and conversation lengths the EER
barely moves, while Linkability and Singling Out move a great deal.

> [!IMPORTANT]
> This repository was previously published with code that did not implement the
> paper. Version 2.0.0 replaces it in full. See [CHANGELOG.md](CHANGELOG.md).

## Install

```bash
pip install -e ".[all]"      # or: pip install -e .  for the numpy/scipy core
```

Python 3.9 or newer. The core needs only NumPy and SciPy; `h5py` is needed to
read HDF5 x-vector files, `matplotlib` to draw figures.

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

## Reproducing the paper

The full pipeline needs roughly 1,700 hours of anonymized Common Voice and three
trained x-vector extractors, so it is not something to re-run casually.
[`docs/reproduction.md`](docs/reproduction.md) sets out what is needed at each
stage and what this repository can verify on its own.

`data/paper_results/` ships the original experiment's Linkability results for the
*Original* condition at `L ∈ {1, 3, 10, 30}`, across 220 population sizes and 5
runs. These reproduce every Linkability value quoted in the paper.

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
