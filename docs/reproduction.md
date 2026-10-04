# Reproducing the paper

What each stage costs, what this repository can verify on its own, and what you
need to supply.

## The pipeline

```
 Common Voice 11.0 ──▶ anonymization ──▶ x-vector extraction ──▶ metrics
   + LibriSpeech        VPC B1 / B1.a      Sidekit ECAPA-TDNN     THIS REPO
      ~1,700 h          GPU-weeks          GPU-weeks              seconds
```

Only the last stage is in this repository. That is deliberate: the first two are
external systems that already have maintained implementations, and the paper's
contribution is the evaluation, not the anonymizer.

| Stage | Cost | Where |
|---|---|---|
| Download Common Voice 11.0 + LibriSpeech | ~1 TB, hours | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/en/datasets), [openslr.org/12](https://www.openslr.org/12) |
| Anonymize with B1 and B1.a | GPU-weeks | [Voice-Privacy-Challenge-2024](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024), [-2022](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022) |
| Train 3 ECAPA-TDNN extractors | GPU-weeks | [Sidekit](https://git-lium.univ-lemans.fr/speaker/sidekit) |
| Extract x-vectors | GPU-days | Sidekit `extract_xvectors.py` |
| **Compute the metrics** | **seconds to minutes** | **this repository** |

## Datasets

| Dataset | Speakers (F/M/total) | Duration | Utterances | Role |
|---|---|---|---|---|
| LibriSpeech `train-clean-360` | 439 / 482 / 921 | 364 h | 104,014 | train the x-vector extractors |
| Common Voice 11.0 **A** | 5,601 / 16,423 / 22,024 | 323 h | 234,945 | ≥ 2 min per speaker |
| Common Voice 11.0 **B** | 1,354 / 3,595 / 4,949 | 1,409 h | 996,971 | ≥ 3 min per speaker |

`B ⊂ A` by speaker, with **disjoint utterances**. The roles swap between metrics:

| Metric | Enrollment | Test |
|---|---|---|
| Linkability, EER | A (22,024) | B (4,949) |
| Singling Out | B (495 sampled) | A (22,024) |

## Attacker models

Three extractors, each trained on `train-clean-360` processed differently:

| Attacker | Extractor trained on |
|---|---|
| Ignorant | original speech |
| Semi-Informed | speech anonymized with VPC 2022 B1.a |
| Informed | speech anonymized with VPC 2024 B1 |

Plus an *Original* condition: original extractor, original test data. Test data
for the three attackers is always anonymized with VPC 2024 B1.

## Stage 4: the metrics

Arrange each condition as a Kaldi-style directory:

```
data/cv11-A-informed/
├── spk2utt       # <speaker-id> <utt-id> <utt-id> ...
└── xvector.h5    # one dataset per utterance id
```

`.npz` works too, if HDF5 is inconvenient. Then, per attacker and per `L`:

```bash
ATTACKER=informed
for L in 1 3 30; do
  legal-eval score-matrix \
      --enroll-dir data/cv11-A-$ATTACKER --test-dir data/cv11-B-$ATTACKER \
      --conversation-length $L --attacker $ATTACKER \
      --output scores/${ATTACKER}_L${L}.npy

  legal-eval linkability --score-matrix scores/${ATTACKER}_L${L}.npy \
      --speaker-counts linear \
      --output results/linkability_${ATTACKER}_L${L}.json

  legal-eval eer --score-matrix scores/${ATTACKER}_L${L}.npy \
      --output results/eer_${ATTACKER}_L${L}.json

  # Singling Out: A and B swap roles
  legal-eval singling-out \
      --enroll-dir data/cv11-B-$ATTACKER --test-dir data/cv11-A-$ATTACKER \
      --conversation-length $L --attacker $ATTACKER \
      --n-enroll-speakers 495 --n-enroll-utterances 30 \
      --n-folds 10 --max-calibration 9 \
      --output results/singling_out_${ATTACKER}_L${L}.json
done

legal-eval plot --results-dir results --output figures/paper_figure.pdf
```

`--speaker-counts linear` gives the paper's Linkability grid (20 to 21,920 in
steps of 100); the default for Singling Out and EER is the geometric grid, which
is what the paper used for those.

### Cost

The 22,024 × 4,949 score matrix is about 870 MB in float64, and building it needs
all enrollment x-vectors in memory at once. With that in hand:

- **Linkability**, full 220-point grid × 5 runs: seconds, thanks to the
  hypergeometric estimator ([metrics.md](metrics.md#why-the-fast-path-is-exact)).
- **EER**, geometric grid: minutes, dominated by the convex hull over millions of
  scores.
- **Singling Out**: the expensive one. Cost scales with folds × `N` values ×
  enrollment speakers × runs, and each fold rebuilds and rescores every
  conversation. Start with `--n-folds 2` and a short `--speaker-counts` list.

## What this repository verifies without any data

```bash
pytest -q                                      # the full suite
python examples/02_reproduce_paper_figures.py  # the published curves vs the paper text
legal-eval demo                                # the whole pipeline on synthetic embeddings
```

`examples/02` checks the shipped `legal_eval/data/paper_results/` against every Linkability
value quoted in the paper's Section 5. The test suite pins the metrics against
independent oracles and against their theoretical baselines. See
[provenance.md](provenance.md#how-correctness-was-established).

## Getting the score matrices

The twelve cosine score matrices are published on Zenodo:

**[10.5281/zenodo.23142030](https://doi.org/10.5281/zenodo.23142030)** (concept DOI [10.5281/zenodo.14976868](https://doi.org/10.5281/zenodo.14976868),
which always resolves to the newest version)

5.2 GB total, 436 MB per matrix, 22,024 x 4,949 float32, CC-BY-4.0. The record
also carries the `cv11-A-filelist` and `cv11-B-filelist` defining the two
Common Voice subsets.

```sh
pip install zenodo_get
zenodo_get 10.5281/zenodo.23142030 -o release/

# or fetch a single condition
curl -LO https://zenodo.org/records/23142030/files/scores_informed_L1.npy
curl -LO https://zenodo.org/records/23142030/files/scores_informed_L1.npy.json
```

Verify the download, then check it reproduces the paper:

```sh
python -c "
import hashlib, json, sys
man = json.load(open('release/MANIFEST.json'))
for f in man['files']:
    h = hashlib.sha256()
    for b in iter(lambda fh=open('release/'+f['file'],'rb'): fh.read(1<<20), b''):
        h.update(b)
    assert h.hexdigest() == f['sha256'], f['file']
print('all checksums match')
"

python examples/03_verify_against_score_matrices.py --release-dir release/
```

The second command recomputes all 2,640 published Linkability points and should
report a worst-case difference near 0.005.

### SHA-256 checksums

| File | Attacker | L | SHA-256 |
|---|---|---|---|
| `scores_ignorant_L1.npy` | ignorant | 1 | `81f635fb55fd8e76…` |
| `scores_ignorant_L3.npy` | ignorant | 3 | `0d9f83942c640c29…` |
| `scores_ignorant_L30.npy` | ignorant | 30 | `1a7426d1da897fae…` |
| `scores_informed_L1.npy` | informed | 1 | `2cb11eea939aff4f…` |
| `scores_informed_L3.npy` | informed | 3 | `5afe70475c080ea5…` |
| `scores_informed_L30.npy` | informed | 30 | `304f96134d0713d8…` |
| `scores_original_L1.npy` | original | 1 | `56a93fc1698eeec0…` |
| `scores_original_L3.npy` | original | 3 | `1a39c72214fe35d8…` |
| `scores_original_L30.npy` | original | 30 | `4500f4c0f08af9f4…` |
| `scores_semi_informed_L1.npy` | semi_informed | 1 | `a2e95417c03b2e52…` |
| `scores_semi_informed_L3.npy` | semi_informed | 3 | `77ed25a9945f78a6…` |
| `scores_semi_informed_L30.npy` | semi_informed | 30 | `8fc59718dbfc9b4f…` |

Full digests are in `MANIFEST.json` on the record. Before using these, read
[`data-release.md`](data-release.md): the matrices encode more about speaker
identity than their name suggests, and the pseudonymous labels are not a
safeguard.

## Evaluating your own system

You do not need any of the above. Anonymize your audio, extract embeddings with
whatever extractor models your attacker, arrange them as `spk2utt` + `xvector.h5`,
and run stage 4. The things to get right are not in the code:

- **Keep enrollment, calibration and test utterances disjoint.** Sharing a
  recording measures recording identity, not speaker identity, and inflates every
  metric.
- **Report the conversation length.** A Linkability number without an `L` is not
  interpretable; the paper's own curves move by tens of points between `L = 1`
  and `L = 30`.
- **State the attacker.** An *Ignorant* result is close to meaningless as a
  compliance argument. The *Informed* attacker is the defensible worst case.
- **Report against the population size you are claiming for.** Both legal metrics
  fall as the search population grows, so a favourable number at small `N` says
  little about deployment at scale.
