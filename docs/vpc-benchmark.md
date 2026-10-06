# Benchmarking the VoicePrivacy Challenge baselines

How to compute Singling Out and Linkability for the VPC 2026 Track 1 baselines,
and what to watch out for when reading the result.

## Why this is cheap

VPC's evaluation already extracts speaker embeddings while computing the EER,
and `evaluation/privacy/asv/speaker_extraction.py` **caches them to disk**:

```
exp/asv_anon<suffix>/cosine_out/emb_xvect/<dataset>/utt-level/
    speaker_vectors.pt    one row per utterance
    id2idx                utterance id -> row
    idx2spk               row -> speaker
    spk2gender
```

The legal metrics consume exactly those vectors. So once a baseline's normal
evaluation has run, adding the two metrics costs **seconds of CPU**: no model
load, no GPU, no second pass over audio. That is the argument to make to the
organizers, and it is why `legal_eval.vpc` reads the cache instead of re-running
extraction.

## Running it

```sh
pip install "voice-anonymization-legal-eval[vpc]"

legal-eval vpc \
    --results-dir exp/asv_anon_mcadams \
    --distractors train-clean-360_mcadams \
    --conversation-lengths 1,3 \
    --output results/legal_mcadams.csv
```

Dataset pairs are discovered automatically: `libri_dev_enrolls<suffix>` is
matched to `libri_dev_trials_mixed<suffix>`, and likewise for `libri_test`. The
output has one row per (dataset, metric, L, population size).

From Python, if you want the pieces:

```python
from legal_eval.vpc import benchmark_vpc_run
rows = benchmark_vpc_run("exp/asv_anon_mcadams", distractor_dataset="train-clean-360_mcadams")
```

## Getting the baselines run

Four Track 1 baselines, with anonymization costs quoted in VPC's own
`02_run_track1.sh`:

| Baseline | Config | Anonymization |
|---|---|---|
| B2 mcadams | `configs/track1/anon_mcadams.yaml` | ~2 h |
| B3 sttts | `configs/track1/anon_sttts.yaml` | ~13 h |
| B4 nac | `configs/track1/anon_nac.yaml` | not stated |
| B5 asrbn | `configs/track1/anon_asrbn.yaml` | ~1 h |

On top of each, `eval_post.yaml` finetunes an ECAPA-SSL attacker on the
anonymized `train-clean-360` for 2 epochs. Budget roughly a day per baseline on
one modern GPU, so a few days for all four. This is far from the GPU-weeks that
reproducing the paper's own pipeline would need.

```sh
git clone https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2026
cd Voice-Privacy-Challenge-2026
./00_install.sh
./01_download_data_model_track1.sh
./02_run_track1.sh configs/track1/anon_mcadams.yaml
```

Two practical notes:

- `01_download_data_model_track1.sh` **exits 1 without IEMOCAP**, which needs a
  signed licence. IEMOCAP is only used for the SER utility metric, so it is not
  needed for any privacy number. Either obtain it or run the privacy steps
  directly.
- The evaluation data, the pretrained ASV/ASR/SER models and LibriSpeech are all
  downloaded by that script. Only IEMOCAP is gated.

## Running it on cloud compute

[`gcp-benchmark.md`](gcp-benchmark.md) has a GCP plan: machine shape, staging the
corpus once in a bucket, spot instances, costs, and the order to run the
baselines in. The long-lead item is GPU quota, which a fresh project does not
have, so request it before anything else.

## Reading the result

### Population size is the thing to be careful about

The paper sweeps the number of speakers an attacker must search up to 22,024.
VPC's LibriSpeech dev and test sets hold around 40 speakers each. At `N' = 40`
the Linkability chance level is already 2.5%, and the population axis that
carries most of the paper's argument barely exists.

This is why `--distractors` matters. Passing `train-clean-360<suffix>` adds its
921 speakers to the enrollment population: they enlarge the search space without
ever being the correct answer, which is exactly their role in the metric. The
recipe has already anonymized and embedded that set to train the attacker, so
they come for free.

Even then, 961 speakers is not 22,024. **Report VPC numbers as a different
operating point, not as a replication of the paper.**

### The attacker naming differs

| VPC | This paper |
|---|---|
| Semi-informed (attacker trained on anonymized data) | **Informed** |
| Ignorant | Ignorant |

VPC 2026 Track 1 is the semi-informed condition, so its numbers correspond to
the paper's *Informed* attacker, which is the worst case and the right one for a
compliance argument.

### There is already a "linkability" in the VPC repo

`evaluation/privacy/asv/metrics/linkability.py` computes the Gomez-Barrero
`D_sys`, which measures the separation of mated and nonmated score
distributions. The metric here is a different quantity: a closed-set
identification rate, as the paper's footnote states explicitly.

If both end up in one repository they must not both be called "Linkability".
Settle the naming with the organizers before writing the integration, not during
review. See [metrics.md](metrics.md#linkability-pi_link).

## What a result would mean

The interesting outcome is a **ranking change**. If the four baselines order
differently under Singling Out or Linkability than under the EER, that is direct
evidence for the paper's claim, measured on systems the authors did not design
and with no new data collection.

If the ranking is unchanged but the spread is much wider, that is still worth
reporting: it says the EER compresses differences that matter legally.

Report alongside each number: the conversation length, the population size, the
attacker, and whether distractors were used. A legal metric without those four is
not interpretable.
