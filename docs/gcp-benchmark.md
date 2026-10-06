# Running the VPC benchmark on GCP

A plan for computing Singling Out and Linkability for the four VoicePrivacy
Challenge 2026 Track 1 baselines on Google Cloud.

Costs below are order-of-magnitude estimates from list prices. Check the pricing
calculator before committing: GPU prices vary by region and change.

## Do this first: request GPU quota

A fresh project has **zero** GPU quota, and an increase can take anywhere from
minutes to days. It is the only step with an unpredictable lead time, so start
it before anything else.

```sh
gcloud auth login
gcloud config set project YOUR_PROJECT
gcloud services enable compute.googleapis.com storage.googleapis.com

# What you currently have.
gcloud compute regions describe us-central1 \
    --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -i gpu
```

Request through **IAM and Admin, Quotas** in the console: metric
`NVIDIA_L4_GPUS` (and `PREEMPTIBLE_NVIDIA_L4_GPUS` if you intend to use spot) in
your chosen region, limit 1. Say it is for academic speech privacy research.

While waiting, do everything in "Stage the data" below. None of it needs a GPU.

## What actually has to run

Per baseline:

1. Anonymize the LibriSpeech evaluation sets and `train-clean-360`.
2. Finetune the ECAPA-SSL attacker on the anonymized `train-clean-360`, two epochs.
3. Run the ASV evaluation, which caches speaker embeddings to disk.
4. Run `legal-eval vpc` over that cache. Seconds, on CPU.

**Only the privacy track is needed.** ASR word error rate and SER unweighted
average recall are utility metrics and do not feed either legal metric. Skipping
them matters for two reasons: it removes a large amount of compute, and it
removes IEMOCAP, which is licence-gated and whose absence makes
`01_download_data_model_track1.sh` exit 1.

`configs/track1/eval_post.yaml` already declares `eval_steps: privacy: [asv]`, so
the minimal run is anonymization plus that one evaluation config:

```sh
python run_anonymization.py --config configs/track1/anon_mcadams.yaml
python run_evaluation.py --config configs/track1/eval_post.yaml \
    --overwrite '{"anon_data_suffix": "_mcadams"}'
```

Anonymization costs quoted in VPC's own `02_run_track1.sh`: B2 mcadams about
2 hours, B3 sttts about 13 hours, B5 asrbn about 1 hour, B4 nac unstated. Add
the attacker finetune on top of each.

## Machine shape

| Choice | Reason |
|---|---|
| `g2-standard-32` (1x L4, 32 vCPU) | Anonymization is largely CPU-bound and parallel, so vCPUs matter as much as the GPU. The L4's 24 GB is the constraint to watch when finetuning WavLM-Large; drop `batch_size` in `eval_post.yaml` if it will not fit. |
| `a2-highgpu-1g` (1x A100 40 GB) | Fall back here if the L4 runs out of memory. Faster, roughly four times the price. |
| 1 TB balanced persistent disk | LibriSpeech `train-clean-360` is about 23 GB as FLAC, and each baseline's anonymized copy is roughly 40 GB as 16 kHz 16-bit WAV. Four baselines plus the originals will not fit on a small disk. |
| Deep Learning VM image | CUDA and drivers preinstalled. Building them by hand wastes a day. |

```sh
gcloud compute instances create vpc-bench \
    --zone=us-central1-a \
    --machine-type=g2-standard-32 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=common-cu124-ubuntu-2204-py310 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=1000GB \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

### On spot instances

Spot cuts the bill by roughly two thirds but the VM can be reclaimed at any
time. That is survivable here because VPC's `--force_compute False` skips work
whose output already exists, so a restarted run resumes rather than starting
over. Put the working directory on a **separate** persistent disk that is not
deleted with the instance, or the saving is illusory.

Add `--provisioning-model=SPOT --instance-termination-action=STOP`.

## Stage the data once

Downloading LibriSpeech onto every VM is wasted time and egress. Put it in a
bucket in the same region and copy from there.

```sh
gcloud storage buckets create gs://YOUR-BUCKET --location=us-central1

curl -O https://www.openslr.org/resources/12/train-clean-360.tar.gz
gcloud storage cp train-clean-360.tar.gz gs://YOUR-BUCKET/corpora/
# Also the VPC evaluation data and pretrained models, from the URLs in
# 01_download_data_model_track1.sh.
```

On the VM, `gcloud storage cp` from the bucket is an order of magnitude faster
than fetching from openslr, and costs nothing within the region.

## Cost, roughly

Assume about a day per baseline, so about 96 hours for four.

| | On demand | Spot |
|---|---|---|
| `g2-standard-32`, 96 h | roughly 170 to 200 USD | roughly 60 to 70 USD |
| `a2-highgpu-1g`, 96 h | roughly 350 to 400 USD | roughly 120 to 150 USD |
| 1 TB balanced disk, one month | roughly 100 USD | same |

So the whole benchmark is a few hundred dollars at most, and well under a
hundred on spot if the disk is released promptly. Set a **billing budget alert**
before starting, and delete the disk when the results are copied off. An idle
1 TB disk costs more over a year than the compute did.

## Suggested order

1. **Request quota.** Unpredictable lead time, so start it now.
2. **Stage the data to a bucket.** No GPU needed.
3. **One baseline end to end: B5 asrbn.** Cheapest anonymization at about an
   hour, so it validates the whole path for the least money. Do not start the
   other three until `legal-eval vpc` has produced a plausible results CSV for
   this one.
4. **The remaining three.** B2 mcadams, then B4 nac, then B3 sttts last since it
   is the most expensive.
5. **Collect and compare.** Copy every `results/legal_*.csv` off the VM, delete
   the instance and the disk.

Step 3 is the one to be disciplined about. The expensive mistake is running all
four and discovering afterwards that the embeddings were cached somewhere the
adapter did not look, or that the population size made the numbers
uninterpretable.

## Checking the result as it arrives

```sh
pip install "voice-anonymization-legal-eval[vpc]"
legal-eval vpc --results-dir exp/asv_anon_asrbn \
    --distractors train-clean-360_asrbn \
    --output results/legal_asrbn.csv
```

Sanity checks before trusting a number:

- Linkability at the smallest population should be well above `1/N'`, and should
  fall as the population grows. A flat curve means the population is not
  actually varying.
- Singling Out should sit near `exp(-1) = 0.368` for a weak attacker and above
  it for a strong one. Far below 0.368 everywhere suggests the calibration set
  is not what you think it is.
- `1 - EER` should barely move across population sizes. That it does not move is
  the paper's point, and it also confirms the sweep is wired up.

Remember to pass `--distractors`, and to report the conversation length,
population size, attacker and distractor status with every number. See
[`vpc-benchmark.md`](vpc-benchmark.md) for why.
