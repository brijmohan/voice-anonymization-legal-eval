# Does our setup conform to the VoicePrivacy 2026 evaluation plan?

Checked against the plan, version 2.0, 1 September 2026
(`VPC_2026_sept01.pdf`), by reading it against what is actually installed and
what was actually run **[M]**.

Two findings matter. Our attacker is **mislabelled**, and we are missing one of
the four Track 1 baselines. Neither changes a computed number; the first
changes what we are entitled to say about one.

## The attacker is lazy-informed, and we called it ignorant

The plan's submission table maps the directories explicitly:

| Plan | Directory | Used for ranking |
|---|---|---|
| ASV (lazy-informed) | `exp/asv_ssl/results*<suffix>.csv` | No, "post-evaluation analysis only" |
| ASV (semi-informed) | `exp/asv_anon<suffix>/` | **Yes** |

Everything we labelled *ignorant* came from `exp/asv_ssl`, which the plan calls
**lazy-informed**. The challenge's own config agrees: `eval_post_privacy.yaml`
opens with "semi-informed attacker: ECAPA-SSL finetuned on anonymized
train-clean-360", and `eval_pre_privacy.yaml` uses `exp/asv_ssl` untouched.

The distinction is not cosmetic. In the taxonomy, an **ignorant** attacker does
not know the anonymization system and so does not anonymize its enrollment
data, leaving a mismatch that flatters privacy. A **lazy-informed** attacker
knows the system and anonymizes enrollment to remove that mismatch, without
retraining the ASV. Our comparison built systems from the `oo` and `aa`
scenarios only, so the anonymized side is anonymized enrollment against
anonymized trials: lazy-informed.

**Direction of the error.** Lazy-informed is the *stronger* attack of the two,
so our numbers come from a stronger attacker than the label implied. We were
conservative, not optimistic. That is the better way to be wrong, and it is
still a misstatement, and it bears directly on CNIL obligation G2, which
requires risk quantified against a concrete attacker with stated capability.

**We never computed the ignorant condition at all.** It is the `oa` scenario,
original enrollment against anonymized trials, and `build_vpc_systems` filters
it out deliberately. If we want an ignorant number we have to ask for it.

Corrected in this repository. The result files on the compute host still carry
the old names and are listed below.

## Baseline coverage

The plan's Track 1 baselines, with `B1` removed and `B6` excluded this edition:

| ID | What it is | Our name | Status |
|---|---|---|---|
| B2 | McAdams coefficients, DSP only | `mcadams` | run, both attackers, plus Common Voice |
| B3 | Phone aligner + Praat, E2E ASR, GST, FastSpeech2 + HiFi-GAN, GAN speaker anon | `sttts` | run on LibriSpeech, both attackers. Common Voice stopped on throughput |
| B4 | HuBERT Base quantized semantic encoder + EnCodec, selection | `nac` | **not run** |
| B5 | YAAPT, wav2vec2 + TDNN-F + VQ, one-hot speaker, HiFi-GAN, selection | `asrbn` | run on LibriSpeech, both attackers |

**B4 is missing**, dropped on cost: the projected wall time was 188 days on the
hardware available **[M]**. Any statement of the form "across the VPC 2026
baselines" therefore covers three of four and must say so.

## Evaluation model

The plan specifies, for Track 1 ASV:

> WavLM-ECAPA (WavLM-large weighted features + Fbank), WavLM pre-trained on 94k
> h from LibriLight, VoxPopuli and GigaSpeech, further trained on about 9k h of
> voice-converted data from the Source Speaker Tracing Challenge, then
> fine-tuned on language-specific anonymized speech.

What we run is `model_type: ecapa_ssl` against `exp/asv_ssl`, holding
`WavLM-Large.pt` at 1,261,965,425 bytes plus `embedding_model.ckpt`, and its
extractor combines a filterbank stream with a WavLM stream **[M]**. That is the
specified model, as shipped by the challenge. **Conforms.**

One property of it is worth recording here because it is a property of the
*official* evaluation model and not of our code. The filterbank stream receives
audio that has been loudness-normalised; the WavLM stream re-reads the file
from disk and does not. Constitution section 3.7 has the detail. It does not
bias a comparison, since the same attacker is applied to every condition, but
it is a caution for corpora whose level varies more than read audiobook
speech, which includes Common Voice.

## Trial lists

The plan is new here: "in addition to same-gender trials, cross-gender trials
are also considered. Together, these are referred to as **mixed**", and "the
EER is calculated using the Mixed trials".

We use `libri_dev_trials_mixed` and `libri_test_trials_mixed`. **Conforms.**

This matters more than it looks, because the published target-leakage result
(arXiv:2508.09803) shows that same-gender target selection overestimates
privacy. The 2026 plan moving to mixed trials is the challenge correcting for
the same effect.

## Evaluation data

| Plan | Ours |
|---|---|
| LibriSpeech dev and test subsets | used |
| IEMOCAP dev and test | **not used** |
| train-clean-360, anonymized, for the semi-informed attacker | used |

We ran the `_noiemocap` configs. So our LibriSpeech numbers are directly
comparable to the plan's LibriSpeech numbers, and our evaluation is **not** a
complete Track 1 evaluation. An EER we quote is not a Track 1 submission EER.

## Metrics

| Plan | Ours |
|---|---|
| Privacy: ASV EER, semi-informed, used for ranking | computed |
| Privacy: ASV EER, lazy-informed, analysis only | computed |
| Utility: ASR WER, wav2vec2 on train-960 | **not run** |
| Utility: SER UAR, wav2vec2 on IEMOCAP | **not run** |

The two utility metrics are out of scope for this project, which is about
whether a privacy number means what it claims. Worth being explicit that
skipping them means we cannot speak to the privacy and utility trade-off at
all, and several of the interesting claims in the field are about that
trade-off.

## What this audit does not establish

It compares our configuration against the plan. It does not verify that our
installed checkpoints are byte-identical to the ones the organisers published,
only that the paths, model type and architecture are the specified ones. A
checksum manifest from the organisers would settle it and we do not have one.

## Summary

| | |
|---|---|
| Attacker architecture | conforms |
| Attacker labelling | **was wrong**, corrected here |
| Ignorant condition | never computed |
| Semi-informed condition | conforms, all three systems run |
| Trial lists | conforms, mixed |
| Baselines | 3 of 4, B4 missing |
| Evaluation data | LibriSpeech only, no IEMOCAP |
| Utility metrics | not run |
