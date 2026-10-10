# Making the GPU do the work

Measured on an NVIDIA A100-SXM4-40GB running the VoicePrivacy 2026 `sttts`
baseline over `train-clean-360`, 104,014 utterances.

## The measurement that started this

```
A100-SXM4-40GB   3546MiB / 40960MiB   0% util   56W / 400W   35C
vmstat            8% user, 1% sys, 91% idle, 0% iowait
worker           72.7% of ONE core, out of 12
```

Nothing was saturated. Not the GPU, not the CPU, and 0% iowait rules out the
Filestore mount. About 6% of the machine was in use, and a 104,014 utterance
stage was running at 2.0 to 2.5 utterances per second, roughly 13 hours.

## Why

Two findings in the upstream code, both structural rather than accidental.

**Parallelism is tied to the device count.** In every stage that has any:

```python
self.n_processes = len(self.devices)
```

One GPU gives one worker. This is the same shape as the `nac` pipeline's
`accelerate launch --num_processes=1`.

**The stage that dominates has no parallelism at all.** Only the ASR module
contains `n_processes`. `speech_synthesis.synthesize_speech` receives the whole
corpus and loops over it:

```python
if len(self.tts_models) == 1:
    instances = []
    for text, utt, speaker in texts:
        ...
    wavs.update(synthesis_job(instances=instances, ...))
```

There is an `utterance_list` parameter that looks built for sharding, but the
pipeline never passes it, and it exists only in the ASR module.

## The patch

Two properties of the upstream code make sharding safe, and both are worth
stating because they are the reason this is a small change rather than a
rewrite:

1. **Output is one file per utterance**, at
   `exp/anon_pipeline_<p>/anon_speech/<model>/<dataset>/<utt>.wav`. Concurrent
   writers touch disjoint paths.
2. **Synthesis is already idempotent.** It globs the output directory, removes
   already synthesised utterances from `texts`, and only then synthesises. So a
   worker that is restarted, or one that shares a dataset with another worker,
   does not redo finished work.

The patch adds a shard filter immediately after that resume step, driven by two
environment variables:

```sh
VPC_NUM_SHARDS=8 VPC_SHARD_INDEX=0 python run_anonymization.py --config ... &
VPC_NUM_SHARDS=8 VPC_SHARD_INDEX=1 python run_anonymization.py --config ... &
...
```

Environment variables rather than a new argument, because `synthesize_speech` is
called with fixed keyword arguments from the pipeline, so threading a parameter
through `run_anonymization.py` and the pipeline constructor would touch several
files of upstream code for no extra capability. The upstream file is backed up
beside itself as `speech_synthesis.py.orig`.

### Why the assignment is a hash of the id

The rule lives in [`legal_eval/sharding.py`](../legal_eval/sharding.py) with
tests, so it has one definition rather than existing only inside a patch to
someone else's repository. It must satisfy four things:

| Property | Why it matters | How |
|---|---|---|
| Disjoint | A duplicate means two workers synthesise one utterance and one overwrites the other | shard = f(id) is single valued |
| Complete | A miss means a short corpus, which no later stage checks | every id maps somewhere |
| Stable across processes | Workers are separate processes, and `hash()` of a str is salted per process, so it would give each worker a different split | `zlib.crc32` |
| Stable across restarts | The resume step removes finished utterances first, so the list is shorter on a resume. A position based split would reassign work | depends on the id, never its index |

The third and fourth are the ones that bite. Using `hash()` or
`enumerate()` here looks correct and silently corrupts the output.

### Two guards in the patch

`create_clean_dir` wipes the output directory. With several workers writing one
namespace, whichever starts last would delete the others' output, so when
sharding is active the patch only ensures the directory exists, and it refuses
to run at all if `force_compute` is set:

```
ValueError: force_compute cannot be combined with VPC_NUM SHARDS:
cleaning the shared output directory would delete the other shards' output
```

An out of range `VPC_SHARD_INDEX` is rejected rather than silently producing an
empty shard, which would look like a successful run that wrote nothing.

## Verification

The rule is pinned by `tests/test_sharding.py`, including a subprocess test
with a fixed `PYTHONHASHSEED` that fails if the implementation ever reverts to
`hash()`.

Checked against the real `train-clean-360` list, 104,014 utterances:

```
  4 shards: total=104014 union=104014 min=25973 max=26046 balance=1.0016
  8 shards: total=104014 union=104014 min=12983 max=13038 balance=1.0028
  resume: 16014 remaining split consistently with the cold start split
  sharding disabled -> no-op
  VPC_SHARD_INDEX=9 with VPC_NUM_SHARDS=4 -> rejected
```

Balance matters because a run ends when its slowest shard ends: 1.0028 means
under 0.3% of the wall clock is lost to the tail.

### What is not yet verified

**End to end throughput.** The patch's logic is verified, but no sharded run has
been timed, because the only dataset large enough to measure on is the one a
16 hour job was still writing to, and launching unsharded and sharded workers
into one output namespace would have corrupted it. The measurement is deferred
rather than skipped.

Sizing from the measurements above: 3.5 GB of GPU memory and 0.73 cores per
worker, so 8 workers is about 28 GB of 40 and 6 cores of 12. Both fit, and the
expectation is 5 to 8x. That expectation is an inference from resource headroom,
not a measurement, and it is labelled as such until it is one.

## Earlier approach, and why it was abandoned

The first attempt cloned the dataset into `<base>_shK` directories, since every
intermediate is namespaced per dataset name and would therefore not collide. It
failed, and the failure is instructive: the stages before synthesis store
aggregate tensors per dataset, several of them precomputed and downloaded, and
copying those confuses the resume bookkeeping.

```
No speech recognition necessary for 104014 of 400 utterances
Recognize speech of 400 utterances...
Job 0: 0it [00:00, ?it/s]
IndexError: list index out of range
```

It found 104,014 transcripts for a 400 utterance dataset, computed the remainder
as 400 rather than 0, and crashed saving an empty list. Sharding inside the
synthesis loop avoids all of this: there is one dataset, one set of
intermediates, and one output namespace.

A benchmark from that attempt was also invalid for an unrelated reason worth
recording: the harness intersected the requested utterance list with the
dataset, so a request for 400 utterances of a 48 utterance dataset produced 48
and reported nothing wrong. A silent intersection that invalidates a
measurement without failing is worse than a crash, and the harness now refuses
it.
