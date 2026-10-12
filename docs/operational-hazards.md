# Operational hazards, with the test or habit that catches each

> **The mechanical ones are now a command.** `legal-eval preflight
> results.json` runs every hazard that can be judged from a run's own output,
> and names the failure each one guards against. Run it before believing a
> result and before sending one to anybody. The entries below are the whole
> record, including the hazards no check can reach; the preflight covers the
> subset it can.
>
> Each failure here produced a plausible number, not an error. That is the
> reason for the command: nothing raised, and the worst of them reached a draft.

Every entry here cost real time on this project. They are recorded because
several recurred after being fixed once, which means the fix was not the
problem: not writing it down was.

Grouped by whether a test can catch it or only a habit can.

---

## Caught by a test

### A standard error of exactly zero

**What happened.** The metric comparison reported that LID resolved a pair the
rank metrics could not, and separately an EER-versus-LID inversion. Both were
artifacts. The spread came from the closed-set draw alone, and when `N` equals
the number of enrollment speakers that draw is deterministic, so the standard
error was exactly `0.0000` and every difference looked resolved.

**The rule.** An exactly-zero spread is a signal that the resampling is not
capturing the right source of variation, not a sign of precision.

**Test.** `tests/test_comparison.py` resamples trials as well as closed sets.
`legal_eval/pseudo_speaker.py` carries its own chance level so a rate cannot be
read without one.

### Noise that does not scale with dimension

**What happened, twice.** A fixture added `N(0, 0.35)` per dimension to
unit-norm centres at dim 256. The noise norm is `0.35 * sqrt(256) = 5.6`, so the
clusters vanished, mated and nonmated similarity coincided, and the fixture's
verifier sat at EER 0.38. Every threshold then admitted everything and the
pseudo-speaker audit appeared to fire on unrelated vectors. The same mistake had
already been made in the demo scatter.

**The rule.** In high dimensions, per-dimension scale is not perceptual scale.
Divide by `sqrt(dim)` whenever noise is meant to be a fraction of a vector's
norm.

**Test.** `tests/test_pseudo_speaker.py` asserts the fixture's own verifier
reaches EER below 0.01, so a broken fixture fails loudly instead of making the
measurement look permissive. (That bound is a property of the fixture, not a
result: no marker applies.)

### A classifier that memorises the speaker, not the attribute

**What happened.** The first attribute-inference draft pooled many rows per
speaker. A speaker appearing in both folds lets the probe recognise the speaker
and score well on the attribute for the wrong reason, reporting identity
leakage as attribute leakage.

**Test.** `tests/test_inference.py` compares grouped against ungrouped on
features that encode only the speaker.

### Reporting the majority class as the exposed one

**What happened.** `AttributeDisclosure.summary` reported the best per-class
recall, which is almost always the majority class, which is exactly what the
prior already predicts. The legally salient case is a rare value recovered well.

**Test.** `tests/test_inference.py` plants a strongly encoded rare class and
asserts the summary names it.

### A silent intersection that invalidates a measurement

**What happened.** The sharding harness intersected the requested utterance list
with the dataset. Asking for 400 utterances of a 48 utterance dataset produced
48 and said nothing, so the benchmark was meaningless without failing.

**The rule.** A filter that silently returns less than asked is worse than a
crash. Refuse.

### Library readers that assume a plain path

**What happened.** A `wav.scp` value is not always a path. Kaldi allows an
extended filename, a shell pipeline ending in `|`, which is how a corpus
distributed as mp3 is addressed without transcoding it first. `soundfile.read`
raises `TypeError: Invalid file` on one, and `torchaudio.load` raises too.

Four separate sites in one pipeline had this defect:

1. `anonymization/modules/sttts/text/recognition/ims_asr.py`
2. `anonymization/modules/sttts/prosody/extraction/ims_prosody_extraction.py`
3. `evaluation/privacy/asv/speaker_extraction.py`
4. `evaluation/privacy/asv/speechbrain_vectors.py`

Site 4 is the one worth remembering. It is reached from site 3, it re-reads the
file site 3 has just decoded, and so fixing site 3 did not fix it: the smoke
test still died on the first utterance, after the pipeline had been declared
clean.

There is a quieter half. `read_kaldi_format` returns a pipe entry as a list of
tokens, and site 3 narrowed it with a literal index, `wav_path = wav_path[1]`.
For a pipe entry token 1 is `-v`, an ffmpeg flag. That does not raise where it
is written, which makes it worse than the loud version.

**The test.** `scripts/check_scp_readers.py` walks a pipeline checkout for both
halves, with an explicit waiver comment for a reviewed call:

```sh
python scripts/check_scp_readers.py --root /path/to/pipeline
```

`tests/test_check_scp_readers.py` pins it against the shape of all four sites.
This started as a habit, "grep the pipeline before running it", and the habit
failed twice in a row, which is the reason it is now a script.

### Intermediate files named by extending a real name

**What happened.** Sharded embedding extraction writes one directory per
shard. They were named by extending the dataset name,
`cv27_enrolls_mcadams_sh0` beside `cv27_enrolls_mcadams`, which is the obvious
scheme and also makes them satisfy the naming pattern
`legal_eval.vpc.pair_datasets` recognises. So the shards **paired with each
other**, and the comparison produced a complete, entirely fictitious system per
shard.

Twenty-four leftovers, twelve spurious systems, each a sixth of the data. It
was noticed only because a progress line read `=== cv27 mcadams_sh0 ===`. The
wasted compute was the smaller cost: the output would have carried twelve
systems that do not exist, and the partial order over systems would have been
drawn partly from them.

Note which guard did not fire. `unpaired_datasets`, added the same day for the
opposite failure, finds datasets that pair with nothing. These pair perfectly.

**The test.** `legal_eval.vpc.suspicious_datasets`, which the comparison calls
and reports on. Two rules, because the first is a guess about naming and the
second is not:

- the name matches an intermediate pattern: `_sh0`, `_shard0`, `_part0`,
  `temp`, `tmp`, `smoke`, `warm`
- its system suffix strictly **extends** another present suffix, which catches
  a scheme nobody anticipated

**The habit that would also have worked.** Delete or move intermediates when
the step that produced them succeeds. The merge driver had no cleanup, so the
shards outlived their purpose by two hours and a wasted run.

### A merge that discards what it should keep

**What happened.** The ASR shard merge was written to overwrite the canonical
`text`, and a test asserted that as correct. But the pipeline's resume reads
only `text`, so a shard covers the remainder and discarding `text` loses exactly
the utterances an earlier run completed. The test was reversed.

**Test.** `tests/test_merge_asr_shards.py` asserts the canonical file is folded
in and that shard entries win on conflict.

---

## Caught only by a habit

### `pgrep -f` and `pkill -f` match their own command line

**Three occurrences, the third fatal.** `pgrep -f run_all.sh` reported a job was
already running when it was not. `pgrep -f "tar xzf common-voice"` reported a
finished extraction as still running. Then `pkill -TERM -f anon_sttts_cv27`
killed the SSH session carrying it, producing three `ssh exited with return code
[255]` failures that looked like a network outage while the workstation was
`RUNNING` and reachable.

**Fourth occurrence, and it finally settles which fix to use: the pid file.**

Asking "is `shard_extract.sh` running?" before editing it took three attempts,
each failing differently:

```sh
pgrep -f "shard_extract"        # matched this command's own line
pgrep -f "shard_extrac[t]"      # still matched: the command line names the
                                # script elsewhere, in the edit it performs
pgrep -f ... | grep -v "^$$\$"   # $$ was expanded by the LOCAL shell before
                                # the string ever reached the remote one
```

All three were false positives, and all three refused a safe edit. The pid file
worked first time:

```sh
pid=$(cat logs/thing.pid 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo running; fi
```

**The habit.** Use a pid file. Bracketing one character is a weaker fix with
its own two failure modes, shown above: it does nothing when the command line
mentions the target elsewhere, which a script that edits a file named after the
process always does, and a self-excluding `$$` is expanded on the wrong side of
a transport boundary. Bracketing is still worth it for a one-off interactive
check where neither applies.

```sh
pgrep -f "anon_sttts_cv2[7]"    # acceptable interactively
pkill -TERM -f "anon_sttts_cv2[7]"
```

### Nested heredocs and `$` expansion

**What happened.** An outer unquoted `<<EOF` containing an inner `<<'INNER'`
expanded `$BENCH` at write time, so the remote script sourced `/env.sh`.

**The habit.** Do not nest heredocs across a transport boundary. Write the
script locally, `base64` it, and decode on the far side. Every patch in this
project that worked first time used that pattern.

### `set -u` against someone else's environment script

**What happened.** Adding `set -u` to a runner aborted immediately, because
VPC's `env.sh` dereferences `LD_LIBRARY_PATH` before setting it.

**The habit.** `set -e` only when sourcing third-party shell. Their
`run_baseline.sh` does the same, for the same reason.

### zsh does not word-split unquoted parameters

**What happened.** `for p in $TARGETS; do kill $p; done` iterated once with the
entire string, so twelve processes produced one "could not signal". In bash this
works; in zsh it does not.

**The habit.** Pass explicit arguments, or use an array. This shell is zsh.

### `grep -P` is not portable, and fails open

**What happened.** The documented em-dash check was `grep -nP`. BSD grep, which
macOS ships, has no `-P` and exits with a usage error, which in a pipeline reads
as "nothing found". A dash reached `CONTRIBUTING.md` and survived several passes.

**The fix.** `scripts/check_no_dashes.py`, in CI, no grep.

### Testing the wrong command's exit status

**What happened.** A retry loop was written as
`if gcloud ... | grep -v noise; then break; fi`. That tests grep, which
succeeds, so the loop exited after one failed attempt.

**The habit.** Capture into a variable, test `$?` of the command that matters,
then filter for display.

### `wget -q` hides a failed download

**What happened.** Editing `install.sh` tripped VPC's checksum, which re-ran its
download block. `wget -q` failed silently, the GAN model vanished, and the next
run died on a missing `style-embed_wgan.pt` after the expensive work was done.

**The habit.** Never `-q` on a fetch whose absence breaks a later stage. Log it
and check the exit status.

### Editing an upstream file can trigger its reinstall

**What happened.** VPC hashes `install.sh` and reinstalls when it changes. Eight
concurrent shard workers each ran that install and clobbered `env_sttts.sh`.

**The habit.** `export NEW_INSTALL_TRIGGER=no` once installed, and warm up with
a single worker before launching many.

### Editing a shell script while it is running

**What happened.** The finding driver was mid-run when its `python` invocation
was patched to add `-u`, to fix the buffering hazard below. That edit changed
the length of a line, which shifts every byte after it.

Bash reads a script in chunks and keeps a byte offset into the file. If it
re-reads after the offset has been invalidated, it can resume mid-token and
execute something that was never written. The script here was 2 KB, small
enough that bash had almost certainly read it whole, and the run continued
normally. That is luck, not a method.

**The habit.** Write the new version to a different filename and launch that.
Never edit the file a running shell is reading, however small the change. The
same applies to a config a long job re-reads between stages.

This one was caught by noticing it rather than by anything failing, which is
the weakest way to catch something.

### Buffered output makes a long job indistinguishable from a hung one

**What happened.** Three times in one day, a job that was working looked dead.

Python buffers stdout when it is redirected rather than attached to a terminal,
so a 30 minute stage wrote nothing at all to its log. `grep` does the same in a
pipeline, so a remote `ssh ... | grep` returned an empty file while the command
behind it ran fine. In the worst case a watch loop treated the resulting empty
output as a success: its `case` matched the wildcard branch and it reported a
merge that had not happened.

**The habit.**

- `python -u` for anything whose output is redirected, always.
- `grep --line-buffered` and `awk ... fflush()` in any pipeline you intend to
  read while it runs.
- Judge liveness from `ps -o times,pcpu`, not from log output. Climbing CPU
  time means computing; silence means nothing either way.
- In a watch loop, an empty result is **not** a pass. Match the success
  condition explicitly and let everything else, including empty, fall through
  to "unknown". The Monitor rule of thumb applies: if the job died right now,
  would the filter emit anything?

### `nvidia-smi` reports no GPU when only `PATH` is set

**What happened.** On this workstation image the driver lives outside the
default paths. Setting `PATH=/var/lib/nvidia/bin:$PATH` makes the binary
findable, and it then fails on its own library:

```
NVIDIA-SMI couldn't find libnvidia-ml.so library in your system.
```

Earlier in the same project a `find -maxdepth 4` missed the binary altogether
and produced the claim that the GPU tooling was absent, which Brij corrected by
pasting his own `nvidia-smi` output. The second time it read as `GPU n/a` in a
status check and was nearly reported as an idle GPU.

**The habit.** Export both, which is what `env.sh` already does:

```sh
export PATH=/var/lib/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/var/lib/nvidia/lib64:$LD_LIBRARY_PATH
```

A GPU reading that says "not found" is a broken command until proven
otherwise. Never report it as an idle or absent GPU.

### One worker and many workers are different code paths

**What happened.** CV27 embedding extraction was smoke-tested twice at one
process, on both the original and the anonymized audio. Both passed. The first
real launch at six processes died on the first batch:

```
RuntimeError: Cowardly refusing to serialize non-leaf tensor which
requires_grad, since autograd does not support crossing process boundaries.
```

`tqdm.contrib.concurrent.process_map` pickles each worker's return value back
to the parent. The single-process branch calls the worker function directly and
pickles nothing, so the entire serialisation path, which is where the bug was,
had never run. The fix was `detach().cpu()` on the embedding, which changes no
value.

This is the second time a parallel launch failed for a reason a single worker
could not have surfaced. The first was `recognition_job` deriving its output
suffix from a `job_id` of `None`, where one worker wrote the canonical file and
eight would have overwritten each other.

**The habit.** Smoke test at the worker count you will actually run, on a slice
small enough to be cheap. "Warm up with one worker" is still right for catching
a bad path or a missing model, but it is not evidence that the parallel run
works, and twice it was mistaken for exactly that.

### A per-job suffix derived from `None`

**What happened.** `recognition_job` computes
`add_suffix = f'_{job_id}' if job_id is not None else None`, and the
single-process path passes `None`, so it writes the canonical `text` every 100
utterances. Eight shards would have overwritten each other, which is the exact
race the sharding design existed to avoid.

**The habit.** Before parallelising someone else's loop, find every write inside
it. The outer save was patched first and would not have been enough.

---

## Before launching anything long

0. `legal-eval preflight` on the last comparable result, so a known
   failure is not carried into a new run.

1. Warm up with **one** worker and read its first log lines, then warm up
   again **at the worker count you will run**. One worker does not exercise
   the parallel path, and twice it was taken as proof that it did.
2. Confirm the output lands where the design says, by name.
3. Check a pid file, never `pgrep -f`, for "is it already running".
4. Make sure a restart resumes rather than restarts, and prove it.
5. If the corpus is reached through a pipe, run
   `scripts/check_scp_readers.py` over the checkout, then confirm one entry
   decodes first. The check finds the readers; only a run finds the rest.
