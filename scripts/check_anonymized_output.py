"""Is the ASR-BN output speech, or did the system remove the speech?

A system that emits silence is rewarded for it. A silent recording produces an
embedding that sits at chance distance from every enrolled speaker, and chance
is the best privacy score attainable, so the evaluation reports perfect privacy
for a destroyed file. The 2026 McAdams baseline did exactly this to one Common
Voice utterance: 93,487 audible samples in, 67 out, 99.93% exact zeros. No
evaluation in the field checks for it.

Two passes, for two different failures.

A size pass over every file, because truncation is cheap to detect and a
truncated file is not detectable from a sample. A wav of the wrong length is
wrong whatever its contents.

A deep pass over a random sample, compared against the original recording,
because reading 255,580 files costs about twenty minutes and repeats work the
extractor does anyway. A sample of a few thousand cannot prove every file is
sound, and this script does not claim that. It is sized to catch a systematic
failure, which is the failure that would invalidate the experiment. The sample
is seeded so the claim is reproducible, and every file it rejects is named.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from legal_eval.audio_integrity import check_anonymized, optimistic_bias_note

#: Kaldi scp files hold paths relative to the directory the pipeline was run
#: from, not relative to the scp itself, so the root has to be supplied.

def _reader():
    """The audio reader, imported at its point of use.

    Kept out of the module header so that importing this script to test its
    path handling does not require an audio library, which is the same reason
    the package defers scipy, h5py and torch.
    """
    try:
        import soundfile
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Checking anonymized audio requires soundfile. Install it with "
            "`pip install soundfile`, or `pip install "
            "voice-anonymization-legal-eval[audio]`."
        ) from exc
    return soundfile


#: Below this many bytes a 16 kHz 16-bit wav holds under a tenth of a second,
#: which is not an utterance whatever else is true of it.
MIN_BYTES = 3200


def read_scp(path: Path) -> dict[str, str]:
    """Utterance id to audio path, from a Kaldi scp."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            utt, rest = line.split(None, 1)
            out[utt] = rest.strip()
    return out


def resolve(path: str, root: Path) -> Path | None:
    """A plain path, or None when the entry is a pipe.

    Kaldi extended filenames are commands rather than paths, and handing one to
    a reader that expects a file is a defect this project has hit at four
    separate sites. Returning None rather than raising lets a mixed scp be
    checked for the entries that can be checked.

    Args:
        path: One scp entry's value.
        root: Directory that a relative path resolves against.

    Returns:
        An absolute path, or None for a pipe.
    """
    if "|" in path:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True,
                    help="anonymization suffix, as the data directory spells it")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="unanonymized data directory names under <root>/data")
    ap.add_argument("--root", type=Path, default=None,
                    help="directory holding data/, and what relative scp "
                         "paths resolve against")
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    # Resolved here rather than as an argparse default, which would freeze the
    # working directory at import time.
    if args.root is None:
        args.root = Path.cwd()

    sf = _reader()
    report: dict[str, dict] = {}
    worst = 0
    for dataset in args.datasets:
        anon_dir = args.root / "data" / f"{dataset}_{args.tag}"
        orig_dir = args.root / "data" / dataset
        if not (anon_dir / "wav.scp").exists():
            print(f"{dataset}: no anonymized wav.scp, skipping")
            continue
        anon = read_scp(anon_dir / "wav.scp")
        orig = (read_scp(orig_dir / "wav.scp")
                if (orig_dir / "wav.scp").exists() else {})
        print(f"\n=== {dataset}_{args.tag}: {len(anon)} utterances ===")

        # Pass one: every file, by size.
        tiny, missing = [], []
        for utt, path in anon.items():
            resolved = resolve(path, args.root)
            if resolved is None:
                continue
            try:
                size = resolved.stat().st_size
            except OSError:
                missing.append(utt)
                continue
            if size < MIN_BYTES:
                tiny.append((utt, size))
        print(f"  size pass over {len(anon)} files: {len(missing)} missing, "
              f"{len(tiny)} under {MIN_BYTES} bytes")
        for utt, size in tiny[:10]:
            print(f"    {utt}: {size} bytes")
        for utt in missing[:10]:
            print(f"    {utt}: no such file")

        # Pass two: a seeded random sample, against the original.
        rng = random.Random(args.seed)
        chosen = sorted(anon)
        rng.shuffle(chosen)
        chosen = chosen[:args.sample]
        unusable, actives, ratios = [], [], []
        for utt in chosen:
            a_path = resolve(anon[utt], args.root)
            o_path = resolve(orig.get(utt, ""), args.root)
            if a_path is None:
                continue
            try:
                # resolve() returns None for a pipe entry and those are
                # skipped above, so the reader only ever sees a plain path.
                # scp-reader-ok: pipes cannot reach this call
                a_signal, _ = sf.read(a_path)
            except Exception as exc:                      # noqa: BLE001
                unusable.append((utt, f"unreadable: {exc}"))
                continue
            o_signal = None
            if o_path is not None and o_path.exists():
                try:
                    # scp-reader-ok: as above, pipes cannot reach this call
                    o_signal, _ = sf.read(o_path)
                except Exception:                         # noqa: BLE001
                    o_signal = None
            result = check_anonymized(a_signal, o_signal)
            actives.append(result.active)
            if result.reference_active:
                ratios.append(result.active / result.reference_active)
            if not result.usable:
                unusable.append((utt, "; ".join(result.reasons)))

        active = np.asarray(actives) if actives else np.zeros(1)
        ratio = np.asarray(ratios) if ratios else np.zeros(1)
        print(f"  deep pass over {len(actives)} sampled files "
              f"(seed {args.seed})")
        print(f"    active fraction  min {active.min():.4f}  "
              f"median {np.median(active):.4f}  max {active.max():.4f}")
        if ratios:
            print(f"    active vs original  min {ratio.min():.4f}  "
                  f"median {np.median(ratio):.4f}  max {ratio.max():.4f}")
        print(f"    {len(unusable)} judged not speech")
        for utt, why in unusable[:10]:
            print(f"      {utt}: {why}")

        bad = len(tiny) + len(missing) + len(unusable)
        worst = max(worst, bad)
        if bad:
            print("    " + optimistic_bias_note(bad, len(anon)))
        report[f"{dataset}_{args.tag}"] = {
            "utterances": len(anon),
            "missing": missing,
            "undersized": [{"utterance": u, "bytes": s} for u, s in tiny],
            "sampled": len(actives),
            "sample_seed": args.seed,
            "active_min": float(active.min()),
            "active_median": float(np.median(active)),
            "ratio_median": float(np.median(ratio)) if ratios else None,
            "unusable": [{"utterance": u, "reason": w} for u, w in unusable],
        }

    if not report:
        print("nothing checked", file=sys.stderr)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")

    if worst:
        print("\nSome output is not speech. Those utterances must be excluded "
              "and the exclusion")
        print("declared with the population, or every metric reads more "
              "private than the truth.")
        # Not a failure of this script: the chain should continue and the
        # exclusion is handled where the population is recorded. Returning
        # non-zero here would stop the experiment over one bad file.
    else:
        print("\nEvery file passed the size pass and every sampled file is "
              "speech. The sample")
        print("does not prove the whole set is sound, only that no systematic "
              "failure is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
