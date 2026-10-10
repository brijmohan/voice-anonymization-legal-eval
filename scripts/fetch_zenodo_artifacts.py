#!/usr/bin/env python3
"""Fetch the published score matrices from Zenodo, verifying every file.

The 5.34 GB of score matrices behind
[10.5281/zenodo.23142030](https://doi.org/10.5281/zenodo.23142030) are the
sufficient statistic for every metric in this project except Singling Out, so
anyone reproducing the published numbers needs them and nobody should keep them
in git.

This script makes that fetch reproducible and checkable:

* every file is verified against the md5 Zenodo published, recorded in
  ``data/zenodo-checksums.json``, so a truncated or substituted download fails
  loudly instead of producing wrong numbers quietly;
* it is idempotent, so an interrupted fetch resumes by skipping what already
  verifies;
* it never writes outside the output directory.

Usage::

    python scripts/fetch_zenodo_artifacts.py --output-dir data/zenodo
    python scripts/fetch_zenodo_artifacts.py --verify-only
    python scripts/fetch_zenodo_artifacts.py --only MANIFEST.json cv11-A-filelist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

CHECKSUMS = Path(__file__).resolve().parents[1] / "data" / "zenodo-checksums.json"
CHUNK = 1 << 20


def md5_of(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - matching Zenodo's published algorithm
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_md5: str, expected_size: int) -> tuple[bool, str]:
    """Whether a local file matches what Zenodo published."""
    if not path.exists():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return False, f"size {actual_size} != {expected_size}"
    actual = md5_of(path)
    if actual != expected_md5:
        return False, f"md5 {actual[:12]} != {expected_md5[:12]}"
    return True, "ok"


def download(url: str, destination: Path) -> None:
    """Stream to a temporary file, then move, so a partial file is never used."""
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(partial, "wb") as handle:  # noqa: S310
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/zenodo"))
    parser.add_argument("--verify-only", action="store_true",
                        help="check what is present, download nothing")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these filenames")
    args = parser.parse_args()

    spec = json.loads(CHECKSUMS.read_text())
    files = spec["files"]
    if args.only:
        wanted = set(args.only)
        files = [f for f in files if f["key"] in wanted]
        missing = wanted - {f["key"] for f in files}
        if missing:
            print(f"not in the record: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    print(f"record {spec['record']}, {len(files)} file(s), "
          f"{sum(f['size'] for f in files):,} bytes")
    ok = bad = fetched = 0
    for entry in files:
        destination = args.output_dir / entry["key"]
        good, why = verify(destination, entry["md5"], entry["size"])
        if good:
            print(f"  [ok]      {entry['key']}")
            ok += 1
            continue
        if args.verify_only:
            print(f"  [{why}] {entry['key']}")
            bad += 1
            continue
        print(f"  [fetch]   {entry['key']}  ({entry['size']:,} bytes, {why})")
        download(entry["url"], destination)
        good, why = verify(destination, entry["md5"], entry["size"])
        if not good:
            print(f"  [FAILED]  {entry['key']}: {why}", file=sys.stderr)
            bad += 1
        else:
            fetched += 1

    print(f"\nverified {ok + fetched}, fetched {fetched}, failed {bad}")
    if bad:
        print("A checksum mismatch means the local copy is not what was "
              "published. Do not compute results from it.", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
