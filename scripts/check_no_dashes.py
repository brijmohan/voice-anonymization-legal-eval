#!/usr/bin/env python3
"""Fail if any tracked file uses an em dash or en dash as punctuation.

The repository's writing rule forbids both. The obvious one-liner for this is
``git ls-files -z | xargs -0 grep -nP '[\\x{2014}\\x{2013}]'``, which needs a
PCRE-capable grep. BSD grep, which is what macOS ships, has no ``-P`` and exits
with a usage error, and a usage error in a pipeline reads as "nothing found".
That is how a dash reached ``CONTRIBUTING.md`` and stayed there. This script
does the same job without depending on which grep is installed.
"""

from __future__ import annotations

import subprocess
import sys

# Built from code points so that this file does not trip its own check.
FORBIDDEN = {chr(0x2014): "em dash", chr(0x2013): "en dash"}

# Binary and vendored paths that are not prose and not ours to rewrite.
SKIP_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".pt", ".npy", ".npz", ".h5",
                 ".woff", ".woff2", ".ico", ".gz", ".zip")


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split("\0") if p]


def main() -> int:
    findings = []
    checked = 0
    for path in tracked_files():
        if path.lower().endswith(SKIP_SUFFIXES):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except (UnicodeDecodeError, OSError):
            continue
        checked += 1
        for number, line in enumerate(lines, 1):
            for character, name in FORBIDDEN.items():
                if character in line:
                    findings.append((path, number, name, line.strip()[:100]))

    if findings:
        print(f"Found {len(findings)} forbidden dash(es):\n", file=sys.stderr)
        for path, number, name, text in findings:
            print(f"  {path}:{number}: {name}: {text}", file=sys.stderr)
        print(
            "\nRewrite the sentence rather than substituting another character. "
            "A spaced hyphen is the same habit in disguise. See CLAUDE.md.",
            file=sys.stderr,
        )
        return 1

    print(f"checked {checked} tracked text files, no em or en dashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
