#!/usr/bin/env python3
"""Refuse to let internal or personal material into this public repository.

This exists because it nearly happened. PSST deliverables, including career
development plans naming doctoral candidates, and OneDrive exports of grant and
supervisory board documents were placed in ``docs/``. They were untracked but
not ignored, so the next ``git add -A`` would have published them.

Being careful is not a control. This is, and it runs in CI.

Exits non-zero and names every offending path.
"""

from __future__ import annotations

import re
import subprocess
import sys

#: Filename patterns that should never be tracked here.
FORBIDDEN_NAMES = [
    (re.compile(r"(?i)onedrive"), "OneDrive export, likely internal documents"),
    (re.compile(r"(?i)^psst"), "PSST deliverable, may name doctoral candidates"),
    (re.compile(r"(?i)career.?development"), "career development plan, personal data"),
    (re.compile(r"(?i)grant.?agreement"), "grant agreement, internal"),
    (re.compile(r"(?i)progress.?report"), "internal progress report"),
    (re.compile(r"(?i)supervisory.?board"), "supervisory board material"),
    (re.compile(r"(?i)\.(docx?|xlsx?|pptx?)$"), "office document, rarely belongs here"),
    (re.compile(r"(?i)\.zip$"), "archive, contents cannot be reviewed in a diff"),
]

#: Content patterns in text files. Deliberately narrow, to stay useful.
FORBIDDEN_CONTENT = [
    (re.compile(r"\b10116819\d\b"), "PSST grant number"),
    (re.compile(r"(?i)\bclient_id\s*[:=]\s*[0-9a-f]{40,}"), "Common Voice client id"),
]

#: Paths that legitimately mention the patterns above, such as this checker and
#: the ignore rules that implement the same policy.
ALLOWED = {
    "scripts/check_no_private_material.py",
    ".gitignore",
    "CLAUDE.md",
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def main() -> int:
    problems: list[str] = []
    for path in tracked_files():
        if path in ALLOWED:
            continue
        name = path.rsplit("/", 1)[-1]
        for pattern, why in FORBIDDEN_NAMES:
            if pattern.search(name):
                problems.append(f"  {path}\n      {why}")
                break
        else:
            try:
                with open(path, encoding="utf-8") as handle:
                    text = handle.read(200_000)
            except (UnicodeDecodeError, OSError):
                continue
            for pattern, why in FORBIDDEN_CONTENT:
                if pattern.search(text):
                    problems.append(f"  {path}\n      contains a {why}")
                    break

    if problems:
        print("Private or internal material is tracked in this public repository:\n")
        print("\n".join(problems))
        print(
            "\nMove it outside the working tree, for example to ../psst-private/,\n"
            "and confirm it was never committed. If it was, the history needs\n"
            "rewriting and anyone who cloned it has a copy."
        )
        return 1
    print(f"checked {len(tracked_files())} tracked files, none look private")
    return 0


if __name__ == "__main__":
    sys.exit(main())
