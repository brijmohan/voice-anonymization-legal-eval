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
    # Reference markers from regulatory correspondence. A file reference
    # identifies a specific case and belongs with the letter, not here.
    (re.compile(r"N/R[eé]f"), "reference marker from correspondence"),
    (re.compile(r"(?i)\bsaisine\s+n"), "case number from correspondence"),
]

#: Phrases by which a document admits that the source it draws on is not
#: public. Matching one is not a problem. Matching one and then quoting that
#: source at length is.
NON_PUBLIC_SOURCE = [
    re.compile(r"(?i)\bis not public\b"),
    re.compile(r"(?i)held outside this repository"),
    re.compile(r"(?i)\bnot for publication\b"),
]

#: How many quoted lines a document may carry while declaring its source
#: non-public. One or two is a necessary extract. The file that prompted this
#: check carried twenty one, which is reproduction.
MAX_QUOTED_LINES = 2

#: Paths that legitimately mention the patterns above, such as this checker and
#: the ignore rules that implement the same policy.
ALLOWED = {
    "scripts/check_no_private_material.py",
    ".gitignore",
    "CLAUDE.md",
}


def quoted_lines(text: str) -> int:
    """Count Markdown blockquote lines.

    Args:
        text: A document's contents.

    Returns:
        The number of lines beginning a blockquote. Nested markers count once,
        because what matters is how much of the source is reproduced.
    """
    return sum(1 for line in text.splitlines() if line.lstrip().startswith(">"))


def quotes_a_non_public_source(text: str) -> int | None:
    """Whether a document reproduces a source it says is not public.

    This is the check that was missing. A document described a letter as not
    public and then reproduced twenty one extracts from it, along with the
    reference identifying the case and the name of the official who signed it.
    Every individual sentence looked like careful scholarship. The file as a
    whole was the letter.

    The rule is the document's own admission: if it says the source is not
    public, it may not quote the source at length. Stating the substance in
    your own words is always allowed, which is what the public version does.

    Args:
        text: A document's contents.

    Returns:
        The number of quoted lines when the document both declares a
        non-public source and exceeds `MAX_QUOTED_LINES`, otherwise None.
    """
    if not any(pattern.search(text) for pattern in NON_PUBLIC_SOURCE):
        return None
    count = quoted_lines(text)
    return count if count > MAX_QUOTED_LINES else None


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
            else:
                count = quotes_a_non_public_source(text)
                if count is not None:
                    problems.append(
                        f"  {path}\n      says its source is not public and "
                        f"then quotes it on {count} lines. State the substance "
                        f"in your own words instead."
                    )

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
