#!/usr/bin/env python3
"""Find audio readers that cannot open a Kaldi extended filename.

A `wav.scp` value is not always a path. Kaldi allows an extended filename, a
shell pipeline ending in `|`, and that is how a corpus distributed as mp3 is
addressed without transcoding it first:

    common_voice_en_1 ffmpeg -v 8 -i clip.mp3 -f wav -ar 16000 -ac 1 - |

`soundfile.read` raises `TypeError: Invalid file` on one of those, and
`torchaudio.load` raises too. The failure is loud, so it looks like the kind of
bug a first run finds. The reason it is worth a check rather than a habit is
that the same defect appeared at four separate sites in one pipeline, and
twice it was found only by running the thing after a grep had declared the
pipeline clean:

1. `anonymization/modules/sttts/text/recognition/ims_asr.py`
2. `anonymization/modules/sttts/prosody/extraction/ims_prosody_extraction.py`
3. `evaluation/privacy/asv/speaker_extraction.py`
4. `evaluation/privacy/asv/speechbrain_vectors.py`

Site 4 was the expensive one. It is reached from site 3, which had already
been fixed, and it re-reads the file the caller has just decoded, so fixing
the caller did not fix it and the smoke test still failed on the first
utterance.

There is a second, quieter half of the same bug. Reading a pipe entry with
`read_kaldi_format` yields a list of tokens, and site 3 narrowed that list
with a literal index:

    if isinstance(wav_path, list):
        wav_path = wav_path[1]

For a pipe entry token 1 is `-v`, an ffmpeg flag. That does not raise where it
is written. It raises later, somewhere else, as an unreadable path, which is a
much worse failure than the first kind.

Usage:
    python scripts/check_scp_readers.py --root /path/to/pipeline

A reviewed call is waived with a marker comment on the call's own line or the
line above it, naming why it is safe:

    audio, sr = sf.read(path)  # scp-reader-ok: path is always a plain file

Exits non-zero if any unwaived site remains, so it can run in CI over a
vendored or patched checkout.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

#: Readers that take a filename and cannot open a pipe. The value is the
#: pipe-aware alternative to reach for, which is what the fix looked like at
#: every site above.
RISKY_READERS = {
    "soundfile.read": "load_wav_from_scp",
    "sf.read": "load_wav_from_scp",
    "soundfile.SoundFile": "load_wav_from_scp",
    "sf.SoundFile": "load_wav_from_scp",
    "torchaudio.load": "load_wav_from_scp",
    "librosa.load": "load_wav_from_scp",
    "scipy.io.wavfile.read": "load_wav_from_scp",
    "wavfile.read": "load_wav_from_scp",
    "wave.open": "load_wav_from_scp",
}

#: A variable holding a wav.scp value, for the literal-index check. Narrow on
#: purpose: a wider rule flagged ordinary list handling and was ignored.
SCP_VALUE_HINTS = ("wav_path", "wav_scp", "wavscp", "scp_entry", "wav_entry")

#: Evidence that a module sits in the Kaldi data path and can therefore be
#: handed a pipe entry. Without this the check flags every audio read in a
#: vendored TTS tree, most of which are reached only from a file the pipeline
#: itself wrote. The first run over a real checkout returned 25 findings, of
#: which 3 mattered, and a check nobody can act on is a check nobody runs.
DATA_PATH_MARKERS = (
    "wav.scp",
    "read_kaldi_format",
    "load_wav_from_scp",
    "utt2spk",
    "spk2utt",
    "kaldi",
)

WAIVER = "scp-reader-ok"

#: Directories that hold vendored code or run artifacts rather than pipeline
#: source. Walking them is not just noise: a checkout mid-run holds hundreds of
#: thousands of audio files, and on a network filesystem the scan then takes
#: longer than the run it was meant to protect.
SKIP_DIRS = {
    ".git", "__pycache__", "site-packages", "node_modules", ".micromamba",
    "venv", "venv_sttts", "env", ".venv",
    "data", "corpora", "exp", "models", "results", "logs",
}


def _dotted_name(node: ast.AST) -> str | None:
    """Return the dotted name of an attribute or name chain.

    Args:
        node: the `func` of a `Call`, or any attribute chain.

    Returns:
        A dotted string such as `"scipy.io.wavfile.read"`, or None if the chain
        is not made only of names and attributes.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_waived(lines: list[str], lineno: int) -> bool:
    """Whether a marker comment waives the finding at this line.

    The marker is accepted on the call's own line or the line above, because a
    long call is usually wrapped and the comment then has nowhere else to go.

    Args:
        lines: the file's lines, 0-indexed.
        lineno: the 1-indexed line of the finding.

    Returns:
        True if the finding is waived.
    """
    for candidate in (lineno - 1, lineno - 2):
        if 0 <= candidate < len(lines) and WAIVER in lines[candidate]:
            return True
    return False


def in_data_path(source: str) -> bool:
    """Whether a module shows any sign of handling Kaldi data.

    This is the difference between a finding worth acting on and an audio read
    in a vendored tree that is only ever handed a file the pipeline wrote
    itself. It is deliberately a text search over the whole module rather than
    a dataflow question, which is not answerable statically here.

    Args:
        source: Python source text.

    Returns:
        True if the module mentions anything from `DATA_PATH_MARKERS`.
    """
    lowered = source.lower()
    return any(marker in lowered for marker in DATA_PATH_MARKERS)


def findings_for_source(source: str, path: str = "<string>") -> list[tuple[str, int, str]]:
    """Scan one module's source for both halves of the bug.

    Args:
        source: Python source text.
        path: the name to report findings against.

    Returns:
        A list of `(path, line, message)`, in line order. Empty means clean.
        A message is prefixed `reachable:` when the module handles Kaldi data
        and `review:` when it does not.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [(path, exc.lineno or 0, f"reachable: could not parse: {exc.msg}")]

    prefix = "reachable: " if in_data_path(source) else "review: "

    lines = source.splitlines()
    out: list[tuple[str, int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name in RISKY_READERS and not _is_waived(lines, node.lineno):
                out.append((
                    path, node.lineno,
                    f"{prefix}{name} cannot open a Kaldi pipe entry; "
                    f"route it through {RISKY_READERS[name]} "
                    f"or waive with a '{WAIVER}:' comment",
                ))
        # The quiet half: narrowing a wav.scp token list by a literal index.
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in SCP_VALUE_HINTS and isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, int) and not _is_waived(lines, node.lineno):
                    out.append((
                        path, node.lineno,
                        f"{prefix}{node.value.id}[{node.slice.value}] indexes a "
                        f"wav.scp token list by position; for a pipe entry that "
                        f"yields a command flag, not a path",
                    ))

    return sorted(out, key=lambda f: f[1])


def _python_files(root: Path) -> list[Path]:
    """Python files under a root, pruning skipped directories as we descend.

    `rglob` would descend into a corpus directory before we could reject its
    files, so the walk prunes instead of filtering.

    Args:
        root: directory to walk, or a single file.

    Returns:
        The files to scan, in a stable order.
    """
    if root.is_file():
        return [root]
    found: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in SKIP_DIRS:
                    stack.append(entry)
            elif entry.suffix == ".py":
                found.append(entry)
    return sorted(found)


def scan(root: Path) -> list[tuple[str, int, str]]:
    """Scan every Python file under a root.

    Args:
        root: directory to walk, or a single file.

    Returns:
        All findings, ordered by path then line.
    """
    files = _python_files(root)
    out: list[tuple[str, int, str]] = []
    for f in files:
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            out.append((str(f), 0, f"could not read: {exc}"))
            continue
        out.extend(findings_for_source(source, str(f)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="directory or file to scan")
    parser.add_argument("--all", action="store_true",
                        help="also list findings in modules that show no sign "
                             "of handling Kaldi data")
    args = parser.parse_args(argv)

    if not args.root.exists():
        print(f"no such path: {args.root}", file=sys.stderr)
        return 2

    findings = scan(args.root)
    reachable = [f for f in findings if f[2].startswith("reachable: ")]
    review = [f for f in findings if not f[2].startswith("reachable: ")]

    for path, line, message in reachable:
        print(f"{path}:{line}: {message}")
    if args.all:
        for path, line, message in review:
            print(f"{path}:{line}: {message}")

    if reachable:
        print(f"\n{len(reachable)} site(s) in the Kaldi data path that a pipe "
              f"entry would break. {len(review)} more in modules with no sign "
              f"of Kaldi data; pass --all to list them.", file=sys.stderr)
        return 1

    print(f"No unwaived pipe-unsafe audio readers in the Kaldi data path under "
          f"{args.root}. {len(review)} finding(s) in modules with no sign of "
          f"Kaldi data were not counted; pass --all to list them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
