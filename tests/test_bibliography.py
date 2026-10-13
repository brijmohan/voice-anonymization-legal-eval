"""Tests for the reference library.

A bibliography has one characteristic failure: it drifts from the index that
describes it. An entry is added to the `.bib` and never categorised, or a
category is written up in prose and its entries quietly disappear. Either way
the library still looks complete, which is why these are checks rather than a
convention.

The second thing guarded here is the dash exemption. `references.bib` quotes
published titles verbatim, and at least one of them contains an em dash, so the
house style has to yield to citation fidelity. The exemption is per line and
these tests pin that it stays narrow, because a whole-file exemption would turn
a 600 line document that is mostly our own prose into unchecked territory.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "docs" / "references.bib"
INDEX = ROOT / "docs" / "bibliography.md"

SPEC = importlib.util.spec_from_file_location(
    "check_no_dashes", ROOT / "scripts" / "check_no_dashes.py")
cnd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cnd)

SECTION_LETTERS = set("ABCDEFGHZ")

ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.M)


def entries():
    """Every entry as (type, key, body)."""
    text = BIB.read_text(encoding="utf-8")
    out = []
    for block in re.split(r"\n(?=@)", text):
        m = ENTRY.match(block.strip())
        if m:
            out.append((m.group(1).lower(), m.group(2), block))
    return out


def field(body, name):
    m = re.search(name + r"\s*=\s*\{(.*?)\}\s*,?\s*\n", body, re.I | re.S)
    return " ".join(m.group(1).split()) if m else None


# The library exists and is well formed.

def test_the_bibliography_exists_and_is_not_trivial():
    assert BIB.exists() and INDEX.exists()
    assert len(entries()) > 200


def test_every_entry_has_a_key_a_title_and_a_category():
    """A categorisation that lives only in the prose index is one that drifts."""
    missing = []
    for _, key, body in entries():
        if not field(body, "title"):
            missing.append(f"{key}: no title")
        if not field(body, "keywords"):
            missing.append(f"{key}: no keywords, so it has no place")
    assert missing == [], missing[:10]


def test_every_category_is_well_formed():
    bad = []
    for _, key, body in entries():
        kw = field(body, "keywords") or ""
        if not re.fullmatch(r"[A-Z]/[a-z-]+", kw):
            bad.append(f"{key}: {kw!r}")
        elif kw[0] not in SECTION_LETTERS:
            bad.append(f"{key}: unknown section {kw[0]}")
    assert bad == [], bad[:10]


def test_keys_are_unique():
    """Two entries sharing a key means one of them cannot be cited."""
    keys = [key for _, key, _ in entries()]
    duplicates = {k for k in keys if keys.count(k) > 1}
    assert duplicates == set(), duplicates


def test_titles_are_unique_so_the_dedupe_held():
    """Three bibliographies were merged. A repeated title means the merge
    stopped working, and the IS26 bibliography did contain one duplicate."""
    def norm(t):
        return re.sub(r"[^a-z0-9]", "", (t or "").lower())
    titles = [norm(field(body, "title")) for _, _, body in entries()]
    titles = [t for t in titles if t]
    duplicates = {t for t in titles if titles.count(t) > 1}
    assert duplicates == set(), list(duplicates)[:5]


# The index and the bibliography have to agree, in both directions.

#: A citation key ends its line in the index, optionally followed by the
#: working-set marker. Matching a backticked word anywhere instead picked up
#: prose mentions of field names such as the keywords field.
KEY_IN_INDEX = re.compile(
    r"`([A-Za-z0-9]+)`(?:\s*\*\*\[working set\]\*\*)?\s*$", re.M)


def index_keys():
    return set(KEY_IN_INDEX.findall(INDEX.read_text(encoding="utf-8")))


def test_every_key_in_the_index_exists_in_the_bibliography():
    bib_keys = {key for _, key, _ in entries()}
    orphans = index_keys() - bib_keys
    assert orphans == set(), sorted(orphans)[:10]


def test_every_entry_appears_in_the_index():
    """Otherwise an entry is in the library and in no category a reader sees."""
    bib_keys = {key for _, key, _ in entries()}
    unlisted = bib_keys - index_keys()
    assert unlisted == set(), sorted(unlisted)[:10]


def test_the_working_set_is_marked_in_both_places():
    """The working set is the part with notes. If the marker and the notes
    disagree, the index is claiming guidance it does not give."""
    text = INDEX.read_text(encoding="utf-8")
    marked = set(re.findall(r"`([A-Za-z0-9]+)` \*\*\[working set\]\*\*", text))
    assert len(marked) >= 20, f"only {len(marked)} marked"
    # Every marked entry must carry a note, which in the index is a blockquote.
    working_section = text.split("## The working set")[1].split("## The full library")[0]
    noted = set(KEY_IN_INDEX.findall(working_section))
    assert marked == noted, (marked ^ noted)


def test_the_working_set_notes_say_something():
    """A note that repeats the title is not a note."""
    text = INDEX.read_text(encoding="utf-8")
    working = text.split("## The working set")[1].split("## The full library")[0]
    notes = re.findall(r"^> (.+)$", working, re.M)
    assert len(notes) >= 20
    for note in notes:
        assert len(note) > 40, note


# The dash exemption, which must stay narrow.

def test_a_dash_in_a_quoted_title_is_allowed():
    em = chr(0x2014)
    assert cnd.TITLE_LINE.match(f"  title = {{The vocoder{em}Electrical}},")
    assert cnd.TITLE_LINE.match(f"- Dudley (1940). The vocoder{em}x. `dudley40`")


def test_a_dash_in_our_own_prose_is_still_caught():
    """The exemption is for citations, not for a licence to use dashes in a
    file that happens to contain citations."""
    assert not cnd.TITLE_LINE.match("This library is organised by question, not topic.")
    assert not cnd.TITLE_LINE.match("- Section B is thin, which is worth stating.")
    assert not cnd.TITLE_LINE.match("**The working set** is what the project depends on.")


def test_only_the_bibliography_files_are_exempt():
    assert cnd.QUOTED_TITLE_FILES == {"docs/references.bib", "docs/bibliography.md"}


def test_the_repository_still_passes_the_dash_check():
    assert cnd.main() == 0


# What the library claims about itself.

@pytest.mark.parametrize("key", [
    "art29wp2014opinion05",
    "cohen2020singlingout",
    "hanisch2024falsesense",
    "nautsch2020zebra",
])
def test_the_foundations_are_present(key):
    """These four were missing from every bibliography we had, which is the
    finding that prompted building this file."""
    assert key in {k for _, k, _ in entries()}


def test_unverified_metadata_is_marked_rather_than_guessed():
    """Four entries could not have an author initial confirmed. They must say
    so, because a plausible-looking citation is worse than an incomplete one."""
    unconfirmed = [key for _, key, body in entries()
                   if "to confirm" in (field(body, "note") or "")]
    assert len(unconfirmed) >= 3, unconfirmed
