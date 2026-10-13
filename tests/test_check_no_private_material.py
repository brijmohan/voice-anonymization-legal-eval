"""Tests for the private-material check.

This check exists because publishing is irreversible, and it was extended
because it passed on a file it should have stopped. That file described a
regulator's letter as not public and then reproduced twenty one extracts from
it, along with the reference identifying the case and the name of the official
who signed it. Every sentence looked like careful scholarship, and the file as
a whole was the letter.

So the tests that matter here are the ones where the document is plausible and
the check has to object anyway, and the ones where a legitimate document must
not be objected to. The second set is as important as the first: a check that
fires on honest work gets switched off.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_no_private_material",
    Path(__file__).resolve().parents[1] / "scripts"
    / "check_no_private_material.py",
)
cnpm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cnpm)

DECLARES = "The letter is held outside this repository. It is not public.\n"


# Counting what is reproduced.

def test_quoted_lines_counts_blockquotes():
    assert cnpm.quoted_lines("a\n> one\n> two\nb\n") == 2


def test_quoted_lines_counts_an_indented_quote():
    """A quote inside a list is still a quote."""
    assert cnpm.quoted_lines("- item\n  > quoted\n") == 1


def test_quoted_lines_counts_a_nested_marker_once():
    """What matters is how much of the source is reproduced, not the nesting."""
    assert cnpm.quoted_lines("> > deep\n") == 1


def test_quoted_lines_ignores_prose_mentioning_a_greater_than():
    assert cnpm.quoted_lines("where x > y holds\n") == 0


# The rule itself.

def test_a_document_quoting_a_source_it_calls_non_public_is_caught():
    text = DECLARES + "".join(f"> extract {i}\n" for i in range(21))
    assert cnpm.quotes_a_non_public_source(text) == 21


def test_the_threshold_is_the_boundary_and_is_pinned():
    """At the limit it passes, one over it fails. A boundary that moves with a
    rewrite is a boundary that is not pinned."""
    at_limit = DECLARES + "> a\n" * cnpm.MAX_QUOTED_LINES
    over = DECLARES + "> a\n" * (cnpm.MAX_QUOTED_LINES + 1)
    assert cnpm.quotes_a_non_public_source(at_limit) is None
    assert cnpm.quotes_a_non_public_source(over) == cnpm.MAX_QUOTED_LINES + 1


def test_stating_the_substance_in_your_own_words_is_allowed():
    """Which is exactly what the public version of the document does, so this
    is the case that must not be objected to."""
    text = DECLARES + ("The review identified no obstacle to the solution "
                       "demonstrating compliance. Stated in our words, with "
                       "no sentence of the source reproduced.\n")
    assert cnpm.quotes_a_non_public_source(text) is None


def test_quoting_a_public_source_at_length_is_allowed():
    """Published papers are quoted throughout this repository and must stay so."""
    text = "".join(f"> published finding {i}\n" for i in range(30))
    assert cnpm.quotes_a_non_public_source(text) is None


@pytest.mark.parametrize("phrase", [
    "it is not public",
    "held outside this repository",
    "not for publication",
])
def test_each_admission_phrase_triggers_the_rule(phrase):
    text = f"{phrase}\n" + "> a\n" * 10
    assert cnpm.quotes_a_non_public_source(text) == 10


# Reference markers, which identify a specific case file.

#: Marker fragments, assembled at runtime so that this file never itself
#: contains a string the checker would match. Writing a realistic marker into
#: a fixture is how a real case number reached a public test file once, and
#: putting this file on the checker's allow list would have been the wrong fix:
#: it would create an unchecked hole to hold the very patterns being tested.
MARKER_PARTS = [
    ("N/R", "ef MAI/SA000000"),
    ("N/R", "\u00e9f. MAI/SA000000"),
    ("Saisine", " n 00000000"),
    ("saisine", " No 123"),
]


@pytest.mark.parametrize("head,tail", MARKER_PARTS)
def test_correspondence_reference_markers_are_forbidden_content(head, tail):
    marker = head + tail
    assert any(pattern.search(marker)
               for pattern, _ in cnpm.FORBIDDEN_CONTENT), marker


def test_this_test_file_contains_no_marker_it_would_flag():
    """Otherwise the checker has to be told to skip its own tests, and an
    allow list entry is a hole exactly where the sensitive patterns live."""
    source = Path(__file__).read_text(encoding="utf-8")
    for pattern, why in cnpm.FORBIDDEN_CONTENT:
        assert not pattern.search(source), f"{why}: {pattern.pattern}"


def test_an_ordinary_document_reference_is_not_a_correspondence_marker():
    """Opinion 05/2014 and a DOI are cited constantly and must not trip."""
    for benign in ("Opinion 05/2014", "doi:10.21437/Interspeech.2025-1699",
                   "ref. MAI/SA" + "241024"):
        assert not any(pattern.search(benign)
                       for pattern, _ in cnpm.FORBIDDEN_CONTENT), benign


# The repository itself, which is the check that runs in CI.

def test_this_repository_is_clean():
    assert cnpm.main() == 0
