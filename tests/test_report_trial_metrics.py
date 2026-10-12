"""Tests for the trial-metric report.

The one thing that can go badly wrong here is the mated/non-mated split. If a
mated score leaks into the non-mated set, every metric in the report shifts
toward "more private" and nothing in the output says so.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "report_trial_metrics",
    Path(__file__).resolve().parents[1] / "scripts" / "report_trial_metrics.py",
)
rtm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rtm)


def test_the_split_takes_exactly_one_mated_score_per_trial():
    scores = np.arange(12, dtype=float).reshape(3, 4)
    true_rows = np.array([0, 1, 2, 0])
    mated, nonmated = rtm.mated_and_nonmated(scores, true_rows)
    assert mated.size == 4
    np.testing.assert_array_equal(mated, [0.0, 5.0, 10.0, 3.0])


def test_the_split_is_exhaustive_and_disjoint():
    """Every score lands in exactly one set, or a metric is computed on a lie."""
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, (7, 20))
    true_rows = rng.integers(0, 7, 20)
    mated, nonmated = rtm.mated_and_nonmated(scores, true_rows)
    assert mated.size + nonmated.size == scores.size
    assert mated.size == scores.shape[1]


def test_no_mated_score_appears_among_the_nonmated():
    """Constructed so the mated scores are uniquely identifiable by value."""
    scores = np.zeros((4, 4))
    true_rows = np.arange(4)
    scores[true_rows, np.arange(4)] = [10.0, 11.0, 12.0, 13.0]
    mated, nonmated = rtm.mated_and_nonmated(scores, true_rows)
    np.testing.assert_array_equal(np.sort(mated), [10.0, 11.0, 12.0, 13.0])
    assert not np.isin(nonmated, [10.0, 11.0, 12.0, 13.0]).any()


def test_repeated_true_rows_are_handled():
    """Several trials may belong to one enrollment speaker."""
    scores = np.arange(6, dtype=float).reshape(2, 3)
    true_rows = np.array([1, 1, 1])
    mated, nonmated = rtm.mated_and_nonmated(scores, true_rows)
    np.testing.assert_array_equal(mated, [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(np.sort(nonmated), [0.0, 1.0, 2.0])
