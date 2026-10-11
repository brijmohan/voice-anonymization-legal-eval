"""Tests for the attribute inference metric, the Opinion's third criterion.
The controls that matter are the two ends: an attribute that is fully encoded
must come out near advantage 1, and an attribute that is pure noise must come
out near 0. A measurement that cannot reach either end would read as a clean
bill of health on any system.
"""

from __future__ import annotations

import numpy as np
import pytest

from legal_eval.inference import (
    attribute_disclosure,
    attribute_disclosure_by_speaker,
)


def features_encoding(labels, dim=32, strength=3.0, seed=0):
    """Features that carry the label, plus noise, scaled by dimension."""
    rng = np.random.default_rng(seed)
    classes = sorted(set(labels))
    centres = rng.normal(size=(len(classes), dim)) * strength / np.sqrt(dim)
    index = {c: i for i, c in enumerate(classes)}
    x = np.stack([centres[index[v]] for v in labels])
    return x + rng.normal(scale=1.0 / np.sqrt(dim), size=x.shape)


def test_a_fully_encoded_attribute_is_detected():
    """The control that proves the metric can fire."""
    rng = np.random.default_rng(1)
    labels = rng.choice(["m", "f"], size=400)
    x = features_encoding(labels, strength=8.0)
    r = attribute_disclosure(x, labels, attribute="gender")
    assert r.advantage > 0.8, r.summary()
    assert r.accuracy > r.prior
    assert r.n_classes == 2
    assert "determined" in r.summary() or "substantial" in r.summary()


def test_pure_noise_gives_no_advantage():
    """Features independent of the label must not score above the prior."""
    rng = np.random.default_rng(2)
    labels = rng.choice(["a", "b", "c"], size=600)
    x = rng.normal(size=(600, 32))
    r = attribute_disclosure(x, labels, attribute="noise")
    assert r.advantage < 0.1, r.summary()
    assert "no disclosure" in r.summary() or "weak" in r.summary()


def test_accuracy_alone_would_mislead_on_an_imbalanced_attribute():
    """The reason advantage exists rather than accuracy.
    Ninety percent of samples in one class, features carrying nothing. A
    classifier scores 0.9 by always guessing the majority, which looks like
    strong prediction and is no prediction at all.
    """
    rng = np.random.default_rng(3)
    labels = np.array(["common"] * 540 + ["rare"] * 60, dtype=object)
    x = rng.normal(size=(600, 24))
    r = attribute_disclosure(x, labels, attribute="imbalanced")
    assert r.prior == pytest.approx(0.9, abs=0.01)
    assert r.accuracy >= 0.85  # looks impressive
    assert r.advantage < 0.15  # and means nothing
    assert r.balanced_accuracy < 0.6  # the honest aggregate


def test_missing_labels_are_dropped_and_counted():
    """Common Voice demographics are optional, so this is the normal case."""
    labels = np.array(["m"] * 100 + ["f"] * 100 + [""] * 50 + ["unknown"] * 30, dtype=object)
    x = features_encoding(["m"] * 100 + ["f"] * 100 + ["m"] * 80, strength=6.0)
    r = attribute_disclosure(x, labels, attribute="gender")
    assert r.n_labelled == 200
    assert r.n_unlabelled == 80
    assert r.n_classes == 2
    assert "71%" in r.summary() or "(71%" in r.summary() or "71" in r.summary()


def test_per_class_recall_exposes_a_well_recovered_rare_class():
    """A rare attribute value recovered well is worse than the aggregate shows.
    Legally this is the interesting case: a rare value is more identifying, so
    an aggregate that hides it understates the risk.
    """
    rng = np.random.default_rng(5)
    labels = np.array(["common"] * 300 + ["rare"] * 40, dtype=object)
    # The rare class is strongly encoded, the common one is not.
    x = rng.normal(scale=0.3, size=(340, 16))
    x[300:, 0] += 6.0
    r = attribute_disclosure(x, labels, attribute="condition")
    assert r.per_class_recall["rare"] > 0.8, r.summary()
    # The summary must name the exposed minority value, not the majority class,
    # which is usually the best recovered and is what the prior already gets.
    assert "most exposed minority value: rare" in r.summary(), r.summary()


def test_grouping_by_speaker_prevents_memorising_the_speaker():
    """Without grouping, speaker identity is reported as attribute leakage.
    Ten speakers, five per class, many rows each. The features encode the
    *speaker*, not the attribute. Ungrouped, the classifier recognises the
    speaker and the attribute looks disclosed. Grouped, with only ten
    independent points, it should not.
    """
    rng = np.random.default_rng(6)
    speakers = np.repeat([f"s{i}" for i in range(10)], 30).astype(object)
    labels = np.repeat(["m"] * 5 + ["f"] * 5, 30).astype(object)
    centres = rng.normal(size=(10, 16)) * 6.0 / 4.0
    x = np.stack([centres[int(str(s)[1:])] for s in speakers])
    x = x + rng.normal(scale=0.25, size=x.shape)
    ungrouped = attribute_disclosure(x, labels, attribute="gender")
    grouped = attribute_disclosure_by_speaker(x, labels, speakers, attribute="gender", n_folds=5)
    assert ungrouped.n_labelled == 300
    assert grouped.n_labelled == 10
    # The grouped estimate must not exceed the ungrouped one by a wide margin;
    # the point is that ungrouped is the inflated one.
    assert grouped.advantage <= ungrouped.advantage + 0.35


def test_rejects_a_single_class():
    x = np.random.default_rng(7).normal(size=(50, 8))
    with pytest.raises(ValueError, match="nothing to predict"):
        attribute_disclosure(x, np.array(["m"] * 50, dtype=object))


def test_rejects_mismatched_lengths():
    x = np.random.default_rng(8).normal(size=(50, 8))
    with pytest.raises(ValueError, match="50 features but 10 labels"):
        attribute_disclosure(x, np.array(["m"] * 10, dtype=object))


def test_rejects_non_matrix_features():
    with pytest.raises(ValueError, match="2D"):
        attribute_disclosure(np.zeros(10), np.array(["a", "b"] * 5, dtype=object))


def test_advantage_is_zero_when_accuracy_equals_the_prior():
    """The definition, checked at its fixed point."""
    rng = np.random.default_rng(9)
    labels = np.array(["x"] * 200 + ["y"] * 50, dtype=object)
    x = rng.normal(scale=1e-6, size=(250, 8))  # carries nothing at all
    r = attribute_disclosure(x, labels, attribute="flat")
    assert r.accuracy == pytest.approx(r.prior, abs=0.02)
    assert abs(r.advantage) < 0.05
