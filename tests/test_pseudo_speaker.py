"""Tests for the pseudo-voice identifiability audit (CNIL G3).

The audit's job is to answer "would a verifier accept this pseudo-voice as a
real person". These tests pin that it says yes when it should and no when it
should, because a measurement that always returns zero would look like a clean
bill of health.
"""

from __future__ import annotations

import numpy as np
import pytest

from legal_eval.pseudo_speaker import (
    audit_pseudo_speakers,
    verification_threshold,
)


def real_speakers(n_speakers=40, per_speaker=6, dim=256, spread=0.35, seed=0):
    """Real speakers as tight clusters, which is what embeddings look like.

    ``spread`` is the within-speaker noise as a fraction of the centre's norm,
    and the per-dimension scale is divided by ``sqrt(dim)`` to achieve that.
    Drawing ``N(0, spread)`` per dimension instead would give a noise vector of
    norm ``spread * sqrt(dim)``, which at dim=256 is 5.6 times the unit-norm
    centre: the clusters vanish, same-speaker and different-speaker similarity
    coincide, and the fixture's verifier lands at an EER near 0.38. Every
    threshold then admits everything and the audit appears to fire on
    unrelated vectors.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_speakers, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    scale = spread / np.sqrt(dim)
    rows, ids = [], []
    for index, centre in enumerate(centres):
        noise = rng.normal(scale=scale, size=(per_speaker, dim))
        rows.append(centre + noise)
        ids.extend([f"spk{index:03d}"] * per_speaker)
    return np.concatenate(rows), np.array(ids), centres


def test_threshold_sits_between_the_two_distributions():
    """A measured threshold must separate genuine pairs from impostor pairs."""
    embeddings, ids, _ = real_speakers()
    threshold, eer, refs = verification_threshold(embeddings, ids)
    assert refs["nonmated_mean"] < threshold < refs["mated_mean"]
    assert 0.0 <= eer <= 0.5
    # The fixture must give a competent verifier, or every threshold admits
    # everything and the rest of this file proves nothing.
    assert eer < 0.01, f"fixture verifier is too weak, EER {eer:.4f}"


def test_threshold_needs_a_mated_pair():
    """One item per speaker gives no genuine pair, so no threshold exists."""
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(10, 32))
    ids = np.array([f"s{i}" for i in range(10)])
    with pytest.raises(ValueError, match="no same-speaker pair"):
        verification_threshold(embeddings, ids)


def test_a_pseudo_voice_that_is_a_real_speaker_is_identified():
    """The control that proves the audit can fire at all.

    Hand it the real speaker centroids as the pseudo-voices. Every one is that
    speaker, so every one must be identified. A measurement that returned zero
    here would be useless and would read as a pass.
    """
    embeddings, ids, centres = real_speakers()
    audit = audit_pseudo_speakers(centres, embeddings, ids)
    assert audit.identification_rate == pytest.approx(1.0)
    assert audit.n_identified == len(centres)
    assert np.all(audit.margin > 0)
    # And it names the right speaker for each.
    assert sorted(audit.nearest_index) == sorted(np.unique(ids))


def test_unrelated_pseudo_voices_are_not_identified():
    """Independent random directions in a 256-dim space are near orthogonal."""
    embeddings, ids, _ = real_speakers()
    rng = np.random.default_rng(99)
    pseudo = rng.normal(size=(60, 256))
    audit = audit_pseudo_speakers(pseudo, embeddings, ids)
    assert audit.identification_rate == pytest.approx(0.0)
    assert audit.n_identified == 0
    assert np.all(audit.margin < 0)


def test_a_planted_near_duplicate_is_caught():
    """One pseudo-voice close to one real speaker, the rest unrelated.

    This is the realistic failure: not a whole generator that memorised, but a
    single sampled target that happens to land on somebody.
    """
    embeddings, ids, centres = real_speakers()
    rng = np.random.default_rng(7)
    pseudo = rng.normal(size=(50, 256))
    pseudo[13] = centres[5] + rng.normal(scale=0.02 / np.sqrt(256), size=256)
    audit = audit_pseudo_speakers(pseudo, embeddings, ids)
    assert audit.n_identified == 1
    assert audit.identification_rate == pytest.approx(1 / 50)
    assert audit.nearest_index[13] == np.unique(ids)[5]
    assert audit.margin[13] > 0


def test_collision_rate_detects_mutually_acceptable_pseudo_voices():
    """Two source speakers given near-identical targets become linkable."""
    embeddings, ids, _ = real_speakers()
    rng = np.random.default_rng(3)
    pseudo = rng.normal(size=(30, 256))
    # Make three of them near duplicates of one another.
    pseudo[1] = pseudo[0] + rng.normal(scale=0.01 / np.sqrt(256), size=256)
    pseudo[2] = pseudo[0] + rng.normal(scale=0.01 / np.sqrt(256), size=256)
    audit = audit_pseudo_speakers(pseudo, embeddings, ids)
    # Three mutually close vectors give three colliding pairs out of 435.
    assert audit.collision_rate > 0
    assert audit.collision_rate == pytest.approx(3 / (30 * 29 / 2), abs=1e-9)


def test_no_collisions_among_independent_pseudo_voices():
    embeddings, ids, _ = real_speakers()
    rng = np.random.default_rng(11)
    audit = audit_pseudo_speakers(rng.normal(size=(40, 256)), embeddings, ids)
    assert audit.collision_rate == pytest.approx(0.0)


def test_single_pseudo_voice_has_no_collision_rate():
    embeddings, ids, centres = real_speakers()
    audit = audit_pseudo_speakers(centres[:1], embeddings, ids)
    assert audit.collision_rate == 0.0


def test_margin_is_scaled_by_the_impostor_spread():
    """A margin in raw cosine units is not comparable between embedding spaces."""
    embeddings, ids, centres = real_speakers()
    audit = audit_pseudo_speakers(centres, embeddings, ids)
    raw = audit.nearest_similarity - audit.threshold
    assert np.allclose(audit.margin, raw / audit.nonmated_std)


def test_an_explicit_threshold_overrides_the_measured_one():
    embeddings, ids, centres = real_speakers()
    strict = audit_pseudo_speakers(centres, embeddings, ids, threshold=0.999)
    assert strict.threshold == pytest.approx(0.999)
    assert strict.identification_rate < 1.0


def test_rejects_non_matrix_input():
    embeddings, ids, _ = real_speakers()
    with pytest.raises(ValueError, match="2D array"):
        audit_pseudo_speakers(np.zeros(256), embeddings, ids)


def test_summary_states_the_verdict():
    embeddings, ids, centres = real_speakers()
    identified = audit_pseudo_speakers(centres, embeddings, ids).summary()
    assert "clear the threshold" in identified
    rng = np.random.default_rng(5)
    clean = audit_pseudo_speakers(
        rng.normal(size=(20, 256)), embeddings, ids
    ).summary()
    assert "NO pseudo-voice clears" in clean


def test_chance_rate_grows_with_population():
    """The baseline the identification rate must be read against."""
    from legal_eval.pseudo_speaker import chance_identification_rate

    assert chance_identification_rate(0.0, 1000) == pytest.approx(0.0)
    assert chance_identification_rate(0.1, 1) == pytest.approx(0.1)
    # The number that makes the point: a verifier at EER 0.087 searched against
    # 29 speakers accepts an unrelated voice about 92% of the time.
    assert chance_identification_rate(0.0867, 29) == pytest.approx(0.92792, abs=1e-5)
    rates = [chance_identification_rate(0.05, n) for n in (1, 10, 100)]
    assert rates[0] < rates[1] < rates[2]


def test_chance_rate_rejects_impossible_input():
    from legal_eval.pseudo_speaker import chance_identification_rate

    with pytest.raises(ValueError, match=r"far must be in"):
        chance_identification_rate(1.5, 10)
    with pytest.raises(ValueError, match="at least 1"):
        chance_identification_rate(0.1, 0)


def test_population_for_chance_inverts_the_baseline():
    from legal_eval.pseudo_speaker import (
        chance_identification_rate,
        population_for_chance,
    )

    # A verifier good enough that one comparison is under the target.
    for far in (0.001, 0.01):
        n = population_for_chance(far, target=0.05)
        assert n >= 1
        assert chance_identification_rate(far, n) <= 0.05 + 1e-9
        assert chance_identification_rate(far, n + 1) > 0.05

    # The important case. At EER 0.087 a single comparison already exceeds a 5%
    # target, so no enrolment size helps and the answer is zero. This is why a
    # 29 speaker G3 experiment cannot answer the CNIL's question: the verifier,
    # not the population, is the limit.
    assert population_for_chance(0.0867, target=0.05) == 0
    assert population_for_chance(0.05, target=0.05) >= 1


def test_audit_reports_its_own_chance_level():
    embeddings, ids, _ = real_speakers()
    rng = np.random.default_rng(42)
    audit = audit_pseudo_speakers(rng.normal(size=(30, 256)), embeddings, ids)
    assert audit.n_real_speakers == len(np.unique(ids))
    assert 0.0 <= audit.chance_identification_rate <= 1.0
    assert "chance rate at" in audit.summary()
