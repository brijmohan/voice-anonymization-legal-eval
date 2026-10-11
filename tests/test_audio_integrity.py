"""Tests for the anonymized audio integrity check.

The case this exists for is real and is pinned below: the 2026 McAdams
baseline on Common Voice 27 emitted a file of the right length whose samples
are 99.93% exact zeros with 67 near-full-scale impulses.

The thing to keep in mind when reading these tests is the direction of the
error. A destroyed utterance scores at chance, chance is the best attainable
privacy score, so the failure makes the system look better. Any check here
that is too loose therefore flatters the system under test, which is the
opposite of what a privacy evaluation should err towards.

The tests also pin that normal anonymization passes. A system is allowed to
sound bad, and several good ones do.
"""

from __future__ import annotations

import numpy as np
import pytest

from legal_eval.audio_integrity import (
    active_fraction,
    check_anonymized,
    clipped_fraction,
    optimistic_bias_note,
)

RNG = np.random.default_rng(0)


def speech(n: int = 96768, level: float = 0.04) -> np.ndarray:
    """Something with the gross statistics of speech, not actual speech."""
    return (RNG.standard_normal(n) * level).clip(-1, 1)


def the_observed_failure(n: int = 96768, n_impulses: int = 67) -> np.ndarray:
    """Exact shape of the McAdams output for common_voice_en_25475281.

    Correct length, 99.93% exact zeros, a handful of near-full-scale impulses.
    """
    out = np.zeros(n)
    positions = RNG.choice(n, size=n_impulses, replace=False)
    out[positions] = RNG.choice([-0.9999, 0.984], size=n_impulses)
    return out


# The case this module exists for.

def test_the_observed_mcadams_failure_is_caught():
    result = check_anonymized(the_observed_failure(), speech())
    assert not result.usable
    assert result.n_samples == 96768
    # 67 audible samples in 96768 is 0.07%, against 96% for the original.
    assert result.active < 0.001
    assert result.reference_active is not None and result.reference_active > 0.9
    assert any("silence rather than speech" in r for r in result.reasons)
    assert any("removed the speech rather than changing it" in r
               for r in result.reasons)


def test_it_is_caught_without_the_original_too():
    """The original is often not to hand when the check runs."""
    result = check_anonymized(the_observed_failure())
    assert not result.usable
    assert result.reference_active is None
    assert any("silence rather than speech" in r for r in result.reasons)


def test_pure_digital_silence_is_caught():
    result = check_anonymized(np.zeros(16000), speech())
    assert not result.usable
    assert result.active == 0.0
    assert result.rms == 0.0


def test_impulses_in_silence_are_flagged_as_clipping_too():
    """Many full-scale samples are the other half of the signature."""
    out = np.zeros(1000)
    out[:50] = 1.0
    result = check_anonymized(out)
    assert not result.usable
    assert result.clipped == pytest.approx(0.05)
    assert any("impulses rather than speech" in r for r in result.reasons)


# Normal anonymization has to pass, or the check gets switched off.

def test_ordinary_anonymized_speech_passes():
    result = check_anonymized(speech(level=0.03), speech(level=0.04))
    assert result.usable
    assert result.reasons == ()


def test_much_quieter_output_still_passes():
    """A system may change level substantially without destroying speech."""
    original = speech(level=0.08)
    result = check_anonymized(original * 0.05, original)
    assert result.usable


def test_a_vocoder_that_smooths_the_signal_passes():
    """Output can be heavily altered and still be speech."""
    original = speech()
    smoothed = np.convolve(original, np.ones(64) / 64, mode="same")
    result = check_anonymized(smoothed, original)
    assert result.usable


def test_a_genuinely_quiet_original_does_not_trigger_the_ratio():
    """A quiet input must not make its faithful output look destroyed."""
    original = np.zeros(16000)
    original[:200] = 0.01
    result = check_anonymized(original.copy(), original)
    # Low absolute activity, so min_active still fires; the point is that the
    # collapse ratio does not, because nothing collapsed.
    assert not any("removed the speech" in r for r in result.reasons)


def test_some_clipping_is_tolerated():
    out = speech(n=10000, level=0.3)
    out[:50] = 1.0
    result = check_anonymized(out)
    assert result.usable, result.reasons


# The measured quantities.

def test_active_fraction_counts_audible_samples():
    signal = np.array([0.0, 0.0, 0.5, -0.5])
    assert active_fraction(signal) == 0.5


def test_active_fraction_ignores_dither_below_the_floor():
    """One 16-bit quantisation step is about 3e-5, under the floor."""
    assert active_fraction(np.full(100, 3e-5)) == 0.0


def test_active_fraction_of_an_empty_signal_is_zero():
    assert active_fraction(np.array([])) == 0.0


def test_clipped_fraction_counts_full_scale_samples():
    assert clipped_fraction(np.array([0.1, 1.0, -1.0, 0.2])) == 0.5


def test_clipped_fraction_of_an_empty_signal_is_zero():
    assert clipped_fraction(np.array([])) == 0.0


def test_a_multichannel_signal_is_flattened():
    result = check_anonymized(speech(2048).reshape(2, 1024))
    assert result.n_samples == 2048


def test_an_empty_anonymized_signal_raises():
    """A zero-length output is a failure to report, not to measure."""
    with pytest.raises(ValueError, match="empty"):
        check_anonymized(np.array([]))


# The sentence a report has to carry.

def test_the_bias_note_states_the_direction():
    note = optimistic_bias_note(1, 195580)
    assert "1 of 195580" in note
    assert "upper bound" in note
    assert "flatters the system" in note


def test_the_bias_note_is_empty_when_nothing_failed():
    assert optimistic_bias_note(0, 195580) == ""


def test_the_bias_note_reports_the_remaining_population():
    assert "population is 195579" in optimistic_bias_note(1, 195580)


@pytest.mark.parametrize("unusable,total", [(-1, 10), (11, 10), (0, 0)])
def test_the_bias_note_refuses_impossible_counts(unusable, total):
    with pytest.raises(ValueError):
        optimistic_bias_note(unusable, total)
