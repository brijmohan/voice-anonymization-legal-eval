"""Whether an anonymization system actually produced speech.

A destroyed utterance and a perfectly anonymized one look the same to every
privacy metric in this package, and the destroyed one looks better.

This is not hypothetical. Running the 2026 McAdams baseline over Common Voice
27 produced, for `common_voice_en_25475281`, a file of the correct length whose
samples are 99.93% exact zeros with 67 near-full-scale impulses:

    original        96768 samples, rms 0.0376, 93487 non-zero
    anonymized      96768 samples, rms 0.0096,    67 non-zero

The attacker raised on it, which is the fortunate outcome because it is
visible. The unfortunate outcome is the normal one: a more forgiving extractor
embeds six seconds of near-silence, that embedding sits at chance distance from
the enrolled speaker, and the utterance is counted as a privacy success. Every
destroyed utterance therefore biases the measured privacy of the system
**optimistically**, and nothing in the reported numbers indicates it.

So integrity of the anonymized audio is a precondition for interpreting any
metric computed from it, not a separate quality concern. A protocol that does
not check it cannot distinguish "the system protected this speaker" from "the
system emitted silence".

The thresholds here are deliberately loose. The purpose is to catch output that
is not speech at all, not to judge quality: a system is allowed to sound bad,
and several good ones do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Amplitude below which a sample counts as silence. Well under the quietest
#: speech in a 16-bit recording, which is about 3e-5 for one quantisation step.
SILENCE_FLOOR = 1e-4

#: Output is suspect when the share of audible samples falls below this
#: fraction of the input's. The observed failure collapsed from 0.966 to
#: 0.0007, a factor of 1400, so anything near 1/10 is far outside normal
#: processing loss and well clear of what a vocoder does.
ACTIVE_COLLAPSE_RATIO = 0.1

#: Output with fewer than this share of audible samples is suspect regardless
#: of its input, because no utterance of speech is 99% digital silence.
MIN_ACTIVE_FRACTION = 0.01

#: A sample within this of full scale counts as clipped.
CLIP_LEVEL = 0.999

#: Output is suspect when this share of samples is clipped. Isolated full-scale
#: impulses in silence are the signature of the observed failure.
MAX_CLIPPED_FRACTION = 0.01


def active_fraction(signal: np.ndarray, floor: float = SILENCE_FLOOR) -> float:
    """Share of samples that are audible rather than silence.

    Args:
        signal: Audio samples, any shape, assumed scaled to [-1, 1].
        floor: Amplitude below which a sample counts as silence.

    Returns:
        A fraction in [0, 1]. An empty signal gives 0.0.
    """
    flat = np.asarray(signal, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(flat) > floor) / flat.size)


def clipped_fraction(signal: np.ndarray, level: float = CLIP_LEVEL) -> float:
    """Share of samples at or beyond full scale.

    Args:
        signal: Audio samples, assumed scaled to [-1, 1].
        level: Amplitude counted as clipped.

    Returns:
        A fraction in [0, 1]. An empty signal gives 0.0.
    """
    flat = np.asarray(signal, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(flat) >= level) / flat.size)


@dataclass(frozen=True)
class AudioIntegrity:
    """What an anonymized signal looks like, and whether it is usable.

    Attributes:
        n_samples: Length of the anonymized signal.
        active: Share of audible samples in the anonymized signal.
        reference_active: Share of audible samples in the original, or None
            when no original was given.
        rms: Root mean square of the anonymized signal.
        peak: Largest absolute sample.
        clipped: Share of samples at full scale.
        usable: False when the signal is not speech. A metric computed from an
            unusable signal is not a privacy measurement.
        reasons: Why it was judged unusable, empty when it is usable.
    """

    n_samples: int
    active: float
    reference_active: float | None
    rms: float
    peak: float
    clipped: float
    usable: bool
    reasons: tuple[str, ...]


def check_anonymized(
    anonymized: np.ndarray,
    original: np.ndarray | None = None,
    *,
    floor: float = SILENCE_FLOOR,
    collapse_ratio: float = ACTIVE_COLLAPSE_RATIO,
    min_active: float = MIN_ACTIVE_FRACTION,
    max_clipped: float = MAX_CLIPPED_FRACTION,
) -> AudioIntegrity:
    """Judge whether an anonymized signal is speech at all.

    Args:
        anonymized: The system's output, scaled to [-1, 1].
        original: The input it was produced from, when available. Supplying it
            makes the check far stronger, because a quiet original and a
            destroyed output are otherwise hard to tell apart.
        floor: Amplitude below which a sample counts as silence.
        collapse_ratio: Output is unusable when its audible share falls below
            this multiple of the original's.
        min_active: Output is unusable when its audible share is below this.
        max_clipped: Output is unusable when more than this share is clipped.

    Returns:
        An :class:`AudioIntegrity` describing the signal.

    Raises:
        ValueError: when the anonymized signal is empty, which is a failure
            that needs reporting rather than measuring.
    """
    anon = np.asarray(anonymized, dtype=np.float64).reshape(-1)
    if anon.size == 0:
        raise ValueError("the anonymized signal is empty")

    active = active_fraction(anon, floor)
    clipped = clipped_fraction(anon)
    rms = float(np.sqrt(np.mean(anon**2)))
    peak = float(np.max(np.abs(anon)))

    reference_active = None
    reasons: list[str] = []

    if active < min_active:
        reasons.append(
            f"only {active:.4%} of samples are audible, so this is silence "
            f"rather than speech"
        )
    if clipped > max_clipped:
        reasons.append(
            f"{clipped:.2%} of samples are at full scale, which is impulses "
            f"rather than speech"
        )
    if original is not None:
        orig = np.asarray(original, dtype=np.float64).reshape(-1)
        if orig.size:
            reference_active = active_fraction(orig, floor)
            if reference_active > 0 and active < collapse_ratio * reference_active:
                reasons.append(
                    f"audible share fell from {reference_active:.4%} to "
                    f"{active:.4%}, a factor of "
                    f"{reference_active / max(active, 1e-12):.0f}, so the "
                    f"system removed the speech rather than changing it"
                )

    return AudioIntegrity(
        n_samples=int(anon.size),
        active=active,
        reference_active=reference_active,
        rms=rms,
        peak=peak,
        clipped=clipped,
        usable=not reasons,
        reasons=tuple(reasons),
    )


def optimistic_bias_note(n_unusable: int, n_total: int) -> str:
    """One sentence a report must carry when any utterance was unusable.

    The direction is not neutral and has to be stated. A destroyed utterance
    scores at chance, chance is the best possible privacy score, so excluding
    or silently keeping these utterances both flatter the system. Excluding
    them is the honest option, and it still has to be declared because it
    changes the population.

    Args:
        n_unusable: Utterances whose anonymized audio is not speech.
        n_total: Utterances attempted.

    Returns:
        A sentence for the report. Empty when nothing was unusable.

    Raises:
        ValueError: when the counts are impossible.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    if not 0 <= n_unusable <= n_total:
        raise ValueError(
            f"n_unusable must be between 0 and {n_total}, got {n_unusable}"
        )
    if n_unusable == 0:
        return ""
    return (
        f"{n_unusable} of {n_total} utterances ({n_unusable / n_total:.4%}) "
        f"have anonymized audio that is not speech, and are excluded. The "
        f"population is {n_total - n_unusable}. Had they been kept they would "
        f"have scored near chance, which is the best attainable privacy score, "
        f"so this failure flatters the system in either direction and the "
        f"reported privacy is an upper bound on what the system achieves on "
        f"audio it processes correctly."
    )
