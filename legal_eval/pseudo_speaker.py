r"""Do generated pseudo-voices identify real people? (CNIL obligation G3)

The CNIL's technical opinion on Voice Harbor asks for exactly this:

    The "pseudo-identities" generated to anonymize a voice dataset warrant
    vigilance. While it is understood that a pseudo-identity does not
    correspond to a specific real person but to a "mix" of several people, it
    should be ensured that it does not make it possible to identify an
    individual.

See ``docs/cnil-opinion.md``. Nothing else in this package measures it.

The decision rule
-----------------
"Is this pseudo-voice close to a real speaker?" has no answer without a scale,
because cosine similarity is not interpretable on its own. The rule used here
is operational rather than arbitrary:

    A pseudo-voice **identifies** a real speaker when a speaker verification
    system, operating at its own equal error rate threshold, would accept the
    pseudo-voice as that speaker.

That threshold comes from the same embedding space, measured on genuine
same-speaker and different-speaker pairs, so it is the threshold a real
attacker's system would use. It is a property of the verifier, not a number we
chose.

Three quantities follow, and each answers a different question:

* **Identification rate.** The share of pseudo-voices whose nearest real
  speaker clears the verification threshold. Above zero means some pseudo-voice
  would be accepted as a real person. This is the G3 answer.
* **Collision rate.** The share of pseudo-voice pairs that clear the threshold
  against *each other*. Two source speakers assigned mutually acceptable
  targets become linkable to one another, which is a linkability failure that
  no metric in this package reports.
* **Margin.** How far the nearest-real similarity sits from the threshold, in
  units of the different-speaker standard deviation. A small positive margin on
  a large population is a weaker result than it looks, which is why the
  distribution is returned rather than only the rate.

What this cannot tell you
-------------------------
A pseudo-voice that clears the threshold against a speaker **in the generator's
own training pool** is a different and more serious finding than one that
clears it against an unrelated speaker, because the former suggests the
generator memorised. This module measures similarity to whatever pool it is
given and does not know which pool that is. The caller has to say.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from legal_eval.metrics.eer import rocch_eer


def chance_identification_rate(far: float, n_speakers: int) -> float:
    r"""Chance that a verifier accepts an unrelated voice as *some* enrolled speaker.

    This is the baseline the identification rate must be read against, and
    without it the rate is close to meaningless. A verifier with false accept
    rate ``far``, searched against ``n_speakers`` enrolled identities, accepts
    at least one of them for a voice belonging to none:

    .. math:: 1 - (1 - \mathrm{far})^{n}

    The consequence is uncomfortable. At the equal error rate threshold
    ``far`` is roughly the EER, so a verifier at EER 0.087 searched against 29
    speakers accepts an unrelated voice with probability about 0.92. Any
    "identification rate" near that number is the population size talking, not
    the anonymization.

    So a high rate at a small ``n`` is not evidence that pseudo-voices resemble
    real people. Evidence requires either a rate above this baseline by more
    than its error, or a population large enough for the baseline to be small.
    The second is the honest route, and it is the same constraint that CNIL
    obligation G2 imposes on every other metric here.

    Args:
        far: False accept rate per comparison, in ``[0, 1]``.
        n_speakers: Number of enrolled identities searched.

    Returns:
        The chance identification rate, in ``[0, 1]``.
    """
    if not 0.0 <= far <= 1.0:
        raise ValueError(f"far must be in [0, 1], got {far}")
    if n_speakers < 1:
        raise ValueError(f"n_speakers must be at least 1, got {n_speakers}")
    return float(1.0 - (1.0 - far) ** n_speakers)


def population_for_chance(far: float, target: float = 0.05) -> int:
    """Smallest population at which the chance rate stays below ``target``.

    Inverts :func:`chance_identification_rate`. Answers "how many speakers may
    be enrolled before a chance match becomes likely", which is the sizing
    question for a G3 experiment.

    Args:
        far: False accept rate per comparison.
        target: Acceptable chance rate.

    Returns:
        The largest ``n`` with chance rate at or below ``target``, or ``0`` when
        even a single comparison exceeds it. Zero is the interesting answer: it
        means no enrolment size rescues the experiment and the verifier itself
        has to improve. At EER 0.087, a 5% target returns 0.
    """
    if far <= 0.0:
        raise ValueError("far must be positive for this to be meaningful")
    if far >= 1.0:
        return 0
    if far > target:
        return 0
    return max(1, int(math.log1p(-target) / math.log1p(-far)))

@dataclass
class PseudoSpeakerAudit:
    """Result of a pseudo-voice identifiability audit.

    Attributes:
        threshold: The verification threshold, from the real speakers' own EER.
        eer: The verifier's equal error rate in this embedding space, which says
            how trustworthy the threshold is. A high EER means a weak verifier
            and a permissive threshold.
        identification_rate: Share of pseudo-voices whose nearest real speaker
            clears the threshold. The G3 answer. Zero is the desired result.
        n_identified: How many pseudo-voices those are.
        collision_rate: Share of pseudo-voice pairs that clear the threshold
            against each other.
        nearest_similarity: Per pseudo-voice, the similarity to its nearest real
            speaker.
        nearest_index: Per pseudo-voice, which real speaker that was.
        margin: ``nearest_similarity - threshold``, in units of the
            different-speaker standard deviation. Positive means identified.
        nonmated_mean: Mean similarity between different real speakers, the
            reference for "unrelated".
        nonmated_std: Its standard deviation.
        mated_mean: Mean similarity between utterances of the same real speaker,
            the reference for "a genuine match".
        chance_identification_rate: What ``identification_rate`` would be for
            voices belonging to nobody enrolled, given this verifier's error
            rate and this many speakers. The rate means nothing on its own:
            compare the two.
        n_real_speakers: How many identities were searched.
    """

    threshold: float
    eer: float
    identification_rate: float
    n_identified: int
    collision_rate: float
    nearest_similarity: np.ndarray = field(repr=False)
    nearest_index: np.ndarray = field(repr=False)
    margin: np.ndarray = field(repr=False)
    nonmated_mean: float = 0.0
    nonmated_std: float = 1.0
    mated_mean: float = 0.0
    chance_identification_rate: float = 0.0
    n_real_speakers: int = 0

    def summary(self) -> str:
        """One readable block, for a log or a report."""
        verdict = (
            "NO pseudo-voice clears the verification threshold"
            if self.n_identified == 0
            else f"{self.n_identified} pseudo-voice(s) clear the threshold"
        )
        excess = self.identification_rate - self.chance_identification_rate
        reading = (
            "at or below chance, so this says nothing about the pseudo-voices"
            if excess <= 0.0
            else f"{excess:+.4f} above chance"
        )
        return (
            f"verifier EER {self.eer:.4f}, threshold {self.threshold:.4f}\n"
            f"real speakers: mated mean {self.mated_mean:.4f}, "
            f"nonmated mean {self.nonmated_mean:.4f} "
            f"(sd {self.nonmated_std:.4f})\n"
            f"nearest-real similarity: max {float(self.nearest_similarity.max()):.4f}, "
            f"mean {float(self.nearest_similarity.mean()):.4f}\n"
            f"identification rate {self.identification_rate:.4f} -> {verdict}\n"
            f"chance rate at {self.n_real_speakers} speakers "
            f"{self.chance_identification_rate:.4f}, so the rate is {reading}\n"
            f"collision rate between pseudo-voices {self.collision_rate:.4f}"
        )


def _unit(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D array of embeddings, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def verification_threshold(
    real_embeddings: np.ndarray, speaker_ids: np.ndarray
) -> tuple[float, float, dict[str, float]]:
    """The EER threshold of the real speakers in their own embedding space.

    The threshold is not chosen, it is measured: the similarity at which a
    verifier's false accept and false reject rates are equal on genuine pairs.
    That is the operating point an attacker's system would use.

    Args:
        real_embeddings: ``(n_items, dim)``, several items per speaker.
        speaker_ids: ``(n_items,)`` speaker label per row.

    Returns:
        ``(threshold, eer, references)`` where ``references`` holds the mated and
        nonmated means and the nonmated standard deviation.

    Raises:
        ValueError: If no speaker has two items, so no mated pair exists.
    """
    unit = _unit(real_embeddings)
    ids = np.asarray(speaker_ids)
    similarity = unit @ unit.T
    upper = np.triu_indices(len(unit), k=1)
    same = ids[upper[0]] == ids[upper[1]]
    mated = similarity[upper][same]
    nonmated = similarity[upper][~same]
    if mated.size == 0:
        raise ValueError(
            "no same-speaker pair in the reference set, so no verification "
            "threshold can be measured; supply several items per speaker"
        )
    eer = rocch_eer(mated, nonmated)
    # The threshold at which the two error rates meet: the eer-quantile of the
    # mated scores from below equals the (1-eer)-quantile of the nonmated ones.
    threshold = float(
        0.5 * (np.quantile(mated, eer) + np.quantile(nonmated, 1.0 - eer))
    )
    return (
        threshold,
        float(eer),
        {
            "mated_mean": float(mated.mean()),
            "nonmated_mean": float(nonmated.mean()),
            "nonmated_std": float(nonmated.std()) or 1.0,
        },
    )


def audit_pseudo_speakers(
    pseudo_embeddings: np.ndarray,
    real_embeddings: np.ndarray,
    speaker_ids: np.ndarray,
    threshold: float | None = None,
) -> PseudoSpeakerAudit:
    """Measure whether pseudo-voices would be accepted as real speakers.

    Args:
        pseudo_embeddings: ``(n_pseudo, dim)`` generated speaker embeddings.
        real_embeddings: ``(n_items, dim)`` real speaker embeddings, several per
            speaker so a threshold can be measured.
        speaker_ids: ``(n_items,)`` speaker label per real row.
        threshold: Override the measured threshold. Supplying one makes the
            result depend on a choice, so the default measures it instead.

    Returns:
        The :class:`PseudoSpeakerAudit`.
    """
    measured_threshold, eer, refs = verification_threshold(
        real_embeddings, speaker_ids
    )
    if threshold is None:
        threshold = measured_threshold

    pseudo = _unit(pseudo_embeddings)
    real = _unit(real_embeddings)
    ids = np.asarray(speaker_ids)

    # Compare against per-speaker centroids, which is what a verifier enrolls,
    # rather than against individual utterances.
    unique = np.unique(ids)
    centroids = _unit(
        np.stack([real[ids == speaker].mean(axis=0) for speaker in unique])
    )

    similarity = pseudo @ centroids.T
    nearest_index = similarity.argmax(axis=1)
    nearest = similarity[np.arange(len(pseudo)), nearest_index]
    identified = nearest > threshold

    if len(pseudo) > 1:
        between = pseudo @ pseudo.T
        upper = np.triu_indices(len(pseudo), k=1)
        collisions = between[upper] > threshold
        collision_rate = float(collisions.mean())
    else:
        collision_rate = 0.0

    return PseudoSpeakerAudit(
        threshold=float(threshold),
        eer=eer,
        chance_identification_rate=chance_identification_rate(eer, len(unique)),
        n_real_speakers=len(unique),
        identification_rate=float(identified.mean()),
        n_identified=int(identified.sum()),
        collision_rate=collision_rate,
        nearest_similarity=nearest,
        nearest_index=unique[nearest_index],
        margin=(nearest - threshold) / refs["nonmated_std"],
        nonmated_mean=refs["nonmated_mean"],
        nonmated_std=refs["nonmated_std"],
        mated_mean=refs["mated_mean"],
    )
