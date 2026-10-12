"""Privacy metrics computed from score distributions rather than a population.

Three proposals that the VoicePrivacy Challenge already ships, implemented here
so a single comparison can cover them alongside the population-based metrics:

- **Cllr**, actual and minimum, after Brümmer and du Preez (2006)
- **Linkability** `D_sys`, after Gomez-Barrero, Galbally, Rathgeb and Busch
  (2017), which is a different quantity from this package's Linkability and
  must not be confused with it
- **ZEBRA**, after Nautsch, Patino, Tomashenko, Yamagishi, Noé, Bonastre,
  Todisco and Evans (Interspeech 2020), reporting a population figure `D_ECE`
  and a worst-case individual figure `max |LLR|` with a categorical tag

**These are implemented against the authors' own reference code**, which the
challenge installs at `evaluation/privacy/asv/metrics/`, not from the papers
alone. That is a stronger basis than this package has for SRD and LID, where
`comparison.py` can only claim to be our reading of the published definitions.
`scripts/verify_trial_metrics.py` checks every function here against that
reference numerically, and the agreement is recorded in
`docs/metric-comparison.md`.

## Why they are in a separate module

**None of these has a population axis.** They are functionals of the mated and
non-mated score distributions, so they answer "how distinguishable are these
two distributions" rather than "can the right speaker be found among `N`".
That is why they were never compared against the population-based metrics:
the two kinds do not take the same input.

It also sharpens what this project measures. The metrics sort by what they are
averaged over, and a procedure should require one from each row rather than
several from one:

| averaged over | metrics |
|---|---|
| trials, as a distribution | EER, Cllr, `D_sys`, ZEBRA `D_ECE` |
| the worst individual | ZEBRA `max |LLR|` |
| a candidate population of size `N`, by rank | Linkability, SRD, LID |
| a candidate population of size `N`, by threshold | Singling Out |

ZEBRA's second figure is the only one in the field that reports an individual
rather than an average, and an average that looks acceptable can conceal one
fully exposed person.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Categorical tags for the strength of evidence, keyed to base-10 LLR ranges.
#: Taken from the reference implementation, which took them from the European
#: Network of Forensic Science Institutes' scale. They are the reason a ZEBRA
#: result can be read by somebody who does not work in speech: the number
#: arrives with a category that already has forensic currency.
ENFSI_TAG_BOUNDS = (
    ("0", 0.0, float(np.finfo(float).eps)),
    ("A", float(np.finfo(float).eps), 1.0),
    ("B", 1.0, 2.0),
    ("C", 2.0, 4.0),
    ("D", 4.0, 5.0),
    ("E", 5.0, 6.0),
    ("F", 6.0, float("inf")),
)


def _expit(x: np.ndarray) -> np.ndarray:
    """Logistic function, computed without overflowing on large magnitudes."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_expit`. Infinite at 0 and 1, which callers rely on."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p) - np.log1p(-p)


def pav(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool adjacent violators: the best monotone fit to ``y``.

    Used to calibrate scores optimally, which is what makes the minimum Cllr a
    property of the scores' discriminative power rather than of whoever set the
    threshold.

    Args:
        y: 1-D array to fit.

    Returns:
        ``(fitted, width, height)``: the monotone fit at full length, then the
        width and height of each constant block.

    Raises:
        ValueError: on an empty or non 1-D input.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"pav takes a 1-D array, got shape {y.shape}")
    if y.size == 0:
        raise ValueError("pav takes a non-empty array")

    n = y.size
    index = np.zeros(n, dtype=int)
    length = np.zeros(n, dtype=int)
    fitted = np.zeros(n)

    ci = 0
    index[ci] = 1
    length[ci] = 1
    fitted[ci] = y[0]
    for j in range(1, n):
        ci += 1
        index[ci] = j + 1
        length[ci] = 1
        fitted[ci] = y[j]
        while ci >= 1 and fitted[ci - 1] >= fitted[ci]:
            nw = length[ci - 1] + length[ci]
            fitted[ci - 1] += (length[ci] / nw) * (fitted[ci] - fitted[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    height = fitted[: ci + 1].copy()
    width = length[: ci + 1].copy()

    # Expand the blocks back to full length, walking from the last block down.
    m = n
    while m >= 0:
        for j in range(index[ci], m + 1):
            fitted[j - 1] = fitted[ci]
        m = index[ci] - 1
        ci -= 1
    return fitted, width, height


def optimal_llr(
    target: np.ndarray,
    nontarget: np.ndarray,
    laplace: bool = False,
    monotonicity_epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimally calibrate scores into log-likelihood ratios, by PAV.

    Args:
        target: Mated scores, uncalibrated.
        nontarget: Non-mated scores, uncalibrated.
        laplace: Pad with one artificial trial of each class at each extreme,
            which keeps the calibrated LLRs finite. ZEBRA needs this, because
            its worst-case figure is a maximum and would otherwise be infinite
            whenever the classes separate.
        monotonicity_epsilon: A tiny increasing ramp added to preserve strict
            monotonicity through ties.

    Returns:
        ``(target_llrs, nontarget_llrs)`` in the order given.

    Raises:
        ValueError: If either input is empty.
    """
    target = np.asarray(target, dtype=np.float64).ravel()
    nontarget = np.asarray(nontarget, dtype=np.float64).ravel()
    if target.size == 0 or nontarget.size == 0:
        raise ValueError("both target and nontarget scores are required")

    scores = np.concatenate([nontarget, target])
    ideal = np.concatenate([np.zeros(nontarget.size), np.ones(target.size)])
    order = np.argsort(scores, kind="mergesort")
    ideal = ideal[order]

    if laplace:
        ideal = np.hstack([1.0, 0.0, ideal, 1.0, 0.0])

    posterior, _width, _height = pav(ideal)
    if laplace:
        posterior = posterior[2 : len(posterior) - 2]

    llrs = _logit(posterior) - np.log(target.size / nontarget.size)
    n = target.size + nontarget.size
    llrs = llrs + np.arange(n) * monotonicity_epsilon / n

    inverse = np.zeros(scores.size, dtype=int)
    inverse[order] = np.arange(scores.size)
    restored = llrs[inverse]
    return restored[nontarget.size :], restored[: nontarget.size]


def cllr(target_llrs: np.ndarray, nontarget_llrs: np.ndarray) -> float:
    """Application-independent detection cost, in bits.

    The expected cost of a binary decision taken on calibrated LLRs, so lower
    is a better attacker and therefore less privacy. A value of 1 bit is what a
    system carrying no information achieves.

    Args:
        target_llrs: Mated log-likelihood ratios.
        nontarget_llrs: Non-mated log-likelihood ratios.

    Returns:
        Cllr in bits.

    Raises:
        ValueError: If either input is empty, or a posterior underflows to
            zero, which means the LLRs are so extreme the cost is unbounded.
    """
    target_llrs = np.asarray(target_llrs, dtype=np.float64).ravel()
    nontarget_llrs = np.asarray(nontarget_llrs, dtype=np.float64).ravel()
    if target_llrs.size == 0 or nontarget_llrs.size == 0:
        raise ValueError("both target and nontarget LLRs are required")

    target_posterior = _expit(target_llrs)
    nontarget_posterior = _expit(-nontarget_llrs)
    if np.any(target_posterior == 0) or np.any(nontarget_posterior == 0):
        raise ValueError(
            "a posterior underflowed to zero, so Cllr is unbounded. Calibrate "
            "the scores first, or use min_cllr which does that for you."
        )
    first = (-np.log(target_posterior)).mean() / np.log(2)
    second = (-np.log(nontarget_posterior)).mean() / np.log(2)
    return float((first + second) / 2)


def min_cllr(
    target: np.ndarray,
    nontarget: np.ndarray,
    monotonicity_epsilon: float = 1e-6,
) -> float:
    """Cllr after optimal calibration, so it measures discrimination alone.

    The gap between this and :func:`cllr` on the raw scores is calibration
    loss, which is the attacker's to fix and not a property of the
    anonymization. Reporting the actual Cllr alone therefore flatters a system
    whose scores happen to be badly calibrated.

    Args:
        target: Mated scores, uncalibrated.
        nontarget: Non-mated scores, uncalibrated.
        monotonicity_epsilon: Passed to :func:`optimal_llr`.

    Returns:
        The minimum Cllr in bits.
    """
    target_llrs, nontarget_llrs = optimal_llr(
        target, nontarget, laplace=False, monotonicity_epsilon=monotonicity_epsilon
    )
    return cllr(target_llrs, nontarget_llrs)


def gomez_barrero_linkability(
    mated: np.ndarray,
    non_mated: np.ndarray,
    omega: float = 1.0,
) -> float:
    """Global linkability `D_sys`, after Gomez-Barrero et al. (2017).

    **This is not the Linkability metric of this package.** That one is a
    closed-set rank-1 identification rate over `N` candidates. This one is a
    functional of the two score distributions with no population axis at all.
    The VoicePrivacy repository ships a `linkability.py` computing this, which
    is the naming hazard recorded in `CLAUDE.md`.

    Zero means the two distributions are indistinguishable, so nothing links;
    one means they are fully separable.

    Args:
        mated: Scores of same-speaker pairs.
        non_mated: Scores of different-speaker pairs.
        omega: Prior ratio of mated to non-mated.

    Returns:
        `D_sys` in [0, 1].

    Raises:
        ValueError: If either input is empty, or omega is not positive.
    """
    mated = np.asarray(mated, dtype=np.float64).ravel()
    non_mated = np.asarray(non_mated, dtype=np.float64).ravel()
    if mated.size == 0 or non_mated.size == 0:
        raise ValueError("both mated and non-mated scores are required")
    if omega <= 0:
        raise ValueError(f"omega must be positive, got {omega}")

    n_bins = min(int(mated.size / 10), 100)
    if n_bins < 1:
        raise ValueError(
            f"{mated.size} mated scores is too few to histogram; the reference "
            f"implementation needs at least 10"
        )
    low = min(mated.min(), non_mated.min())
    high = max(mated.max(), non_mated.max())
    edges = np.linspace(low, high, num=n_bins + 1, endpoint=True)
    centres = (edges[1:] + edges[:-1]) / 2

    y1 = np.histogram(mated, bins=edges, density=True)[0]
    y2 = np.histogram(non_mated, bins=edges, density=True)[0]
    ratio = np.divide(y1, y2, out=np.ones_like(y1), where=y2 != 0)

    d = 2 * (omega * ratio / (1 + omega * ratio)) - 1
    d[omega * ratio <= 1] = 0
    # A bin with mated mass and no non-mated mass links perfectly.
    d[(y2 == 0) & (y1 != 0)] = 1
    return float(np.trapezoid(x=centres, y=d * y1)
                 if hasattr(np, "trapezoid") else np.trapz(x=centres, y=d * y1))


def _int_ece(x: np.ndarray, epsilon: float = 1e-6) -> float:
    """Closed-form integral of the empirical cross-entropy contribution.

    From the reference implementation, where it is derived as

        Z(X) = avg( 0.25 - 1/[2(x-1)] + log(x)/[2(x-1)^2] )

    with ``x`` a likelihood ratio and the argument here its logarithm. An
    infinite LLR contributes the constant term alone, and a magnitude below
    ``epsilon`` is hardcoded because ``exp`` loses precision near zero.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    usable = (~np.isinf(x)) & (np.abs(x) > epsilon)
    contribution = np.zeros(x.size)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xx = x[usable]
        lr_minus_one = np.exp(xx) - 1
        contribution[usable] = (xx - lr_minus_one) / lr_minus_one**2
    contribution[np.abs(x) < epsilon] = -0.5
    return float(0.25 + contribution.mean() / 2)


@dataclass(frozen=True)
class ZebraProfile:
    """ZEBRA's two figures and the categorical tag for the second.

    Attributes:
        population: `D_ECE`, expected privacy disclosure over the population,
            in bits. Zero is perfect privacy.
        worst_case: `max |LLR|` in base 10, the strongest evidence available
            about any single individual. Zero is perfect privacy.
        tag: The categorical strength-of-evidence tag for ``worst_case``, from
            ``0`` for none through ``F`` for the strongest.
    """

    population: float
    worst_case: float
    tag: str

    def __str__(self) -> str:
        return (f"ZEBRA ({self.population:.3f}, {self.worst_case:.3f}, "
                f"{self.tag})")


def category_tag(max_abs_llr: float) -> str:
    """Categorical strength-of-evidence tag for a base-10 LLR magnitude.

    Args:
        max_abs_llr: The worst-case figure, non-negative.

    Returns:
        One of ``0``, ``A`` to ``F``.

    Raises:
        ValueError: If the value is negative or not finite.
    """
    if not np.isfinite(max_abs_llr) or max_abs_llr < 0:
        raise ValueError(f"expected a non-negative finite value, got {max_abs_llr}")
    for tag, low, high in ENFSI_TAG_BOUNDS:
        if low <= max_abs_llr < high:
            return tag
    return ENFSI_TAG_BOUNDS[-1][0]


def zebra(mated: np.ndarray, non_mated: np.ndarray) -> ZebraProfile:
    """The ZEBRA profile: a population figure and a worst-case individual one.

    Reporting both is the design choice worth borrowing. An average that looks
    acceptable can conceal one individual about whom the evidence is
    overwhelming, and no other metric in this package reports that.

    Calibration uses PAV with Laplace padding, because the worst case is a
    maximum and would otherwise be infinite whenever the two classes separate.

    Args:
        mated: Scores of same-speaker trials, uncalibrated.
        non_mated: Scores of different-speaker trials, uncalibrated.

    Returns:
        The :class:`ZebraProfile`.
    """
    target_llrs, nontarget_llrs = optimal_llr(mated, non_mated, laplace=True)
    population = (_int_ece(target_llrs) + _int_ece(-nontarget_llrs)) / np.log(2)
    worst = float(np.abs(np.hstack([target_llrs, nontarget_llrs])).max() / np.log(10))
    return ZebraProfile(population=float(population), worst_case=worst,
                        tag=category_tag(worst))
