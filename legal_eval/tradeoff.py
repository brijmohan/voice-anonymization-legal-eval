"""Privacy against utility, on one plane, for a declared use case.

The field reports a privacy number and a utility number and leaves the reader
to put them together. They cannot be put together, because the question anybody
actually has is comparative: is this system better than that one? Two systems
with different privacy and different utility are not ordered by either number
alone, and a table of four figures invites a ranking it does not support.

This module supplies the two things that make the comparison honest.

**Dominance.** A system is better than another only if it is at least as
private and at least as useful, with at least one of those strict. Anything
else is a trade, and a trade has to be stated as a trade. `incomparable` exists
so that "we could not order these" is a reportable result rather than a gap
somebody fills with a preference.

**The role of an attribute.** This is the part the field gets wrong, and it is
not a detail. Whether recovering gender from anonymized speech is a failure or
a requirement depends entirely on what the data is for. A call centre analysing
customer sentiment needs emotion to survive and must not be able to identify
the caller. A corpus released for speech research may need accent preserved and
age removed. The measurement is identical in both directions: the attribute
inference advantage from `legal_eval.inference`. Only the sign of its
interpretation changes.

So a system cannot be scored here without a declared use case. That is enforced
rather than advised, because a privacy evaluation that does not say what the
data is for has quietly assumed every attribute is prohibited, which no real
deployment ever is. Under that assumption a system that destroys the signal
scores perfectly, which is the same failure as rewarding silence.

**What this module does not do.** It takes utility as a number somebody else
measured. Word error rate, mean opinion score and emotion recognition accuracy
are all produced by toolkits that exist, and reimplementing them here would
produce worse copies. What is missing from the field is not another utility
metric. It is the discipline of reading the two axes together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from legal_eval.inference import AT_CHANCE

#: The controller needs this attribute to survive. Disclosure is utility, and
#: its absence is a utility failure.
REQUIRED = "required"

#: The controller must not be able to recover this attribute. Disclosure is
#: leakage under the third Article 29 criterion.
PROHIBITED = "prohibited"

#: Neither needed nor forbidden by this use case. Still measured and still
#: reported, because an attribute nobody asked about is how a surprise arrives.
INCIDENTAL = "incidental"

ROLES = (REQUIRED, PROHIBITED, INCIDENTAL)

#: Re-exported so a caller of this module does not have to know which module
#: owns the threshold. `legal_eval.inference` defines it, and there is exactly
#: one definition on purpose.
__all__ = ["AT_CHANCE"]


@dataclass(frozen=True)
class AttributeRole:
    """What a use case does with one attribute.

    Attributes:
        attribute: Name of the attribute, matching the inference measurement.
        role: One of `REQUIRED`, `PROHIBITED` or `INCIDENTAL`.
        justification: Why this use case needs or forbids it. Required for a
            prohibited or required attribute, because an unjustified role is
            how a convenient one gets assigned.
    """

    attribute: str
    role: str
    justification: str = ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"role must be one of {ROLES}, got {self.role!r}")
        if self.role in (REQUIRED, PROHIBITED) and not self.justification:
            raise ValueError(
                f"{self.attribute} is {self.role} and carries no "
                f"justification. State why this use case needs or forbids it."
            )


@dataclass(frozen=True)
class UseCase:
    """What a release of anonymized speech is for.

    Without this, an evaluation has assumed that every attribute is prohibited
    and that content does not matter, which describes no deployment.

    Attributes:
        name: Short name of the use case.
        roles: What the use case does with each attribute.
        content_required: Whether the verbal content has to survive. When it
            does, obligation G1 binds: identification through what was said is
            then inside the deployment rather than outside it.
        notes: Anything a reader needs to judge whether the roles are right.
    """

    name: str
    roles: tuple[AttributeRole, ...]
    content_required: bool
    notes: str = ""

    def __post_init__(self) -> None:
        seen = [r.attribute for r in self.roles]
        duplicated = {a for a in seen if seen.count(a) > 1}
        if duplicated:
            raise ValueError(
                f"{self.name}: {sorted(duplicated)} given more than one role. "
                f"An attribute cannot be both required and prohibited."
            )

    def role_of(self, attribute: str) -> str:
        """The role this use case gives an attribute, or `INCIDENTAL`."""
        for r in self.roles:
            if r.attribute == attribute:
                return r.role
        return INCIDENTAL


@dataclass(frozen=True)
class AttributeOutcome:
    """One attribute, read against the role its use case gives it.

    Attributes:
        attribute: Name of the attribute.
        role: The role the use case gave it.
        advantage: Inference advantage on the anonymized release, the share of
            available headroom an attacker captured.
        original_advantage: The same figure on the unanonymized recordings,
            which is the most that could have been retained or removed.
        met: Whether the role was satisfied. A required attribute is met when
            it survives, a prohibited one when it is at chance.
        reason: Why, in words, for a report that does not make the reader
            recompute it.
    """

    attribute: str
    role: str
    advantage: float
    original_advantage: float
    met: bool
    reason: str


def _retained(advantage: float, original: float) -> float:
    """The share of the original's recoverable attribute that survived."""
    if original <= 0:
        return 0.0
    return advantage / original


def score_attributes(
    use_case: UseCase,
    advantages: dict[str, float],
    original_advantages: dict[str, float],
    *,
    retain_threshold: float = 0.8,
) -> list[AttributeOutcome]:
    """Read each attribute against the role its use case gives it.

    Args:
        use_case: What the release is for.
        advantages: Attribute name to inference advantage on the anonymized
            release.
        original_advantages: The same on the unanonymized recordings. Needed
            because an attribute the attacker could not recover from the
            original was never there to remove, and a system must not be
            credited with removing it.
        retain_threshold: Share of the original's advantage a required
            attribute must keep. The default is lenient on purpose: a
            requirement that fails on a system the use case would accept gets
            switched off.

    Returns:
        One outcome per attribute mentioned by either the use case or the
        measurements, so an attribute nobody declared is still reported.

    Raises:
        ValueError: when a declared attribute was not measured at all. A role
            assigned and never checked is worse than no role.
    """
    declared = {r.attribute for r in use_case.roles}
    missing = sorted(declared - set(advantages))
    if missing:
        raise ValueError(
            f"{use_case.name} declares {missing} but they were not measured. "
            f"An attribute with a role and no measurement is an unchecked claim."
        )

    out: list[AttributeOutcome] = []
    for attribute in sorted(set(advantages) | declared):
        advantage = float(advantages[attribute])
        original = float(original_advantages.get(attribute, float("nan")))
        role = use_case.role_of(attribute)

        if role == PROHIBITED:
            met = advantage <= AT_CHANCE
            if met:
                reason = (f"at chance, advantage {advantage:+.4f}, so the "
                          f"attribute is not recoverable")
            else:
                reason = (f"advantage {advantage:+.4f} above the {AT_CHANCE} "
                          f"chance band, so it is still recoverable")
        elif role == REQUIRED:
            if not math.isfinite(original) or original <= AT_CHANCE:
                met = False
                reason = ("the original recordings do not carry this attribute "
                          "recoverably, so the use case cannot be served by "
                          "this data at all")
            else:
                share = _retained(advantage, original)
                met = share >= retain_threshold
                reason = (f"{share:.1%} of the original's advantage retained "
                          f"against a {retain_threshold:.0%} requirement")
        else:
            met = True
            reason = (f"incidental to this use case, advantage "
                      f"{advantage:+.4f}, reported rather than judged")

        out.append(AttributeOutcome(
            attribute=attribute, role=role, advantage=advantage,
            original_advantage=original, met=met, reason=reason,
        ))
    return out


@dataclass(frozen=True)
class SystemPoint:
    """One system as a point on the privacy and utility plane.

    Attributes:
        name: System name.
        exposure: Privacy as distance above chance, so lower is better and
            zero means an attacker learned nothing. `legal_eval.criteria`
            produces this from any of the nine quantities.
        utility: Utility on a scale where higher is better. Convert first: a
            word error rate has to be negated or expressed as accuracy, or
            every comparison here inverts.
        utility_metric: What the utility number is, so two points measured
            differently are not silently compared.
    """

    name: str
    exposure: float
    utility: float
    utility_metric: str = "unspecified"


def _check_commensurable(points: list[SystemPoint]) -> None:
    metrics = {p.utility_metric for p in points}
    if len(metrics) > 1:
        raise ValueError(
            f"utility measured {len(metrics)} different ways: "
            f"{sorted(metrics)}. Points on different utility scales are not "
            f"comparable, and ranking them would be arithmetic on unlike units."
        )


def dominates(a: SystemPoint, b: SystemPoint) -> bool:
    """Whether ``a`` is better than ``b`` on both axes, strictly on one.

    This is the only circumstance in which one system can be called better
    than another from a privacy figure and a utility figure.

    Args:
        a: Candidate.
        b: Comparison.

    Returns:
        True when ``a`` is at least as private and at least as useful, and
        strictly better on at least one.
    """
    at_least_as_good = a.exposure <= b.exposure and a.utility >= b.utility
    strictly_better = a.exposure < b.exposure or a.utility > b.utility
    return at_least_as_good and strictly_better


def incomparable(a: SystemPoint, b: SystemPoint) -> bool:
    """Whether neither system dominates the other.

    Reported rather than resolved. Two systems that trade privacy against
    utility are ordered only by a preference, and a preference belongs to the
    controller and the regulator, not to this package.
    """
    return not dominates(a, b) and not dominates(b, a) and a != b


def pareto_frontier(points: list[SystemPoint]) -> list[SystemPoint]:
    """The systems no other system dominates.

    Args:
        points: Systems on one utility scale.

    Returns:
        The non-dominated systems, ordered by exposure then by name. Everything
        omitted is beaten outright on both axes, which is the only claim this
        module will make about ordering.

    Raises:
        ValueError: when the utility numbers are not on one scale.
    """
    _check_commensurable(points)
    frontier = [p for p in points
                if not any(dominates(q, p) for q in points if q != p)]
    return sorted(frontier, key=lambda p: (p.exposure, p.name))


def equal_utility_exposure(
    points: list[SystemPoint], utility: float
) -> float | None:
    """The best exposure achievable at a given utility, by interpolation.

    This answers the question two tables cannot: of two systems at different
    operating points, which is more private *at the utility I need*. The
    frontier is interpolated linearly between its neighbours, which assumes a
    system can be tuned between them. Say so when quoting the result, because
    the assumption is about the systems and not about the arithmetic.

    Args:
        points: Systems on one utility scale.
        utility: The utility level required.

    Returns:
        The interpolated exposure, or None when the requested utility is
        outside the range any system achieved. None is the honest answer there:
        extrapolating a frontier invents an operating point nobody measured.

    Raises:
        ValueError: when the utility numbers are not on one scale.
    """
    frontier = pareto_frontier(points)
    by_utility = sorted(frontier, key=lambda p: p.utility)
    if not by_utility:
        return None
    if utility < by_utility[0].utility or utility > by_utility[-1].utility:
        return None
    for left, right in zip(by_utility, by_utility[1:], strict=False):
        if left.utility <= utility <= right.utility:
            span = right.utility - left.utility
            if span == 0:
                return min(left.exposure, right.exposure)
            weight = (utility - left.utility) / span
            return left.exposure + weight * (right.exposure - left.exposure)
    return by_utility[-1].exposure


@dataclass(frozen=True)
class TradeoffReport:
    """What can and cannot be said about a set of systems.

    Attributes:
        frontier: The systems nothing dominates.
        dominated: ``(loser, winner)`` pairs, where the loser is beaten on both
            axes and can be set aside.
        trades: ``(a, b)`` pairs that are incomparable, which is a result.
        utility_metric: The one scale all of this was computed on.
    """

    frontier: list[SystemPoint]
    dominated: list[tuple[str, str]]
    trades: list[tuple[str, str]]
    utility_metric: str

    def summary(self) -> str:
        """The verdict, including the refusal to rank where ranking is unsound."""
        lines = [
            f"Utility measured as {self.utility_metric}. "
            f"{len(self.frontier)} system(s) on the frontier."
        ]
        for point in self.frontier:
            lines.append(f"  {point.name}: exposure {point.exposure:.4f}, "
                         f"utility {point.utility:.4f}")
        for loser, winner in self.dominated:
            lines.append(f"  {loser} is beaten outright by {winner}, on both "
                         f"axes, so it can be set aside")
        if self.trades:
            lines.append(f"  {len(self.trades)} pair(s) trade privacy against "
                         f"utility and are not ordered here:")
            for a, b in self.trades:
                lines.append(f"    {a} against {b}")
            lines.append("  Ordering those is a choice about how much utility "
                         "a unit of privacy is worth, which belongs to the "
                         "controller and the regulator.")
        return "\n".join(lines)


def report(points: list[SystemPoint]) -> TradeoffReport:
    """Everything that follows from the two axes, and nothing that does not.

    Args:
        points: Systems on one utility scale.

    Returns:
        A `TradeoffReport`.

    Raises:
        ValueError: when fewer than two systems are given, or the utility
            numbers are not on one scale.
    """
    if len(points) < 2:
        raise ValueError("a trade-off needs at least two systems to compare")
    _check_commensurable(points)
    frontier = pareto_frontier(points)
    on_frontier = {p.name for p in frontier}

    dominated: list[tuple[str, str]] = []
    for p in points:
        if p.name in on_frontier:
            continue
        winners = [q.name for q in points if dominates(q, p)]
        dominated.append((p.name, sorted(winners)[0]))

    trades: list[tuple[str, str]] = []
    for i, a in enumerate(points):
        for b in points[i + 1:]:
            if incomparable(a, b):
                trades.append(tuple(sorted((a.name, b.name))))
    return TradeoffReport(
        frontier=frontier,
        dominated=sorted(dominated),
        trades=sorted(set(trades)),
        utility_metric=points[0].utility_metric,
    )
