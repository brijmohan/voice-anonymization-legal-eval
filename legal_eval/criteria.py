"""Report the three Article 29 criteria side by side, on one scale.

Opinion 05/2014 asks three questions of a dataset, and answering two of them
well is not an answer. A controller who reports only verification numbers has
addressed singling out and linkability and has said nothing about inference.
This module exists to make that silence visible.

The problem it solves is that the nine privacy quantities in the field are not
commensurable. They have different units, different chance levels, different
ceilings, and two of them run the other way: a high EER means privacy while a
high linkability means the opposite. Put raw in one table they invite exactly
the comparison they cannot support.

What makes them comparable is not the reading but the distance from chance.
Every metric here is converted to an **exposure**: how far the reading sits
above the value an attacker with no information would obtain, oriented so that
larger always means more disclosure. The original recording's exposure is then
the room a system has to work in, and what a system achieves is the fraction of
that room it removes:

    reduction = (original exposure - system exposure) / original exposure

At 1.0 the system has brought the metric down to chance. At 0.0 it has changed
nothing. Below 0.0 the anonymized copy is more disclosive than the original,
which happens and must not be reported quietly.

Expressed this way the criteria can be set beside each other honestly, because
each is measured against its own chance level in its own units. A system whose
reduction is 0.6 on linkability and 0.0 on inference has not half-anonymized
the data. It has addressed one criterion of three.

Two cautions are built in rather than left to the reader.

First, the room itself is reported. The Singling Out room collapses with
population size, from 0.53 at N=2 to 0.025 at N=9779, so a reduction computed
there divides by almost nothing and swings wildly. `CriterionReading.reliable`
is False when the room is too small for the ratio to mean anything, and no
summary draws a conclusion from an unreliable row.

Second, the mapping from metric to criterion is stated explicitly in
`CRITERION_OF_METRIC` rather than assumed. Six of the nine quantities speak to
linkability, one to singling out, and exactly one to inference. That imbalance
is the finding, not an accident of this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from legal_eval.preflight import chance_levels

#: The three criteria of Opinion 05/2014, in the order the Opinion states them.
CRITERIA = ("singling_out", "linkability", "inference")

#: Which criterion each metric speaks to, and nothing more. Six quantities
#: bear on linkability, one on singling out, and one on inference. A report
#: built from the first six alone answers one question of three.
CRITERION_OF_METRIC = {
    "singling_out": "singling_out",
    "linkability": "linkability",
    "srd_idr": "linkability",
    "lid_pdr": "linkability",
    "eer": "linkability",
    "cllr_min": "linkability",
    "gomez_barrero_dsys": "linkability",
    "zebra_population": "linkability",
    "inference_advantage": "inference",
}

#: Metrics whose chance level does not depend on the population size, with the
#: value an uninformative attacker obtains. The rest come from
#: `preflight.chance_levels`, which needs N.
FIXED_CHANCE = {
    # Singling Out is the one legal metric whose chance level is a constant: a
    # predicate carrying no information isolates exactly one record with
    # probability exp(-1) whatever the population size. That is also why its
    # room above chance collapses as N grows while Linkability's does not.
    "singling_out": math.exp(-1),
    # An EER of 0.5 is a coin flip, and privacy increases as the EER rises, so
    # this is the one family that has to be inverted.
    "eer": 0.5,
    # Cllr is a cost in bits. One bit is what an attacker who learned nothing
    # pays, and lower is better for the attacker, so this inverts too.
    "cllr_min": 1.0,
    "cllr_actual": 1.0,
    "gomez_barrero_dsys": 0.0,
    "zebra_population": 0.0,
    # Advantage is already defined as headroom above the population prior, so
    # its chance level is zero by construction.
    "inference_advantage": 0.0,
}

#: Metrics where a larger reading means more privacy, so exposure runs the
#: other way. Getting this wrong inverts a conclusion rather than blurring it.
INVERTED = frozenset({"eer", "cllr_min", "cllr_actual"})

#: Below this much room above chance, a reduction is a ratio of two small
#: numbers and is not reported as a finding. The Singling Out room at N=9779 is
#: 0.0247, which is how this threshold came to be needed.
MIN_ROOM = 0.05


@dataclass(frozen=True)
class CriterionReading:
    """One metric, for one system, expressed as distance from chance.

    Attributes:
        metric: The metric name.
        criterion: Which of the three Article 29 criteria it speaks to.
        value: The metric as reported, in its own units.
        chance: What an attacker with no information would obtain.
        exposure: Distance above chance, oriented so larger means more
            disclosure. Zero means the reading is at chance.
        room: The original recording's exposure for this metric, which is the
            most any system could remove.
        reduction: Fraction of ``room`` the system removed. 1.0 is down to
            chance, 0.0 is no change, and negative means the anonymized copy
            discloses more than the original did.
        reliable: False when ``room`` is below `MIN_ROOM`, where the reduction
            divides by almost nothing and must not be read as a result.
    """

    metric: str
    criterion: str
    value: float
    chance: float
    exposure: float
    room: float
    reduction: float
    reliable: bool

    def __str__(self) -> str:
        mark = "" if self.reliable else "  (room too small to read)"
        return (f"{self.metric}: {self.value:.4f} against chance "
                f"{self.chance:.4f}, {self.reduction:+.1%} of the room "
                f"removed{mark}")


def chance_for(metric: str, n: int | None = None) -> float:
    """The value a metric takes when the attacker has learned nothing.

    Args:
        metric: Metric name.
        n: Candidate population size, required for the metrics whose chance
            level is ``1/n``.

    Returns:
        The chance level in the metric's own units.

    Raises:
        ValueError: when the metric is unknown, or when it needs ``n`` and none
            was given.
    """
    if metric in FIXED_CHANCE:
        return FIXED_CHANCE[metric]
    if n is None:
        raise ValueError(
            f"{metric} has a population-dependent chance level, so n is required"
        )
    levels = chance_levels(n)
    if metric not in levels:
        raise ValueError(f"no chance level known for {metric}")
    return levels[metric]


def exposure(metric: str, value: float, n: int | None = None) -> float:
    """How far a reading sits above chance, oriented so larger is worse.

    Args:
        metric: Metric name.
        value: The reading, in the metric's own units.
        n: Candidate population size, where the metric needs it.

    Returns:
        Distance above chance. Zero is an attacker who learned nothing.
        Negative readings occur from sampling noise and are kept rather than
        clipped, because clipping them would hide a system that reads below
        chance for the wrong reason.

    Raises:
        ValueError: when ``value`` is not finite.
    """
    if not math.isfinite(value):
        raise ValueError(f"{metric} is {value}, which is not a measurement")
    chance = chance_for(metric, n)
    return chance - value if metric in INVERTED else value - chance


def read_criterion(
    metric: str,
    original: float,
    system: float,
    n: int | None = None,
) -> CriterionReading:
    """Express one metric as the fraction of the original's room removed.

    Args:
        metric: Metric name. Must appear in `CRITERION_OF_METRIC`.
        original: The reading on the unanonymized recordings.
        system: The reading on the anonymized copy.
        n: Candidate population size, where the metric needs it. Both readings
            must come from the same ``n``, or the comparison is between two
            different questions.

    Returns:
        A `CriterionReading`. When the original sits at or below chance there
        was no room to remove, so ``reduction`` is 0.0 and ``reliable`` is
        False rather than the ratio being formed at all.

    Raises:
        ValueError: when the metric is unknown.
    """
    if metric not in CRITERION_OF_METRIC:
        raise ValueError(
            f"{metric} is not mapped to an Article 29 criterion. Add it to "
            f"CRITERION_OF_METRIC with the criterion it actually speaks to."
        )
    room = exposure(metric, original, n)
    got = exposure(metric, system, n)
    reliable = room >= MIN_ROOM
    # Forming the ratio against a room of zero or less would produce an
    # infinity or a sign flip that reads as a strong result.
    reduction = (room - got) / room if room > 0 else 0.0
    return CriterionReading(
        metric=metric,
        criterion=CRITERION_OF_METRIC[metric],
        value=float(system),
        chance=chance_for(metric, n),
        exposure=float(got),
        room=float(room),
        reduction=float(reduction),
        reliable=bool(reliable),
    )


def by_criterion(
    readings: list[CriterionReading],
) -> dict[str, list[CriterionReading]]:
    """Group readings under the criterion each one speaks to.

    Args:
        readings: Readings for a single system.

    Returns:
        Criterion name to its readings, including criteria with no reading at
        all, because an empty criterion is the result worth seeing.
    """
    out: dict[str, list[CriterionReading]] = {c: [] for c in CRITERIA}
    for reading in readings:
        out[reading.criterion].append(reading)
    return out


def criteria_split(readings: list[CriterionReading]) -> dict[str, object]:
    """Summarise whether a system treated the three criteria alike.

    The question this answers is the one a controller needs and no single
    metric provides: having reduced the verification numbers, has anything
    been done about inference?

    Args:
        readings: Readings for a single system, from `read_criterion`.

    Returns:
        A mapping with, per criterion, the median reduction over its reliable
        readings and how many readings it rests on, plus ``spread``, the
        difference between the largest and smallest of those medians, and
        ``unaddressed``, the criteria whose median reduction is no better than
        nothing. ``spread`` is None when fewer than two criteria have a
        reliable reading, because a split cannot be measured against a gap.
    """
    grouped = by_criterion([r for r in readings if r.reliable])
    medians: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for criterion in CRITERIA:
        rows = grouped[criterion]
        counts[criterion] = len(rows)
        if rows:
            values = sorted(r.reduction for r in rows)
            middle = len(values) // 2
            medians[criterion] = (
                values[middle] if len(values) % 2
                else 0.5 * (values[middle - 1] + values[middle])
            )
        else:
            medians[criterion] = None
    present = [v for v in medians.values() if v is not None]
    spread = max(present) - min(present) if len(present) > 1 else None
    return {
        "median_reduction": medians,
        "readings_per_criterion": counts,
        "spread": spread,
        # A criterion with no reading is unaddressed in the strongest sense:
        # nothing was measured, so nothing is known.
        "unmeasured": [c for c in CRITERIA if counts[c] == 0],
        "unaddressed": [c for c in CRITERIA
                        if medians[c] is not None and medians[c] <= 0.0],
    }
