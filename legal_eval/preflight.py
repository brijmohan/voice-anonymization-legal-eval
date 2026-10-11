"""Checks a reported run has to pass, each one born from a real failure.

This module is the part of the harness that carries what went wrong before.
[`docs/operational-hazards.md`](../docs/operational-hazards.md) explains each
failure in prose; this makes the ones that can be checked mechanically into
checks, so a result that repeats them is caught by running something rather
than by someone remembering.

The reason it exists as code and not as a checklist: every failure recorded
here produced a plausible-looking number, not an error. A standard error of
exactly zero reads as precision. A metric pinned against its ceiling reads as
agreement between systems. A population that silently shrank reads as a
different result. None of them raise, and the one that mattered most put a
false finding into a draft before it was caught.

Checks are advisory about intent and strict about arithmetic. Where a check
cannot tell a mistake from a legitimate result it reports ``warn`` and says
what to look at, because a check that fails on good data gets switched off.

Run it over a comparison's JSON output:

    legal-eval preflight results.json

Exit status is 0 when nothing failed, 1 when something did.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Literal

Status = Literal["pass", "fail", "warn", "skip"]

#: Metrics whose value is a probability, with the value an attacker who learned
#: nothing would obtain. Singling Out's baseline is exp(-1) rather than zero:
#: a predicate carrying no information still isolates one of N that often.
PROBABILITY_METRICS = {
    "singling_out": "exp(-1)",
    "linkability": "1/N",
    "srd_idr": "1/N",
    "lid_pdr": "1/N",
}

#: Metrics measured in bits of disclosure, which are zero when the attacker
#: learns nothing and log2(N) when identification is certain.
BIT_METRICS = ("srd_mean", "srd_max", "lid_mean", "lid_max")


@dataclass(frozen=True)
class CheckResult:
    """One check's verdict.

    Attributes:
        name: Stable identifier, usable in a CI filter.
        status: ``pass``, ``fail``, ``warn`` or ``skip``.
        detail: What was found, in terms of the data rather than the code.
        hazard: The failure this check exists to prevent.
    """

    name: str
    status: Status
    detail: str
    hazard: str

    def __str__(self) -> str:
        return f"[{self.status.upper():4}] {self.name}: {self.detail}"


def chance_levels(n: int) -> dict[str, float]:
    """The value each metric takes when the attacker has learned nothing.

    Reporting a metric without this is the single most common way a privacy
    number misleads: 0.37 sounds low until you know that an attacker with no
    information scores 0.37.

    Args:
        n: Size of the candidate population.

    Returns:
        Metric name to its chance level, in the metric's own units.

    Raises:
        ValueError: when ``n`` is below 2, where a candidate set is not a
            choice and no metric is defined.
    """
    if n < 2:
        raise ValueError(f"a candidate population needs at least 2 speakers, got {n}")
    return {
        "singling_out": math.exp(-1),
        "linkability": 1.0 / n,
        "srd_idr": 1.0 / n,
        "lid_pdr": 1.0 / n,
        "srd_mean": 0.0,
        "srd_max": 0.0,
        "lid_mean": 0.0,
        "lid_max": 0.0,
    }


def ceilings(n: int) -> dict[str, float]:
    """The value each metric takes under certain identification.

    A metric at its ceiling cannot distinguish two systems, so a comparison
    drawn from saturated conditions says nothing however clean it looks. Four
    of six published conditions sat here at N=29.

    Args:
        n: Size of the candidate population.

    Returns:
        Metric name to its upper bound.
    """
    bound = math.log2(n)
    out = dict.fromkeys(PROBABILITY_METRICS, 1.0)
    out.update(dict.fromkeys(BIT_METRICS, bound))
    return out


def _rows(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Flatten a report into (system name, one N row) pairs."""
    systems = report.get("systems") or {}
    if isinstance(systems, list):
        systems = {str(i): s for i, s in enumerate(systems)}
    out = []
    for name, body in systems.items():
        if not isinstance(body, dict):
            continue
        for row in body.get("by_n") or []:
            if isinstance(row, dict):
                out.append((str(name), row))
    return out


def check_population_recorded(report: dict[str, Any]) -> CheckResult:
    """Every row must name the population it was computed over.

    CNIL obligation G2 requires risk quantified against a concrete attacker on
    a stated sample. Without N nothing in the row can be compared to its own
    chance level, so the number is not interpretable at all.
    """
    rows = _rows(report)
    if not rows:
        return CheckResult("population_recorded", "skip",
                           "no system rows in this report",
                           "a report with no population is not interpretable")
    missing = [name for name, row in rows if not isinstance(row.get("n"), int)]
    if missing:
        return CheckResult(
            "population_recorded", "fail",
            f"{len(missing)} row(s) have no integer n, e.g. {missing[0]!r}",
            "CNIL G2: risk must be quantified against a stated sample",
        )
    return CheckResult("population_recorded", "pass",
                       f"all {len(rows)} row(s) name their population",
                       "CNIL G2: risk must be quantified against a stated sample")


def check_no_zero_standard_error(report: dict[str, Any]) -> CheckResult:
    """A standard error of exactly zero means the resampling is wrong.

    This is the failure that cost the most. Spread came only from the closed
    set draw, so at N equal to the enrollment count the draw was deterministic
    and every standard error was exactly 0.0000. It read as a precise
    measurement, and produced a published-shaped claim that one metric resolved
    systems the others could not, plus an inversion against the EER. Both
    reversed once trials were resampled too.

    There is a second, innocent way to get a zero. A metric pinned at its
    ceiling in every replicate has no variance to estimate, so its standard
    error is legitimately zero: Linkability for original libri_test is exactly
    1.000 at every N. Running this check over the project's own committed
    results surfaced both kinds at once, and conflating them would send
    somebody to repair resampling that works. So they are separated: a zero at
    the bound is reported as saturation, and only a zero away from the bound is
    the resampling bug.
    """
    rows = _rows(report)
    broken: list[str] = []
    saturated: list[str] = []
    seen = 0
    for name, row in rows:
        n = row.get("n")
        bounds = ceilings(n) if isinstance(n, int) and n >= 2 else {}
        for key, value in row.items():
            if not key.endswith("_sem"):
                continue
            seen += 1
            if not isinstance(value, (int, float)) or value != 0.0:
                continue
            metric = key[: -len("_sem")]
            observed = row.get(metric)
            bound = bounds.get(metric)
            at_bound = (
                isinstance(observed, (int, float))
                and bound is not None
                and math.isclose(observed, bound, rel_tol=1e-9, abs_tol=1e-9)
            )
            where = f"{name} n={n} {key}"
            (saturated if at_bound else broken).append(where)

    hazard = "a standard error of exactly zero is a bug, not precision"
    if not seen:
        return CheckResult("no_zero_standard_error", "skip",
                           "this report carries no standard errors", hazard)
    if broken:
        return CheckResult(
            "no_zero_standard_error", "fail",
            f"{len(broken)} standard error(s) are exactly zero away from the "
            f"metric's bound, e.g. {broken[0]}. Resample trials, not only the "
            f"candidate draw.",
            hazard,
        )
    if saturated:
        return CheckResult(
            "no_zero_standard_error", "warn",
            f"{len(saturated)} standard error(s) are zero because the metric is "
            f"pinned at its bound in every replicate, e.g. {saturated[0]}. That "
            f"is not a resampling fault, but the condition cannot separate "
            f"systems: see saturation_declared.",
            hazard,
        )
    return CheckResult("no_zero_standard_error", "pass",
                       f"all {seen} standard error(s) are non-zero", hazard)


def check_saturation_declared(report: dict[str, Any]) -> CheckResult:
    """A condition at a metric's bound must be declared as such.

    A saturated metric cannot invert and cannot separate systems, so a
    comparison that looks consistent may only be reporting that everything is
    pinned. Any claim of agreement between metrics drawn from saturated
    conditions is weak evidence, and has to say so.
    """
    rows = _rows(report)
    if not rows:
        return CheckResult("saturation_declared", "skip", "no system rows",
                           "a saturated metric cannot separate systems")
    pinned = []
    for name, row in rows:
        n = row.get("n")
        if not isinstance(n, int) or n < 2:
            continue
        bounds = ceilings(n)
        for metric, bound in bounds.items():
            value = row.get(metric)
            if isinstance(value, (int, float)) and math.isclose(
                value, bound, rel_tol=1e-9, abs_tol=1e-9
            ):
                pinned.append(f"{name} n={n} {metric}={value:g}")
    if not pinned:
        return CheckResult("saturation_declared", "pass",
                           "no reported value sits on a metric's bound",
                           "a saturated metric cannot separate systems")
    if report.get("saturation"):
        return CheckResult(
            "saturation_declared", "pass",
            f"{len(pinned)} value(s) are at their bound and the report carries "
            f"a saturation section",
            "a saturated metric cannot separate systems",
        )
    return CheckResult(
        "saturation_declared", "fail",
        f"{len(pinned)} value(s) sit exactly on a metric's bound with no "
        f"saturation section in the report, e.g. {pinned[0]}",
        "a saturated metric cannot separate systems",
    )


def check_srd_idr_matches_linkability(report: dict[str, Any]) -> CheckResult:
    """SRD's identification rate is the Linkability metric.

    Not approximately: the two definitions are the same closed-set rank-1
    quantity, and on real embeddings they agreed to 0.0000. So a run where they
    differ materially has an implementation error in one of them, and the two
    cannot be reported as independent corroboration of each other.
    """
    rows = _rows(report)
    worst, where = 0.0, None
    compared = 0
    for name, row in rows:
        a, b = row.get("srd_idr"), row.get("linkability")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        compared += 1
        d = abs(a - b)
        if d > worst:
            worst, where = d, f"{name} n={row.get('n')}: {a:.4f} vs {b:.4f}"
    if not compared:
        return CheckResult("srd_idr_matches_linkability", "skip",
                           "the report does not carry both metrics",
                           "SRD's IdR and Linkability are the same quantity")
    hazard = "SRD's IdR and Linkability are the same quantity"
    # Both are rank-1 statistics over independently drawn candidate sets, so at
    # small N they separate by sampling noise even when both are right: 0.0345
    # was the worst seen on real embeddings and 0.0613 on the synthetic demo at
    # n=5. The failing threshold is set where the gap cannot be noise, because
    # a check that fails on legitimate data is a check that gets disabled. At N
    # equal to the enrollment count the draw is deterministic and the two agree
    # exactly, which is the strong form of this check.
    if worst > 0.15:
        return CheckResult("srd_idr_matches_linkability", "fail",
                           f"they differ by up to {worst:.4f} ({where}), which "
                           f"is too large to be sampling noise. One of the two "
                           f"implementations is wrong.", hazard)
    if worst > 0.0:
        return CheckResult("srd_idr_matches_linkability", "warn",
                           f"they differ by up to {worst:.4f} ({where}), which "
                           f"is sampling noise at small N but must not be "
                           f"reported as two independent metrics agreeing",
                           hazard)
    return CheckResult("srd_idr_matches_linkability", "pass",
                       f"identical across all {compared} row(s)", hazard)


def check_lid_calibration_recorded(report: dict[str, Any]) -> CheckResult:
    """A LID figure without its calibration is not a property of the system.

    Only the fitted slope survives the softmax, so LID measures the system and
    its calibration together. On real data one condition read 1.561 bits under
    its own calibration and -207.3 bits under another's, against a 4.858
    ceiling. A reported LID with no slope beside it cannot be reproduced or
    compared.
    """
    rows = _rows(report)
    has_lid = any(
        isinstance(row.get(m), (int, float))
        for _, row in rows for m in ("lid_mean", "lid_max", "lid_pdr")
    )
    hazard = "LID is a joint property of the system and its calibration"
    if not has_lid:
        return CheckResult("lid_calibration_recorded", "skip",
                           "no LID values in this report", hazard)
    slopes = report.get("fitted_slopes") or {}
    if not slopes:
        return CheckResult("lid_calibration_recorded", "fail",
                           "LID is reported with no fitted_slopes section, so "
                           "the calibration behind it is unrecorded", hazard)
    separable = report.get("separable") or {}
    degenerate = [k for k, v in separable.items() if v]
    if degenerate:
        return CheckResult(
            "lid_calibration_recorded", "warn",
            f"calibration is on separable scores for {len(degenerate)} "
            f"condition(s), e.g. {degenerate[0]!r}: the slope has no finite "
            f"maximum likelihood estimate and LID collapses to log2(N)",
            hazard,
        )
    return CheckResult("lid_calibration_recorded", "pass",
                       f"a fitted slope is recorded for {len(slopes)} "
                       f"condition(s)", hazard)


def check_metric_directions(report: dict[str, Any]) -> CheckResult:
    """Probability metrics fall with N and bit metrics rise with it.

    Chance is 1/N for the rank metrics, so a larger population makes the task
    harder and the probability lower, while the bit ceiling log2(N) grows. A
    report violating this is usually comparing values computed at different N
    as though they were comparable, which is how a coarsening gets read as an
    inversion.
    """
    hazard = "probability and bit metrics move in opposite directions with N"
    systems = report.get("systems") or {}
    if isinstance(systems, list):
        systems = {str(i): s for i, s in enumerate(systems)}
    violations = []
    checked = 0
    for name, body in systems.items():
        if not isinstance(body, dict):
            continue
        rows = [r for r in (body.get("by_n") or [])
                if isinstance(r, dict) and isinstance(r.get("n"), int)]
        rows.sort(key=lambda r: r["n"])
        if len(rows) < 2:
            continue
        for metric in ("linkability", "srd_idr"):
            series = [(r["n"], r[metric]) for r in rows
                      if isinstance(r.get(metric), (int, float))]
            if len(series) < 2:
                continue
            checked += 1
            if series[-1][1] > series[0][1] + 1e-9:
                violations.append(
                    f"{name} {metric} rises from {series[0][1]:.4f} at n="
                    f"{series[0][0]} to {series[-1][1]:.4f} at n={series[-1][0]}"
                )
        for metric in ("srd_mean", "lid_mean"):
            series = [(r["n"], r[metric]) for r in rows
                      if isinstance(r.get(metric), (int, float))]
            if len(series) < 2:
                continue
            checked += 1
            if series[-1][1] < series[0][1] - 1e-9:
                violations.append(
                    f"{name} {metric} falls from {series[0][1]:.4f} at n="
                    f"{series[0][0]} to {series[-1][1]:.4f} at n={series[-1][0]}"
                )
    if not checked:
        return CheckResult("metric_directions", "skip",
                           "not enough populations to see a trend", hazard)
    if violations:
        return CheckResult("metric_directions", "warn",
                           f"{len(violations)} series run against the expected "
                           f"direction, e.g. {violations[0]}", hazard)
    return CheckResult("metric_directions", "pass",
                       f"all {checked} series run in the expected direction",
                       hazard)


def check_exclusions_declared(report: dict[str, Any]) -> CheckResult:
    """Utterances the attacker could not score must be declared.

    A destroyed utterance scores at chance, and chance is the best attainable
    privacy score, so dropping one quietly makes the system look better. The
    McAdams baseline on Common Voice emitted a file of the right length that
    was 99.93% digital silence; the attacker raised on it, which is the visible
    outcome. A more forgiving extractor would have embedded the silence and
    counted it as a privacy success.

    So a report either states that nothing was excluded or says how many were
    and why. Silence on the question is the one answer that cannot be checked.
    """
    hazard = ("a destroyed utterance scores at chance, so excluding one "
              "quietly flatters the system")
    if "excluded" not in report:
        return CheckResult(
            "exclusions_declared", "fail",
            "the report does not say whether any utterance was excluded. State "
            "excluded: 0 when none were, so the question has an answer.",
            hazard,
        )
    excluded = report["excluded"]
    if isinstance(excluded, int):
        count, detail = excluded, None
    elif isinstance(excluded, dict):
        count = excluded.get("count")
        detail = excluded.get("reason") or excluded.get("ids")
    elif isinstance(excluded, list):
        count, detail = len(excluded), excluded
    else:
        return CheckResult("exclusions_declared", "fail",
                           f"excluded should be a count, a list or an object, "
                           f"got {type(excluded).__name__}", hazard)
    if not isinstance(count, int) or count < 0:
        return CheckResult("exclusions_declared", "fail",
                           f"excluded count is not a non-negative integer: "
                           f"{count!r}", hazard)
    if count == 0:
        return CheckResult("exclusions_declared", "pass",
                           "the report states that nothing was excluded", hazard)
    if not detail:
        return CheckResult(
            "exclusions_declared", "fail",
            f"{count} utterance(s) were excluded with no reason or ids "
            f"recorded, so the exclusion cannot be audited", hazard,
        )
    return CheckResult(
        "exclusions_declared", "warn",
        f"{count} utterance(s) were excluded and recorded. The reported "
        f"privacy is an upper bound on what the system achieves on audio it "
        f"processes correctly.", hazard,
    )


#: Every check, in the order they are reported.
CHECKS = (
    check_population_recorded,
    check_exclusions_declared,
    check_no_zero_standard_error,
    check_saturation_declared,
    check_srd_idr_matches_linkability,
    check_lid_calibration_recorded,
    check_metric_directions,
)


def run_preflight(report: dict[str, Any]) -> list[CheckResult]:
    """Run every check over a parsed comparison report.

    Args:
        report: The JSON a comparison run wrote, already parsed.

    Returns:
        One :class:`CheckResult` per check, in a stable order.
    """
    return [check(report) for check in CHECKS]


def failed(results: list[CheckResult]) -> list[CheckResult]:
    """The results that are outright failures, not warnings or skips."""
    return [r for r in results if r.status == "fail"]


def unmarked_claims(text: str) -> list[tuple[int, str]]:
    """Lines that read as claims but carry no evidence marker.

    The constitution's rule is that every claim carries [M] measured, [P]
    published, [I] inferred or [?] open, and that an unmarked claim is a
    mistake. This finds the lines that state a measured quantity without
    saying where it came from.

    The unit is the **paragraph**, not the line. A first version checked lines,
    and running it over this project's own documentation produced 18
    candidates of which nearly all were one sentence wrapped across two lines
    with its marker on the second. A marker applies to the claim it is part of,
    wherever the text happens to wrap.

    The rule is otherwise deliberately narrow: an assertion has to be followed
    closely by a number. Without that, "one GPU gives one worker" and "a
    preservation commitment measured in decades" both matched.

    **This one is advisory and is not in** :data:`CHECKS`. It cannot tell a
    result from a threshold a test chose, or from a cross-reference that
    happens to contain a numeral, and over this project's own documentation it
    leaves two false positives out of roughly 100 paragraphs that mention a
    number. That is useful as a reading aid and not sound enough to gate on,
    so it does not gate. The tuning history is worth keeping: line-based gave
    18 candidates dominated by wrapped sentences, paragraph-based with a loose
    rule gave 22 dominated by polysemous verbs, and the current rule gives 2.

    Args:
        text: Markdown source.

    Returns:
        ``(line number, first line)`` for each unmarked candidate paragraph,
        1-indexed on the paragraph's first line.
    """
    markers = ("[M]", "[P]", "[I]", "[?]")
    # The assertion has to be followed by a number for the paragraph to be a
    # candidate. Without that, "one GPU gives one worker", "a preservation
    # commitment measured in decades" and "--speaker-counts linear gives the
    # paper's grid" all matched, and a check with that many false alarms is one
    # nobody runs. "measured on" and "measured in" are excluded outright: they
    # introduce provenance or an idiom, never a quantity.
    claim = re.compile(
        r"\b(?:measured|we found|gives|reads|yields|scores|reaches|equals)\b"
        r"(?!\s+(?:on|in|against)\b)"
        r"(?:\s+[A-Za-z,'`\-]+){0,5}\s+[~<>=]?\s*[0-9]",
        re.IGNORECASE,
    )

    # Gather paragraphs, skipping fenced code, tables, quotes and headings.
    paragraphs: list[tuple[int, list[str]]] = []
    current: list[str] = []
    start = 0
    in_code = False
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            if current:
                paragraphs.append((start, current))
                current = []
            continue
        skip = in_code or stripped.startswith(("|", ">", "#"))
        if not stripped or skip:
            if current:
                paragraphs.append((start, current))
                current = []
            continue
        if not current:
            start = number
        current.append(stripped)
    if current:
        paragraphs.append((start, current))

    out = []
    for first_line, lines in paragraphs:
        body = " ".join(lines)
        if any(m in body for m in markers):
            continue
        if claim.search(body):
            out.append((first_line, lines[0]))
    return out
