"""Tests for reading privacy and utility together.

The thesis of the module is that the same measurement is a leak or a
requirement depending on what the data is for, and that two systems trading one
axis against the other are not ordered at all. Both are pinned here, because
both are claims a future change could quietly reverse: a default role of
"prohibited" would make the first disappear, and any tie-break rule would make
the second disappear.
"""

from __future__ import annotations

import pytest

from legal_eval.tradeoff import (
    AT_CHANCE,
    INCIDENTAL,
    PROHIBITED,
    REQUIRED,
    AttributeRole,
    SystemPoint,
    UseCase,
    dominates,
    equal_utility_exposure,
    incomparable,
    pareto_frontier,
    report,
    score_attributes,
)

WER = "1 minus word error rate"


def point(name, exposure, utility, metric=WER):
    return SystemPoint(name=name, exposure=exposure, utility=utility,
                       utility_metric=metric)


# Dominance, which is the only ordering this module will assert.

def test_a_system_does_not_dominate_itself():
    a = point("a", 0.3, 0.8)
    assert not dominates(a, a)


def test_better_on_both_axes_dominates():
    assert dominates(point("a", 0.2, 0.9), point("b", 0.4, 0.7))


def test_better_on_one_axis_and_equal_on_the_other_dominates():
    assert dominates(point("a", 0.2, 0.8), point("b", 0.4, 0.8))
    assert dominates(point("a", 0.3, 0.9), point("b", 0.3, 0.7))


def test_a_trade_is_not_a_win():
    """More private and less useful is a trade. Calling it better is the error
    this module exists to prevent."""
    a, b = point("a", 0.2, 0.5), point("b", 0.5, 0.9)
    assert not dominates(a, b)
    assert not dominates(b, a)
    assert incomparable(a, b)


def test_the_original_and_a_destroyed_system_are_incomparable():
    """The two extremes of the plane. A destroyed signal is perfectly private
    and useless; the original is perfectly useful and not private at all. If a
    ranking existed, one of these would win it, and neither should."""
    original = point("original", 0.90, 1.00)
    destroyed = point("destroyed", 0.00, 0.00)
    assert incomparable(original, destroyed)


# The frontier.

def test_a_dominated_system_is_off_the_frontier():
    good = point("good", 0.2, 0.9)
    bad = point("bad", 0.4, 0.7)
    assert pareto_frontier([good, bad]) == [good]


def test_every_point_on_a_trade_off_curve_is_on_the_frontier():
    points = [point(f"s{i}", 0.1 * i, 0.1 * i) for i in range(1, 6)]
    assert len(pareto_frontier(points)) == len(points)


def test_the_frontier_is_ordered_by_exposure():
    points = [point("b", 0.5, 0.9), point("a", 0.1, 0.2)]
    assert [p.name for p in pareto_frontier(points)] == ["a", "b"]


def test_mixing_utility_scales_is_refused():
    """A word error rate and a mean opinion score are not one axis, and
    ranking across them would be arithmetic on unlike units."""
    with pytest.raises(ValueError, match="not comparable"):
        pareto_frontier([point("a", 0.2, 0.9, "WER"),
                         point("b", 0.3, 4.1, "MOS")])


# Interpolating to a required utility, which is what two tables cannot do.

def test_exposure_at_a_measured_utility_is_that_system_s_exposure():
    points = [point("a", 0.1, 0.2), point("b", 0.5, 0.9)]
    assert equal_utility_exposure(points, 0.2) == pytest.approx(0.1)
    assert equal_utility_exposure(points, 0.9) == pytest.approx(0.5)


def test_exposure_between_two_systems_interpolates_linearly():
    points = [point("a", 0.10, 0.20), point("b", 0.50, 0.60)]
    assert equal_utility_exposure(points, 0.40) == pytest.approx(0.30)


def test_a_utility_nobody_achieved_returns_nothing():
    """Extrapolating a frontier invents an operating point nobody measured."""
    points = [point("a", 0.1, 0.2), point("b", 0.5, 0.6)]
    assert equal_utility_exposure(points, 0.95) is None
    assert equal_utility_exposure(points, 0.05) is None


def test_two_systems_at_the_same_utility_give_the_better_exposure():
    points = [point("a", 0.4, 0.5), point("b", 0.2, 0.5)]
    assert equal_utility_exposure(points, 0.5) == pytest.approx(0.2)


# The report, including what it refuses to do.

def test_the_report_sets_aside_only_what_is_beaten_outright():
    points = [point("good", 0.2, 0.9), point("bad", 0.4, 0.7),
              point("private", 0.1, 0.3)]
    result = report(points)
    assert [p.name for p in result.frontier] == ["private", "good"]
    assert result.dominated == [("bad", "good")]


def test_the_report_names_the_trades_rather_than_ordering_them():
    points = [point("private", 0.1, 0.3), point("useful", 0.6, 0.95)]
    result = report(points)
    assert result.trades == [("private", "useful")]
    assert "belongs to the controller" in result.summary()


def test_one_system_is_not_a_trade_off():
    with pytest.raises(ValueError, match="at least two"):
        report([point("only", 0.2, 0.8)])


# Attribute roles. The thesis of the module.

def _use_case(role):
    return UseCase(
        name="test",
        roles=(AttributeRole("emotion", role, justification="for the test"),),
        content_required=False,
    )


@pytest.mark.parametrize("advantage,role,expected", [
    (0.75, REQUIRED, True),      # survives, and the use case needs it
    (0.75, PROHIBITED, False),   # survives, and the use case forbids it
    (0.00, REQUIRED, False),     # removed, and the use case needed it
    (0.00, PROHIBITED, True),    # removed, and the use case forbade it
])
def test_the_same_advantage_passes_or_fails_on_the_role_alone(
    advantage, role, expected
):
    """The module's whole claim, in one test. A system that keeps emotion is
    correct for sentiment analysis and a breach for a de-identified corpus, and
    the number does not change."""
    outcome = score_attributes(_use_case(role), {"emotion": advantage},
                               {"emotion": 0.80})[0]
    assert outcome.met is expected


def test_a_prohibited_attribute_at_the_chance_band_passes():
    outcome = score_attributes(_use_case(PROHIBITED), {"emotion": AT_CHANCE},
                               {"emotion": 0.8})[0]
    assert outcome.met


def test_a_required_attribute_the_original_never_carried_cannot_be_met():
    """Otherwise a system is credited for preserving something that was never
    recoverable, and the use case is unservable by this data either way."""
    outcome = score_attributes(_use_case(REQUIRED), {"emotion": 0.0},
                               {"emotion": 0.0})[0]
    assert not outcome.met
    assert "cannot be served by this data" in outcome.reason


def test_a_system_is_not_credited_for_removing_what_was_not_there():
    """The mirror case: prohibited and absent from the original. It passes,
    but the original advantage is reported so a reader can see why."""
    outcome = score_attributes(_use_case(PROHIBITED), {"emotion": 0.0},
                               {"emotion": 0.01})[0]
    assert outcome.met
    assert outcome.original_advantage == pytest.approx(0.01)


def test_an_undeclared_attribute_is_reported_as_incidental_not_dropped():
    """An attribute nobody asked about is how a surprise arrives."""
    outcomes = score_attributes(_use_case(PROHIBITED),
                                {"emotion": 0.0, "accent": 0.6},
                                {"emotion": 0.8, "accent": 0.7})
    accent = next(o for o in outcomes if o.attribute == "accent")
    assert accent.role == INCIDENTAL
    assert accent.met
    assert "reported rather than judged" in accent.reason


def test_a_declared_attribute_that_was_not_measured_is_refused():
    """A role assigned and never checked is worse than no role."""
    with pytest.raises(ValueError, match="unchecked claim"):
        score_attributes(_use_case(PROHIBITED), {"accent": 0.3},
                         {"accent": 0.5})


def test_a_role_without_a_justification_is_refused():
    """An unjustified role is how a convenient one gets assigned."""
    with pytest.raises(ValueError, match="justification"):
        AttributeRole("emotion", PROHIBITED)
    with pytest.raises(ValueError, match="justification"):
        AttributeRole("emotion", REQUIRED)


def test_an_incidental_role_needs_no_justification():
    assert AttributeRole("emotion", INCIDENTAL).role == INCIDENTAL


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match="role must be one of"):
        AttributeRole("emotion", "maybe", justification="x")


def test_an_attribute_cannot_be_both_required_and_prohibited():
    with pytest.raises(ValueError, match="more than one role"):
        UseCase(
            name="contradictory",
            roles=(AttributeRole("emotion", REQUIRED, justification="needed"),
                   AttributeRole("emotion", PROHIBITED, justification="forbidden")),
            content_required=False,
        )


def test_an_undeclared_attribute_defaults_to_incidental_not_prohibited():
    """A default of prohibited would make every evaluation assume the strictest
    use case, which rewards a system that destroys the signal."""
    case = UseCase(name="empty", roles=(), content_required=False)
    assert case.role_of("anything") == INCIDENTAL


def test_the_chance_band_has_exactly_one_definition():
    """Two thresholds for one judgement would let a system pass one check and
    fail the other on the same measurement. This is an identity check rather
    than an equality one, so a second copy cannot satisfy it."""
    from legal_eval import inference
    assert AT_CHANCE is inference.AT_CHANCE


# The documented example, so the document cannot drift from the code.

def test_the_use_case_document_example_runs_and_still_reports_a_trade():
    """docs/use-cases.md shows this block and states that the two systems are
    incomparable. If a change made them orderable, the document would be
    wrong and nothing else would say so."""
    import re
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[1] / "docs" / "use-cases.md")
    text = doc.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks, "the document no longer carries a python example"

    namespace: dict = {}
    exec(blocks[0], namespace)  # noqa: S102 - our own documentation
    result = report(namespace["points"])
    assert len(result.frontier) == 2
    assert result.trades == [("asrbn", "mcadams")]
    assert result.dominated == []
