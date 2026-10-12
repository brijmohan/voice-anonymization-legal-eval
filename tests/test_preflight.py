"""Tests for the hazard checks.

Each check exists because a specific wrong number got as far as a draft or a
launch. So each one is tested against the shape of the data that produced that
wrong number, not against a synthetic edge case.

The checks also have to stay quiet on good data. A check that fails on a
legitimate result gets switched off, and then it protects nothing, so the
passing cases are tested as carefully as the failing ones.
"""

from __future__ import annotations

import math

import pytest

from legal_eval.preflight import (
    CHECKS,
    ceilings,
    chance_levels,
    check_exclusions_declared,
    check_lid_calibration_recorded,
    check_metric_directions,
    check_no_zero_standard_error,
    check_population_recorded,
    check_saturation_declared,
    check_srd_idr_matches_linkability,
    failed,
    run_preflight,
    unmarked_claims,
)


def report(rows, system="B5 ASR-BN", **extra):
    """A report with one system and the given per-N rows."""
    out = {"systems": {system: {"by_n": rows}}}
    out.update(extra)
    return out


# Chance levels and ceilings, which the rest is measured against.

def test_chance_levels_are_the_published_baselines():
    levels = chance_levels(29)
    # An attacker whose predicate carries no information still isolates
    # exactly one of N this often, which is why the baseline is not zero.
    assert levels["singling_out"] == pytest.approx(math.exp(-1))
    assert levels["linkability"] == pytest.approx(1 / 29)
    assert levels["srd_idr"] == pytest.approx(1 / 29)
    # The bit metrics are disclosure above chance, so chance is zero bits.
    assert levels["srd_mean"] == 0.0
    assert levels["lid_mean"] == 0.0


def test_chance_level_refuses_a_population_below_two():
    with pytest.raises(ValueError, match="at least 2"):
        chance_levels(1)


def test_ceilings_are_one_and_log2_n():
    bounds = ceilings(29)
    assert bounds["linkability"] == 1.0
    assert bounds["lid_mean"] == pytest.approx(math.log2(29))
    assert bounds["srd_max"] == pytest.approx(math.log2(29))


# The population must be recorded. CNIL G2.

def test_a_row_without_n_fails():
    result = check_population_recorded(report([{"linkability": 0.5}]))
    assert result.status == "fail"
    assert "no integer n" in result.detail


def test_rows_with_n_pass():
    result = check_population_recorded(report([{"n": 29, "linkability": 0.5}]))
    assert result.status == "pass"


def test_an_empty_report_is_skipped_not_passed():
    """Nothing to check is not the same as nothing wrong."""
    assert check_population_recorded({}).status == "skip"


# The standard error collapse. The costliest failure in the project.

def test_a_zero_standard_error_away_from_the_bound_fails():
    """At N equal to the enrollment count the closed-set draw is deterministic.

    Every standard error came out exactly 0.0000, which read as precision and
    produced a false claim that one metric resolved systems the others could
    not, plus a false inversion against the EER.
    """
    result = check_no_zero_standard_error(
        report([{"n": 29, "linkability": 0.386, "linkability_sem": 0.0}])
    )
    assert result.status == "fail"
    assert "away from the metric's bound" in result.detail
    assert "Resample trials" in result.detail


def test_a_zero_standard_error_at_the_bound_warns_instead():
    """A metric pinned at its ceiling in every replicate has no variance.

    Linkability for original libri_test is exactly 1.000 at every N, so its
    standard error is legitimately zero. Running this check over the project's
    own committed results surfaced both kinds at once, and calling this a
    resampling fault would send somebody to repair code that works.
    """
    result = check_no_zero_standard_error(
        report([{"n": 29, "linkability": 1.0, "linkability_sem": 0.0}])
    )
    assert result.status == "warn"
    assert "pinned at its bound" in result.detail
    assert "saturation_declared" in result.detail


def test_a_real_zero_outranks_a_saturated_one():
    """One genuine resampling fault must not be hidden by saturated rows."""
    result = check_no_zero_standard_error(report([
        {"n": 29, "linkability": 1.0, "linkability_sem": 0.0},
        {"n": 29, "srd_mean": 2.1, "srd_mean_sem": 0.0},
    ]))
    assert result.status == "fail"
    assert "srd_mean_sem" in result.detail


def test_non_zero_standard_errors_pass():
    result = check_no_zero_standard_error(
        report([{"n": 29, "linkability": 0.9, "linkability_sem": 0.012}])
    )
    assert result.status == "pass"


def test_a_report_with_no_standard_errors_is_skipped():
    result = check_no_zero_standard_error(report([{"n": 29, "linkability": 0.9}]))
    assert result.status == "skip"


def test_a_tiny_standard_error_is_not_flagged():
    """Only exactly zero is the bug; small is just a precise estimate."""
    result = check_no_zero_standard_error(
        report([{"n": 29, "linkability_sem": 1e-12}])
    )
    assert result.status == "pass"


# Saturation. Four of six published conditions sat against the bound.

def test_a_value_at_its_ceiling_without_a_saturation_section_fails():
    result = check_saturation_declared(
        report([{"n": 29, "linkability": 1.0}])
    )
    assert result.status == "fail"
    assert "bound" in result.detail


def test_a_value_at_its_ceiling_with_a_saturation_section_passes():
    result = check_saturation_declared(
        report([{"n": 29, "linkability": 1.0}], saturation=[{"condition": "x"}])
    )
    assert result.status == "pass"


def test_values_below_the_ceiling_pass():
    result = check_saturation_declared(
        report([{"n": 29, "linkability": 0.386, "lid_mean": 1.561}])
    )
    assert result.status == "pass"


def test_a_bit_metric_at_log2_n_is_detected():
    """LID collapses to exactly log2(N) when the calibration is separable."""
    result = check_saturation_declared(
        report([{"n": 29, "lid_mean": math.log2(29)}])
    )
    assert result.status == "fail"


# SRD's IdR is the Linkability metric, not a second opinion on it.

def test_identical_idr_and_linkability_pass():
    result = check_srd_idr_matches_linkability(
        report([{"n": 29, "srd_idr": 0.386, "linkability": 0.386}])
    )
    assert result.status == "pass"


@pytest.mark.parametrize("idr,link,why", [
    (0.7933, 0.8278, "0.0345, the worst seen on real embeddings at small N"),
    (0.2667, 0.3280, "0.0613, the worst in the synthetic demo at n=5"),
])
def test_sampling_noise_warns_rather_than_fails(idr, link, why):
    """Both are rank-1 statistics over independent draws.

    At small N they separate even when both are right. An earlier threshold of
    0.05 failed on the project's own demo output, which is how a check becomes
    one that people switch off.
    """
    result = check_srd_idr_matches_linkability(
        report([{"n": 5, "srd_idr": idr, "linkability": link}])
    )
    assert result.status == "warn", why
    assert "independent" in result.detail


def test_a_large_divergence_fails():
    result = check_srd_idr_matches_linkability(
        report([{"n": 29, "srd_idr": 0.90, "linkability": 0.40}])
    )
    assert result.status == "fail"
    assert "implementations is wrong" in result.detail
    assert "too large to be sampling noise" in result.detail


def test_missing_one_of_the_two_is_skipped():
    result = check_srd_idr_matches_linkability(
        report([{"n": 29, "linkability": 0.4}])
    )
    assert result.status == "skip"


# LID without its calibration is not a property of the system.

def test_lid_without_fitted_slopes_fails():
    result = check_lid_calibration_recorded(report([{"n": 29, "lid_mean": 1.561}]))
    assert result.status == "fail"
    assert "unrecorded" in result.detail


def test_lid_with_fitted_slopes_passes():
    result = check_lid_calibration_recorded(
        report([{"n": 29, "lid_mean": 1.561}], fitted_slopes={"B5 ASR-BN": 2.1})
    )
    assert result.status == "pass"


def test_a_separable_calibration_warns():
    """Unregularised logistic on separable scores has no finite MLE.

    The slope diverges, a value of 199.4 was observed, and LID then equals
    log2(N) exactly whatever the system does.
    """
    result = check_lid_calibration_recorded(
        report([{"n": 29, "lid_mean": 4.858}],
               fitted_slopes={"original": 199.4}, separable={"original": True})
    )
    assert result.status == "warn"
    assert "no finite maximum likelihood" in result.detail


def test_no_lid_values_is_skipped():
    result = check_lid_calibration_recorded(report([{"n": 29, "linkability": 0.4}]))
    assert result.status == "skip"


# Direction with N, which is how a coarsening gets misread as an inversion.

def test_expected_directions_pass():
    result = check_metric_directions(report([
        {"n": 2, "linkability": 0.93, "srd_mean": 0.57, "lid_mean": 0.45},
        {"n": 29, "linkability": 0.39, "srd_mean": 2.10, "lid_mean": 1.56},
    ]))
    assert result.status == "pass"


def test_a_rank_metric_rising_with_n_warns():
    result = check_metric_directions(report([
        {"n": 2, "linkability": 0.40},
        {"n": 29, "linkability": 0.95},
    ]))
    assert result.status == "warn"
    assert "against the expected direction" in result.detail


def test_a_bit_metric_falling_with_n_warns():
    result = check_metric_directions(report([
        {"n": 2, "lid_mean": 3.0},
        {"n": 29, "lid_mean": 1.0},
    ]))
    assert result.status == "warn"


def test_a_single_population_cannot_show_a_trend():
    result = check_metric_directions(report([{"n": 29, "linkability": 1.0}]))
    assert result.status == "skip"


# The suite as a whole.

def test_run_preflight_returns_one_result_per_check():
    results = run_preflight(report([{"n": 29, "linkability": 0.4}]))
    assert len(results) == len(CHECKS)
    assert [r.name for r in results] == [
        "population_recorded", "exclusions_declared", "no_zero_standard_error",
        "saturation_declared", "srd_idr_matches_linkability",
        "lid_calibration_recorded", "worst_case_reported",
        "cllr_calibration_reported", "metric_directions",
    ]


def test_failed_selects_only_failures():
    results = run_preflight(report([
        {"n": 29, "linkability": 1.0, "srd_mean": 2.1, "srd_mean_sem": 0.0}
    ]))
    names = {r.name for r in failed(results)}
    assert "no_zero_standard_error" in names
    assert "saturation_declared" in names
    assert all(r.status == "fail" for r in failed(results))


def test_a_clean_report_fails_nothing():
    results = run_preflight(report(
        [
            {"n": 2, "linkability": 0.93, "linkability_sem": 0.01,
             "srd_idr": 0.93, "srd_mean": 0.57, "lid_mean": 0.45},
            {"n": 29, "linkability": 0.39, "linkability_sem": 0.02,
             "srd_idr": 0.39, "srd_mean": 2.10, "lid_mean": 1.56},
        ],
        fitted_slopes={"B5 ASR-BN": 2.1},
        separable={"B5 ASR-BN": False},
        excluded=0,
    ))
    assert failed(results) == []


def test_a_list_of_systems_is_handled_as_well_as_a_mapping():
    """An older report wrote systems as a list; it must still be checkable."""
    results = run_preflight({"systems": [{"by_n": [{"linkability": 0.4}]}]})
    assert any(r.name == "population_recorded" and r.status == "fail"
               for r in results)


def test_every_check_carries_the_hazard_it_guards():
    """A failure nobody understands gets waived, so the reason travels with it."""
    for result in run_preflight(report([{"n": 29, "linkability": 1.0}])):
        assert result.hazard, result.name
        # Long enough to say what goes wrong, not just name a rule.
        assert len(result.hazard.split()) >= 5, result.hazard


def test_result_str_is_readable():
    result = check_population_recorded(report([{"n": 29}]))
    assert str(result).startswith("[PASS]")


# Evidence markers, the constitution's rule.

def test_an_unmarked_measured_claim_is_found():
    text = "The verifier gives an EER of 0.0867 on this population."
    assert unmarked_claims(text) == [
        (1, "The verifier gives an EER of 0.0867 on this population.")
    ]


def test_a_marked_claim_is_not_flagged():
    text = "The verifier gives an EER of 0.0867 on this population **[M]**."
    assert unmarked_claims(text) == []


def test_prose_without_a_number_is_not_a_candidate():
    assert unmarked_claims("This section explains where identity survives.") == []


def test_code_blocks_and_tables_are_skipped():
    text = (
        "```\n"
        "linkability reaches 1.0\n"
        "```\n"
        "| metric | gives 0.5 |\n"
        "> quoted: yields 0.3\n"
        "# heading reaches 10\n"
    )
    assert unmarked_claims(text) == []


def test_each_marker_kind_counts_as_marked():
    for marker in ("[M]", "[P]", "[I]", "[?]"):
        assert unmarked_claims(f"it yields 0.5 {marker}") == []


# Excluded utterances. A destroyed one scores at chance, which is the best
# attainable privacy score, so dropping it quietly flatters the system.

def test_a_report_silent_on_exclusions_fails():
    """Silence is the one answer that cannot be checked."""
    result = check_exclusions_declared(report([{"n": 29}]))
    assert result.status == "fail"
    assert "excluded: 0" in result.detail


def test_declaring_zero_exclusions_passes():
    result = check_exclusions_declared(report([{"n": 29}], excluded=0))
    assert result.status == "pass"


def test_exclusions_with_ids_warn_and_state_the_direction():
    result = check_exclusions_declared(
        report([{"n": 29}], excluded={"count": 1,
                                      "ids": ["common_voice_en_25475281"]})
    )
    assert result.status == "warn"
    assert "upper bound" in result.detail


def test_exclusions_without_a_reason_cannot_be_audited():
    result = check_exclusions_declared(report([{"n": 29}], excluded={"count": 3}))
    assert result.status == "fail"
    assert "cannot be audited" in result.detail


def test_a_list_of_excluded_ids_is_accepted():
    result = check_exclusions_declared(
        report([{"n": 29}], excluded=["common_voice_en_25475281"])
    )
    assert result.status == "warn"
    assert "1 utterance" in result.detail


def test_a_bare_non_zero_count_cannot_be_audited():
    result = check_exclusions_declared(report([{"n": 29}], excluded=7))
    assert result.status == "fail"


def test_a_nonsense_excluded_field_fails():
    result = check_exclusions_declared(report([{"n": 29}], excluded="some"))
    assert result.status == "fail"
    assert "got str" in result.detail


def test_a_marker_applies_across_a_wrapped_sentence():
    """The failure that corrected this tool.

    A line-based version produced 18 candidates over this project's own
    documentation, nearly all of them one sentence wrapped across two lines
    with its marker on the second. A marker applies to the claim it is part
    of, wherever the text happens to wrap.
    """
    text = (
        "The optimiser has no finite maximiser; ours stopped at w = 199.4\n"
        "and ALID then equals log2 N exactly **[M]**.\n"
    )
    assert unmarked_claims(text) == []


def test_separate_paragraphs_are_judged_separately():
    """A marker in one paragraph must not excuse the next."""
    text = (
        "The verifier gives an EER of 0.0867 **[M]**.\n"
        "\n"
        "The population gives a collision rate of 0.9279.\n"
    )
    found = unmarked_claims(text)
    assert len(found) == 1
    assert found[0][0] == 3


def test_the_reported_line_number_is_the_paragraph_start():
    text = (
        "intro with no claim\n"
        "\n"
        "This paragraph reaches 0.95 and wraps\n"
        "onto a second line.\n"
    )
    found = unmarked_claims(text)
    assert len(found) == 1
    assert found[0][0] == 3


def test_a_fence_ends_the_preceding_paragraph():
    """A marker inside a code block must not excuse prose before it."""
    text = (
        "This yields 0.42.\n"
        "```\n"
        "value [M] 0.42\n"
        "```\n"
    )
    found = unmarked_claims(text)
    assert len(found) == 1
    assert found[0][0] == 1


def test_a_table_row_ends_the_preceding_paragraph():
    text = (
        "The numbers below.\n"
        "| metric | value |\n"
        "\n"
        "This reaches 0.9 with no marker.\n"
    )
    found = unmarked_claims(text)
    assert len(found) == 1
    assert found[0][0] == 4


# ZEBRA's second figure. An average can conceal one fully exposed person.

def sys_report(**fields):
    return {"systems": {"s": {"by_n": [{"n": 29}], **fields}}}


def test_a_population_figure_without_the_worst_case_fails():
    """Dropping the worst case drops the half that ZEBRA added."""
    from legal_eval.preflight import check_worst_case_reported

    result = check_worst_case_reported(sys_report(zebra_population=0.25))
    assert result.status == "fail"
    assert "no worst-case individual figure" in result.detail


def test_both_zebra_figures_pass():
    from legal_eval.preflight import check_worst_case_reported

    result = check_worst_case_reported(
        sys_report(zebra_population=0.25, zebra_worst_case=1.5)
    )
    assert result.status == "pass"


def test_no_zebra_at_all_is_skipped():
    from legal_eval.preflight import check_worst_case_reported

    assert check_worst_case_reported(sys_report()).status == "skip"


def test_a_worst_case_of_zero_still_counts_as_reported():
    """Zero is the value for perfect privacy, not a missing field."""
    from legal_eval.preflight import check_worst_case_reported

    result = check_worst_case_reported(
        sys_report(zebra_population=0.0, zebra_worst_case=0.0)
    )
    assert result.status == "pass"


# Cllr actual and minimum, so calibration loss is visible.

def test_cllr_actual_without_minimum_fails():
    from legal_eval.preflight import check_cllr_calibration_reported

    result = check_cllr_calibration_reported(sys_report(cllr_actual=0.9))
    assert result.status == "fail"
    assert "not both" in result.detail


def test_cllr_minimum_without_actual_also_fails():
    """Reporting only the minimum hides that the attacker as configured did worse."""
    from legal_eval.preflight import check_cllr_calibration_reported

    assert check_cllr_calibration_reported(
        sys_report(cllr_min=0.5)
    ).status == "fail"


def test_both_cllr_figures_pass():
    from legal_eval.preflight import check_cllr_calibration_reported

    result = check_cllr_calibration_reported(
        sys_report(cllr_actual=0.9, cllr_min=0.5)
    )
    assert result.status == "pass"


def test_an_actual_cllr_above_one_bit_is_still_valid():
    """Treating raw cosine scores as LLRs is worse than learning nothing."""
    from legal_eval.preflight import check_cllr_calibration_reported

    result = check_cllr_calibration_reported(
        sys_report(cllr_actual=1.0209, cllr_min=0.9918)
    )
    assert result.status == "pass"
