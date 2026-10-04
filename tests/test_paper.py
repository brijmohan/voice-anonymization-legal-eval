"""Loading the published results behind the paper's figure."""

import pytest

from legal_eval.paper import ATTACKER_KEYS, PublishedCurve, load_paper_results

LINKABILITY_CSV = """\
attacker,L,enrollment_speakers,curve_type,linkability_mean,linkabilitystd,chance
original,1,20,orig,0.8216,0.0022,0.05
original,1,120,orig,0.7001,0.0055,0.008333
INFORMED,1,20,anon,0.7700,0.0030,0.05
SEMI-INFORMED,1,20,anon,0.2100,0.0040,0.05
IGNORANT,1,20,anon,0.1000,0.0050,0.05
original,3,20,orig,0.9500,0.0010,0.05
"""

SINGLING_OUT_CSV = """\
attacker,L,enrollment_speakers,curve_type,singling_out_mean,singling_out_std,chance
original,1,30,orig,0.5690,0.0,0.37
original,1,100,orig,0.5032,0.0,0.37
"""

EER_CSV = """\
attacker,L,enrollment_speakers,curve_type,eer_mean,eer_std,chance
original,1,20,orig,0.1684,0.0221,0.5
"""


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "linkability_results.csv").write_text(LINKABILITY_CSV, encoding="utf-8")
    (tmp_path / "singling_out_results.csv").write_text(SINGLING_OUT_CSV, encoding="utf-8")
    (tmp_path / "eer_results.csv").write_text(EER_CSV, encoding="utf-8")
    return tmp_path


def test_loads_every_metric_nested_by_length_and_attacker(data_dir):
    results = load_paper_results(data_dir=data_dir)
    assert set(results) == {"linkability", "singling_out", "eer"}
    assert sorted(results["linkability"]) == [1, 3]
    assert set(results["linkability"][1]) == {
        "original", "informed", "semi_informed", "ignorant"
    }


def test_attacker_labels_are_normalised(data_dir):
    results = load_paper_results("linkability", data_dir=data_dir)
    curve = results["linkability"][1]["semi_informed"]
    assert curve.attacker == "semi_informed"
    assert curve.means[20] == pytest.approx(0.21)
    # Every label the CSVs use must map to a key this package recognises.
    assert set(ATTACKER_KEYS.values()) == {
        "original", "informed", "semi_informed", "ignorant"
    }


def test_curve_exposes_the_sweep_result_interface(data_dir):
    """Plotting accepts either type, so the three accessors must line up."""
    curve = load_paper_results("linkability", data_dir=data_dir)["linkability"][1]["original"]
    assert curve.speaker_counts == [20, 120]
    assert curve.mean() == {20: pytest.approx(0.8216), 120: pytest.approx(0.7001)}
    assert curve.std() == {20: pytest.approx(0.0022), 120: pytest.approx(0.0055)}
    assert curve.values == {20: [pytest.approx(0.8216)], 120: [pytest.approx(0.7001)]}


def test_chance_levels_are_carried_through(data_dir):
    curve = load_paper_results("linkability", data_dir=data_dir)["linkability"][1]["original"]
    # Linkability chance is 1/N'.
    assert curve.chance[20] == pytest.approx(1 / 20)
    singling = load_paper_results("singling_out", data_dir=data_dir)
    assert singling["singling_out"][1]["original"].chance[30] == pytest.approx(0.37, abs=0.01)


def test_eer_is_loaded_raw_not_inverted(data_dir):
    """Raw EER in memory, matching eer_sweep; the plotting layer inverts once.

    Loading the already-inverted file instead would invert twice and silently
    flip every EER panel upside down.
    """
    curve = load_paper_results("eer", data_dir=data_dir)["eer"][1]["original"]
    assert curve.means[20] == pytest.approx(0.1684)
    assert curve.chance[20] == pytest.approx(0.5)


def test_loading_one_metric_skips_the_others(data_dir):
    assert set(load_paper_results("singling_out", data_dir=data_dir)) == {"singling_out"}


def test_rejects_an_unknown_metric(data_dir):
    with pytest.raises(ValueError, match="unknown metric"):
        load_paper_results("not_a_metric", data_dir=data_dir)


def test_missing_file_points_at_the_docs(tmp_path):
    with pytest.raises(FileNotFoundError, match="docs/reproduction.md"):
        load_paper_results("linkability", data_dir=tmp_path)


def test_published_curve_is_usable_empty():
    curve = PublishedCurve(metric="linkability", attacker="original", conversation_length=1)
    assert curve.speaker_counts == []
    assert curve.mean() == {}


@pytest.mark.skipif(
    not (
        __import__("legal_eval.paper", fromlist=["DEFAULT_DATA_DIR"]).DEFAULT_DATA_DIR
        / "linkability_results.csv"
    ).exists(),
    reason="the shipped paper results are not present in this checkout",
)
def test_shipped_results_cover_the_whole_paper_figure():
    """All nine panels of the paper's figure must be loadable."""
    results = load_paper_results()
    for metric in ("singling_out", "linkability", "eer"):
        for length in (1, 3, 30):
            panel = results[metric][length]
            assert set(panel) == {"original", "informed", "semi_informed", "ignorant"}
            for curve in panel.values():
                assert curve.speaker_counts
                assert all(0.0 <= v <= 1.0 for v in curve.mean().values())
