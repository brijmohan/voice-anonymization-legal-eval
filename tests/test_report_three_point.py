"""Tests for the three-point report.

The report exists because a privacy number read against its chance level alone
cannot be told from a number read against what the attacker could actually
achieve. Both failure modes it names are measured: at N=29 on Common Voice the
original is pinned at 0.9131 so nothing can be distinguished, and at N=9,779 it
is 0.4515 so the attacker already fails on most unprotected audio.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "report_three_point",
    Path(__file__).resolve().parents[1] / "scripts" / "report_three_point.py",
)
rep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rep)


def write(tmp_path, systems):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"systems": systems}), encoding="utf-8")
    return path


def test_chance_levels_are_the_published_baselines():
    assert rep.CHANCE["singling_out"](9779) == math.exp(-1)
    assert rep.CHANCE["linkability"](9779) == 1 / 9779
    assert rep.CHANCE["srd_mean"](9779) == 0.0


def test_room_and_closed_use_the_original_not_the_bound(tmp_path, capsys):
    """The point of the report: the denominator is what was achievable."""
    path = write(tmp_path, {
        "orig": {"by_n": [{"n": 250, "linkability": 0.7725}]},
        "anon": {"by_n": [{"n": 250, "linkability": 0.6000}]},
    })
    assert rep.main([str(path), "--original", "orig"]) == 0
    out = capsys.readouterr().out
    # room = 0.7725 - 1/250 = 0.7685; closed = (0.7725-0.6)/0.7685 = 0.2245
    assert "0.7685" in out
    assert "0.224" in out or "0.225" in out


def test_a_pinned_original_is_flagged(tmp_path, capsys):
    """N=29 on Common Voice: the original sits at 0.9131."""
    path = write(tmp_path, {
        "orig": {"by_n": [{"n": 29, "linkability": 0.9131}]},
        "anon": {"by_n": [{"n": 29, "linkability": 0.8431}]},
    })
    rep.main([str(path), "--original", "orig"])
    assert "pinned near bound" in capsys.readouterr().out


def test_a_failing_original_is_flagged_as_confounded(tmp_path, capsys):
    """N=9,779 on Common Voice: the original is 0.4515."""
    path = write(tmp_path, {
        "orig": {"by_n": [{"n": 9779, "linkability": 0.4515}]},
        "anon": {"by_n": [{"n": 9779, "linkability": 0.3000}]},
    })
    rep.main([str(path), "--original", "orig"])
    assert "confounded with difficulty" in capsys.readouterr().out


def test_a_usable_population_gets_no_note(tmp_path, capsys):
    path = write(tmp_path, {
        "orig": {"by_n": [{"n": 100, "linkability": 0.8408}]},
        "anon": {"by_n": [{"n": 100, "linkability": 0.7000}]},
    })
    rep.main([str(path), "--original", "orig"])
    out = capsys.readouterr().out
    assert "pinned" not in out
    assert "confounded" not in out


def test_an_original_at_chance_leaves_no_room(tmp_path, capsys):
    """Dividing by zero room would invent a number; it must refuse instead."""
    path = write(tmp_path, {
        "orig": {"by_n": [{"n": 100, "singling_out": math.exp(-1)}]},
        "anon": {"by_n": [{"n": 100, "singling_out": math.exp(-1)}]},
    })
    rep.main([str(path), "--original", "orig", "--metric", "singling_out"])
    out = capsys.readouterr().out
    assert "n/a" in out
    assert "already at chance" in out


def test_a_missing_original_lists_what_is_available(tmp_path, capsys):
    path = write(tmp_path, {"anon": {"by_n": [{"n": 100, "linkability": 0.7}]}})
    assert rep.main([str(path), "--original", "absent"]) == 2
    assert "anon" in capsys.readouterr().err


def test_a_missing_file_exits_two(tmp_path):
    assert rep.main([str(tmp_path / "nope.json"), "--original", "orig"]) == 2


def test_only_shared_populations_are_compared(tmp_path, capsys):
    """Comparing values computed at different N is how a coarsening is misread."""
    path = write(tmp_path, {
        "orig": {"by_n": [{"n": 100, "linkability": 0.84}]},
        "anon": {"by_n": [{"n": 100, "linkability": 0.70},
                          {"n": 9779, "linkability": 0.30}]},
    })
    rep.main([str(path), "--original", "orig"])
    out = capsys.readouterr().out
    assert "   100 " in out
    assert "9779" not in out
