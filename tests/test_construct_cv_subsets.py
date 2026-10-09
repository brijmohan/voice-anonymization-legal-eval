"""Constructing paper-style subsets from an arbitrary Common Voice release."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "construct_cv", SCRIPTS / "construct_common_voice_subsets.py"
)
construct = importlib.util.module_from_spec(spec)
sys.modules["construct_cv"] = construct
spec.loader.exec_module(construct)

import build_cv_common as common  # noqa: E402


@pytest.fixture
def corpus(tmp_path):
    """A locale directory where speakers have deliberately different budgets."""
    root = tmp_path / "en"
    (root / "clips").mkdir(parents=True)
    rows, durations = [], []
    # rich: 10 x 60s = 600s, enough for 120s enrollment and 180s test
    # middling: 4 x 60s = 240s, enough for enrollment only
    # poor: 1 x 30s, enough for neither
    for speaker, gender, n, secs in (("rich", "female", 10, 60),
                                     ("middling", "male", 4, 60),
                                     ("poor", "", 1, 30)):
        for i in range(n):
            clip = f"common_voice_en_{speaker}_{i}.mp3"
            (root / "clips" / clip).write_bytes(b"x")
            rows.append({"client_id": speaker, "path": clip, "sentence": "s",
                         "up_votes": "2", "down_votes": "0", "age": "",
                         "gender": gender, "accents": "", "locale": "en", "segment": ""})
            durations.append({"clip": clip, "duration[ms]": str(secs * 1000)})
    with open(root / "validated.tsv", "w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with open(root / "clip_durations.tsv", "w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=["clip", "duration[ms]"], delimiter="\t")
        writer.writeheader()
        writer.writerows(durations)
    return root


def test_durations_are_read_in_seconds(corpus):
    d = common.read_clip_durations(corpus)
    assert len(d) == 15
    assert all(v in (30.0, 60.0) for v in d.values())


def test_missing_duration_table_is_reported_with_the_workaround(tmp_path):
    (tmp_path / "clips").mkdir()
    with pytest.raises(FileNotFoundError, match="duration"):
        common.read_clip_durations(tmp_path)


def test_enrollment_is_filled_before_test():
    utts = [(f"c{i}", 60.0) for i in range(10)]
    enroll, test = construct.assign_utterances(utts, 120, 180, None, None)
    assert len(enroll) == 2, "120s of enrollment is two 60s clips"
    assert len(test) == 3, "180s of test is three more"


def test_enrollment_and_test_never_share_an_utterance():
    """The property the whole evaluation rests on."""
    utts = [(f"c{i}", 60.0) for i in range(10)]
    enroll, test = construct.assign_utterances(utts, 120, 180, None, None)
    assert set(enroll).isdisjoint(set(test))


def test_speaker_with_enough_for_enrollment_only_still_counts():
    utts = [(f"c{i}", 60.0) for i in range(3)]
    enroll, test = construct.assign_utterances(utts, 120, 180, None, None)
    assert enroll, "must still contribute to the enrollment population"
    assert test == [], "but cannot be a test speaker"


def test_speaker_with_too_little_is_dropped_entirely():
    enroll, test = construct.assign_utterances([("c0", 30.0)], 120, 180, None, None)
    assert enroll == [] and test == []


def test_caps_limit_utterances(corpus):
    utts = [(f"c{i}", 60.0) for i in range(20)]
    enroll, test = construct.assign_utterances(utts, 120, 180, max_enroll=2, max_test=2)
    assert len(enroll) <= 2
    assert len(test) <= 2


def test_end_to_end_build(corpus, tmp_path, capsys):
    out = tmp_path / "subsets"
    sys.argv = ["x", "--cv-root", str(corpus), "--output-dir", str(out),
                "--criterion", "duration",
                "--min-enroll-minutes", "2", "--min-test-minutes", "3"]
    assert construct.main() == 0
    captured = capsys.readouterr().out
    assert "utterance overlap between A and B: 0" in captured

    from legal_eval.io import load_spk2utt, load_utt2spk
    a = load_spk2utt(out / "A_enroll" / "spk2utt")
    b = load_spk2utt(out / "B_test" / "spk2utt")
    # rich and middling clear enrollment; only rich clears test as well.
    assert len(a) == 2
    assert len(b) == 1
    assert set(b).issubset(set(a)), "test speakers must be a subset of enrollment"

    a_utts = {u for us in a.values() for u in us}
    b_utts = {u for us in b.values() for u in us}
    assert a_utts.isdisjoint(b_utts)
    assert load_utt2spk(out / "A_enroll" / "utt2spk").keys() == a_utts


def test_end_to_end_with_the_utterance_criterion(corpus, tmp_path, capsys):
    """The default criterion, sized for the fixture's speakers."""
    out = tmp_path / "subsets_utt"
    sys.argv = ["x", "--cv-root", str(corpus), "--output-dir", str(out),
                "--enroll-utterances", "2", "--max-conversation-length", "2"]
    assert construct.main() == 0
    assert "utterance overlap between A and B: 0" in capsys.readouterr().out

    from legal_eval.io import load_spk2utt
    a = load_spk2utt(out / "A_enroll" / "spk2utt")
    b = load_spk2utt(out / "B_test" / "spk2utt")
    # rich (10 utts) and middling (4) clear a budget of 2; poor (1) does not.
    assert len(a) == 2
    # Only rich has 2 + 4 utterances, so only it can be a test speaker.
    assert len(b) == 1
    assert all(len(v) == 2 for v in a.values()), "fixed budget, not a minimum"
    assert all(len(v) == 4 for v in b.values()), "2 x L_max"


def test_speaker_ids_are_pseudonymised(corpus, tmp_path):
    out = tmp_path / "s2"
    sys.argv = ["x", "--cv-root", str(corpus), "--output-dir", str(out),
                "--criterion", "duration"]
    assert construct.main() == 0
    from legal_eval.io import load_spk2utt
    speakers = load_spk2utt(out / "A_enroll" / "spk2utt")
    assert all(s.startswith("spk-") for s in speakers)
    assert not any(s in ("rich", "middling", "poor") for s in speakers)


def test_utterance_criterion_uses_a_fixed_enrollment_budget():
    """Every speaker contributes the same reference strength, not a minimum."""
    utts = [(f"c{i}", 5.0) for i in range(50)]
    enroll, test = construct.assign_by_utterance_count(utts, enroll_budget=10,
                                                       test_utterances=60)
    assert len(enroll) == 10, "exactly the budget, not more"
    assert test == [], "50 utterances cannot also yield 60 test ones"


def test_utterance_criterion_guarantees_2L_for_singling_out():
    """L=30 needs 60 test utterances so test and calibration stay disjoint."""
    utts = [(f"c{i}", 5.0) for i in range(80)]
    enroll, test = construct.assign_by_utterance_count(utts, 10, 2 * 30)
    assert len(enroll) == 10
    assert len(test) == 60
    assert set(enroll).isdisjoint(set(test))


def test_utterance_criterion_keeps_enrollment_only_speakers():
    """They are the distractor population, which is what buys a large N."""
    utts = [(f"c{i}", 5.0) for i in range(12)]
    enroll, test = construct.assign_by_utterance_count(utts, 10, 60)
    assert len(enroll) == 10
    assert test == []


def test_utterance_criterion_drops_speakers_below_the_budget():
    enroll, test = construct.assign_by_utterance_count(
        [("c0", 5.0)], enroll_budget=10, test_utterances=60)
    assert enroll == [] and test == []


def test_duration_criterion_is_still_available_for_paper_replication():
    """The paper's thresholds must stay reachable, since its numbers used them."""
    utts = [(f"c{i}", 60.0) for i in range(10)]
    enroll, test = construct.assign_utterances(utts, 120, 180, None, None)
    assert len(enroll) == 2 and len(test) == 3
