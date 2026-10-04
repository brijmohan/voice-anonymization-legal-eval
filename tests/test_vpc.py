"""Reading VoicePrivacy Challenge evaluation output.

The fixtures here write the exact layout VPC's
``speaker_extraction.py`` produces, including real ``torch.save`` tensors, so
the loader is tested against the format rather than against a mock of it.
"""

import numpy as np
import pytest

from legal_eval.demo import simulate_anonymization, split_corpus, synthetic_corpus
from legal_eval.vpc import (
    VPCEmbeddings,
    evaluate_vpc_dataset,
    find_embedding_dirs,
    load_vpc_embeddings,
)

torch = pytest.importorskip("torch")


def write_vpc_dir(directory, embeddings, identifier2speaker, spk2gender=None):
    """Write one embedding directory exactly as VPC's extractor does."""
    directory.mkdir(parents=True, exist_ok=True)
    identifiers = list(embeddings)
    matrix = np.vstack([embeddings[i] for i in identifiers])
    torch.save(torch.from_numpy(matrix.astype(np.float32)), directory / "speaker_vectors.pt")

    (directory / "id2idx").write_text(
        "".join(f"{i} {n}\n" for n, i in enumerate(identifiers)), encoding="utf-8"
    )
    (directory / "idx2spk").write_text(
        "".join(f"{n} {identifier2speaker[i]}\n" for n, i in enumerate(identifiers)),
        encoding="utf-8",
    )
    speakers = dict.fromkeys(identifier2speaker.values())
    (directory / "spk2gender").write_text(
        "".join(f"{s} {(spk2gender or {}).get(s, 'f')}\n" for s in speakers), encoding="utf-8"
    )


@pytest.fixture
def vpc_tree(tmp_path):
    """A results tree shaped like exp/asv_anon_x/cosine_out/emb_xvect/."""
    spk2utt, embeddings = synthetic_corpus(n_speakers=24, n_utterances=12, seed=0)
    enroll_utts, test_utts = split_corpus(spk2utt, n_first=6)
    root = tmp_path / "cosine_out" / "emb_xvect"

    for name, mapping in (("libri_dev_enrolls_x", enroll_utts), ("libri_dev_trials_mixed_x", test_utts)):
        i2s = {u: s for s, us in mapping.items() for u in us}
        write_vpc_dir(root / name / "utt-level", {u: embeddings[u] for u in i2s}, i2s)

    # Speaker-level enrollment, as the ASV step also writes.
    spk_level = {s: np.mean([embeddings[u] for u in us], axis=0) for s, us in enroll_utts.items()}
    write_vpc_dir(root / "libri_dev_enrolls_x" / "spk-level", spk_level, {s: s for s in spk_level})
    return tmp_path, embeddings, enroll_utts, test_utts


def test_loads_utterance_level_embeddings(vpc_tree):
    tmp_path, embeddings, enroll_utts, _ = vpc_tree
    loaded = load_vpc_embeddings(
        tmp_path / "cosine_out" / "emb_xvect" / "libri_dev_enrolls_x" / "utt-level"
    )
    assert loaded.level == "utt"
    assert loaded.dataset == "libri_dev_enrolls_x"
    assert len(loaded.speakers) == 24
    assert set(loaded.spk2utt) == set(enroll_utts)
    some_utt = next(iter(loaded.embeddings))
    # float32 on disk, so compare at that precision.
    assert np.allclose(loaded.embeddings[some_utt], embeddings[some_utt], atol=1e-6)


def test_loads_speaker_level_embeddings(vpc_tree):
    tmp_path, _, _, _ = vpc_tree
    loaded = load_vpc_embeddings(
        tmp_path / "cosine_out" / "emb_xvect" / "libri_dev_enrolls_x" / "spk-level"
    )
    assert loaded.level == "spk"
    # At speaker level each identifier is its own speaker.
    assert all(i == s for i, s in loaded.identifier2speaker.items())


def test_discovers_every_dataset_and_level(vpc_tree):
    tmp_path, _, _, _ = vpc_tree
    found = find_embedding_dirs(tmp_path / "cosine_out")
    assert set(found) == {"libri_dev_enrolls_x", "libri_dev_trials_mixed_x"}
    assert set(found["libri_dev_enrolls_x"]) == {"utt", "spk"}
    assert set(found["libri_dev_trials_mixed_x"]) == {"utt"}
    # Searching from a parent works too, which is what a user will try.
    assert find_embedding_dirs(tmp_path).keys() == found.keys()


def test_discovery_reports_a_missing_tree(tmp_path):
    with pytest.raises(FileNotFoundError, match="no emb_xvect"):
        find_embedding_dirs(tmp_path)


def test_loader_rejects_a_directory_that_is_not_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not look like"):
        load_vpc_embeddings(tmp_path)


def test_filter_speakers_keeps_only_those_speakers(vpc_tree):
    tmp_path, _, _, _ = vpc_tree
    loaded = load_vpc_embeddings(
        tmp_path / "cosine_out" / "emb_xvect" / "libri_dev_enrolls_x" / "utt-level"
    )
    keep = set(loaded.speakers[:3])
    filtered = loaded.filter_speakers(keep)
    assert set(filtered.speakers) == keep
    assert len(filtered.embeddings) < len(loaded.embeddings)


def test_end_to_end_evaluation_produces_all_three_metrics(vpc_tree):
    tmp_path, _, _, _ = vpc_tree
    root = tmp_path / "cosine_out" / "emb_xvect"
    rows = evaluate_vpc_dataset(
        load_vpc_embeddings(root / "libri_dev_enrolls_x" / "utt-level"),
        load_vpc_embeddings(root / "libri_dev_trials_mixed_x" / "utt-level"),
        conversation_lengths=(1, 3),
        n_runs=2,
        n_folds=2,
    )
    metrics = {r["metric"] for r in rows}
    assert metrics == {"linkability", "eer", "singling_out"}
    assert {r["L"] for r in rows} == {1, 3}
    assert all(0.0 <= r["value"] <= 1.0 for r in rows)


def test_distractors_enlarge_the_population(vpc_tree):
    """train-clean-360 speakers should extend the sweep beyond the eval set."""
    tmp_path, embeddings, _, _ = vpc_tree
    root = tmp_path / "cosine_out" / "emb_xvect"
    enroll = load_vpc_embeddings(root / "libri_dev_enrolls_x" / "utt-level")
    test = load_vpc_embeddings(root / "libri_dev_trials_mixed_x" / "utt-level")

    extra_spk2utt, extra_emb = synthetic_corpus(
        n_speakers=40, n_utterances=4, seed=99, prefix="distractor"
    )
    distractors = VPCEmbeddings(
        embeddings=extra_emb,
        identifier2speaker={u: s for s, us in extra_spk2utt.items() for u in us},
        spk2gender={},
        dataset="train-clean-360_x",
        level="utt",
    )

    without = evaluate_vpc_dataset(enroll, test, conversation_lengths=(1,), n_runs=2, n_folds=1)
    with_extra = evaluate_vpc_dataset(
        enroll, test, conversation_lengths=(1,), n_runs=2, n_folds=1,
        distractor_utt=distractors,
    )
    largest_without = max(r["speakers"] for r in without if r["metric"] == "linkability")
    largest_with = max(r["speakers"] for r in with_extra if r["metric"] == "linkability")
    assert largest_with > largest_without


def test_anonymization_lowers_the_metrics_end_to_end(tmp_path):
    """The whole path, from VPC-format files to a metric that responds."""
    spk2utt, original = synthetic_corpus(n_speakers=30, n_utterances=10, seed=3)
    enroll_utts, test_utts = split_corpus(spk2utt, n_first=5)

    def linkability_at_full_population(embeddings):
        root = tmp_path / str(id(embeddings)) / "emb_xvect"
        for name, mapping in (("e", enroll_utts), ("t", test_utts)):
            i2s = {u: s for s, us in mapping.items() for u in us}
            write_vpc_dir(root / name / "utt-level", {u: embeddings[u] for u in i2s}, i2s)
        rows = evaluate_vpc_dataset(
            load_vpc_embeddings(root / "e" / "utt-level"),
            load_vpc_embeddings(root / "t" / "utt-level"),
            conversation_lengths=(1,), speaker_counts=[30], n_runs=3, n_folds=1,
        )
        return next(r["value"] for r in rows if r["metric"] == "linkability")

    assert linkability_at_full_population(
        simulate_anonymization(original, strength=0.95, seed=4)
    ) < linkability_at_full_population(original)


def test_mismatched_datasets_are_reported(vpc_tree):
    tmp_path, _, _, _ = vpc_tree
    root = tmp_path / "cosine_out" / "emb_xvect"
    enroll = load_vpc_embeddings(root / "libri_dev_enrolls_x" / "utt-level")
    other = enroll.filter_speakers(set(enroll.speakers[:2]))
    other.identifier2speaker = dict.fromkeys(other.identifier2speaker, "nobody")
    with pytest.raises(ValueError, match="no speaker appears in both"):
        evaluate_vpc_dataset(enroll, other)
