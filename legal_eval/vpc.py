"""Reading VoicePrivacy Challenge evaluation output.

The VPC 2026 recipe already extracts and caches speaker embeddings while
computing the EER, so the legal metrics can be a pure post-processing pass over
what is on disk: no model, no GPU, no second pass over audio. That is the whole
argument for adding them to the challenge, so the loader here deliberately reads
the cache rather than re-running extraction.

Layout written by ``evaluation/privacy/asv/speaker_extraction.py``::

    <results_dir>/emb_xvect/<dataset_name>/utt-level/
        speaker_vectors.pt    torch tensor, one row per identifier
        id2idx                identifier -> row index
        idx2spk               row index -> speaker id
        spk2gender            speaker id -> f or m
    <results_dir>/emb_xvect/<dataset_name>/spk-level/
        ... the same, with speaker ids as identifiers

``results_dir`` is typically ``exp/asv_anon<suffix>/cosine_out`` and
``dataset_name`` the data directory's name, for example
``libri_dev_enrolls_mcadams``.

Reading the tensors needs PyTorch, which this package does not otherwise
require. It is always present in a VPC environment; elsewhere install the
``vpc`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _read_kaldi_mapping(path: Path) -> dict[str, str]:
    """Read a two-column whitespace-separated file into a dict."""
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split(maxsplit=1)
            if len(fields) == 2:
                mapping[fields[0]] = fields[1].strip()
    return mapping


@dataclass
class VPCEmbeddings:
    """Embeddings for one VPC dataset at one level.

    Attributes:
        embeddings: Mapping from identifier to its embedding. Identifiers are
            utterance ids at utterance level and speaker ids at speaker level.
        identifier2speaker: Mapping from identifier to its speaker.
        spk2gender: Mapping from speaker id to ``f`` or ``m``.
        dataset: The VPC dataset name this came from.
        level: ``"utt"`` or ``"spk"``.
    """

    embeddings: dict[str, np.ndarray]
    identifier2speaker: dict[str, str]
    spk2gender: dict[str, str]
    dataset: str
    level: str

    @property
    def spk2utt(self) -> dict[str, list[str]]:
        """Group identifiers by speaker, in insertion order."""
        grouped: dict[str, list[str]] = {}
        for identifier, speaker in self.identifier2speaker.items():
            grouped.setdefault(speaker, []).append(identifier)
        return grouped

    @property
    def speakers(self) -> list[str]:
        """The distinct speakers present, in first-appearance order."""
        return list(dict.fromkeys(self.identifier2speaker.values()))

    def filter_speakers(self, speakers: set[str]) -> VPCEmbeddings:
        """Return a copy restricted to ``speakers``."""
        keep = {i for i, s in self.identifier2speaker.items() if s in speakers}
        return VPCEmbeddings(
            embeddings={i: v for i, v in self.embeddings.items() if i in keep},
            identifier2speaker={i: s for i, s in self.identifier2speaker.items() if i in keep},
            spk2gender={s: g for s, g in self.spk2gender.items() if s in speakers},
            dataset=self.dataset,
            level=self.level,
        )


def load_vpc_embeddings(directory: str | Path) -> VPCEmbeddings:
    """Load one ``utt-level`` or ``spk-level`` directory.

    Args:
        directory: Path ending in ``utt-level`` or ``spk-level``.

    Returns:
        The :class:`VPCEmbeddings`.

    Raises:
        FileNotFoundError: If the expected files are absent.
        ImportError: If PyTorch is unavailable.
    """
    directory = Path(directory)
    vectors_path = directory / "speaker_vectors.pt"
    id2idx_path = directory / "id2idx"
    if not vectors_path.exists() or not id2idx_path.exists():
        raise FileNotFoundError(
            f"{directory} does not look like a VPC embedding directory: expected "
            "speaker_vectors.pt and id2idx. Point at a <dataset>/utt-level or "
            "<dataset>/spk-level directory under <results_dir>/emb_xvect/."
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Reading VPC embeddings needs PyTorch, which is always present in a "
            "VoicePrivacy Challenge environment. Elsewhere: pip install "
            "'voice-anonymization-legal-eval[vpc]'."
        ) from exc

    # weights_only is the safe default, and these files are plain tensors.
    vectors = torch.load(vectors_path, map_location="cpu", weights_only=True)
    array = np.asarray(vectors.detach().cpu().numpy(), dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D tensor in {vectors_path}, got shape {array.shape}")

    id2idx = {k: int(v) for k, v in _read_kaldi_mapping(id2idx_path).items()}
    idx2spk = _read_kaldi_mapping(directory / "idx2spk") if (directory / "idx2spk").exists() else {}
    gender_path = directory / "spk2gender"
    spk2gender = _read_kaldi_mapping(gender_path) if gender_path.exists() else {}

    embeddings: dict[str, np.ndarray] = {}
    identifier2speaker: dict[str, str] = {}
    for identifier, index in id2idx.items():
        if not 0 <= index < array.shape[0]:
            raise ValueError(
                f"{id2idx_path} maps {identifier!r} to row {index}, outside the "
                f"{array.shape[0]} rows of {vectors_path}"
            )
        embeddings[identifier] = array[index]
        # At speaker level the identifier is itself the speaker.
        identifier2speaker[identifier] = idx2spk.get(str(index), identifier)

    return VPCEmbeddings(
        embeddings=embeddings,
        identifier2speaker=identifier2speaker,
        spk2gender=spk2gender,
        dataset=directory.parent.name,
        level="spk" if directory.name == "spk-level" else "utt",
    )


def find_embedding_dirs(results_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Discover every cached embedding directory under a VPC results dir.

    Args:
        results_dir: Either the ``emb_xvect`` directory or a parent containing
            it, for example ``exp/asv_anon_mcadams/cosine_out``.

    Returns:
        Mapping from dataset name to ``{"utt": path, "spk": path}``, with only
        the levels that exist.
    """
    results_dir = Path(results_dir)
    root = results_dir if results_dir.name == "emb_xvect" else results_dir / "emb_xvect"
    if not root.is_dir():
        found = next(results_dir.rglob("emb_xvect"), None) if results_dir.is_dir() else None
        if found is None:
            raise FileNotFoundError(f"no emb_xvect directory under {results_dir}")
        root = found

    discovered: dict[str, dict[str, Path]] = {}
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        levels = {}
        for level, name in (("utt", "utt-level"), ("spk", "spk-level")):
            candidate = dataset_dir / name
            if (candidate / "speaker_vectors.pt").exists():
                levels[level] = candidate
        if levels:
            discovered[dataset_dir.name] = levels
    return discovered


def evaluate_vpc_dataset(
    enroll_utt: VPCEmbeddings,
    test_utt: VPCEmbeddings,
    conversation_lengths: tuple[int, ...] = (1, 3),
    speaker_counts: list[int] | None = None,
    distractor_utt: VPCEmbeddings | None = None,
    n_runs: int = 5,
    n_folds: int = 5,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Compute all three metrics for one VPC evaluation dataset.

    Linkability and the EER use the enrollment speakers as the population the
    attacker searches. VPC's LibriSpeech dev and test sets hold around 40
    speakers each, which is small next to the 22,024 of the paper, so
    ``distractor_utt`` can supply extra enrollment speakers (anonymized
    ``train-clean-360`` is the natural choice, and the recipe has already
    produced its embeddings for ASV training). Distractors enlarge the
    population without ever being the correct answer, which is exactly their
    role in the metric.

    Singling Out uses the test utterances, since it needs several disjoint
    conversations per speaker.

    Args:
        enroll_utt: Utterance-level embeddings of the enrollment set.
        test_utt: Utterance-level embeddings of the trial set.
        conversation_lengths: Values of ``L`` to evaluate.
        speaker_counts: Population sizes. Defaults to a geometric grid up to the
            number of available enrollment speakers.
        distractor_utt: Extra speakers to enlarge the enrollment population.
        n_runs: Random draws per point.
        n_folds: Folds for Singling Out.
        seed: Base seed.

    Returns:
        One row per (metric, L, population size), ready for a DataFrame.
    """
    from legal_eval.embeddings import build_speaker_embeddings, build_test_embeddings
    from legal_eval.scoring import cosine_score_matrix
    from legal_eval.sweeps import (
        eer_sweep,
        geometric_speaker_counts,
        linkability_sweep,
        singling_out_sweep,
    )

    shared = sorted(set(enroll_utt.speakers) & set(test_utt.speakers))
    if not shared:
        raise ValueError(
            "no speaker appears in both the enrollment and trial sets, so "
            "linkage is undefined. Are these the matching VPC directories?"
        )

    enroll_spk2utt = {s: u for s, u in enroll_utt.spk2utt.items() if s in shared}
    enroll_vectors = dict(enroll_utt.embeddings)
    if distractor_utt is not None:
        extra = {s: u for s, u in distractor_utt.spk2utt.items() if s not in shared}
        enroll_spk2utt.update(extra)
        enroll_vectors.update(distractor_utt.embeddings)

    enrollment = build_speaker_embeddings(enroll_spk2utt, enroll_vectors)
    counts = speaker_counts or geometric_speaker_counts(len(enrollment), start=2, base=10)
    test_spk2utt = {s: u for s, u in test_utt.spk2utt.items() if s in shared}

    rows: list[dict[str, object]] = []
    for length in conversation_lengths:
        eligible = {s: u for s, u in test_spk2utt.items() if len(u) >= length}
        if not eligible:
            continue
        test_embeddings, _ = build_test_embeddings(
            eligible, test_utt.embeddings, length, selection="first"
        )
        matrix = cosine_score_matrix(
            enrollment,
            test_embeddings,
            metadata={"conversation_length": length, "dataset": test_utt.dataset},
        )

        for result in (
            linkability_sweep(matrix, speaker_counts=counts, n_runs=n_runs, seed=seed,
                              conversation_length=length),
            eer_sweep(matrix, speaker_counts=counts, n_runs=n_runs, seed=seed,
                      conversation_length=length),
        ):
            means, stds = result.mean(), result.std()
            for count in result.speaker_counts:
                rows.append({
                    "dataset": test_utt.dataset, "metric": result.metric,
                    "L": length, "speakers": count,
                    "value": means[count], "std": stds[count],
                })

        # Singling Out needs 2L utterances per speaker for disjoint
        # test and calibration conversations.
        so_eligible = {s: u for s, u in test_spk2utt.items() if len(u) >= 2 * length}
        if len(so_eligible) >= 2:
            result = singling_out_sweep(
                enrollment, so_eligible, test_utt.embeddings,
                conversation_length=length,
                speaker_counts=[c for c in counts if c <= len(so_eligible)],
                n_runs=n_runs, n_folds=n_folds, seed=seed,
            )
            means, stds = result.mean(), result.std()
            for count in result.speaker_counts:
                rows.append({
                    "dataset": test_utt.dataset, "metric": "singling_out",
                    "L": length, "speakers": count,
                    "value": means[count], "std": stds[count],
                })
    return rows


def pair_datasets(discovered: dict[str, dict[str, Path]]) -> list[tuple[str, Path, Path]]:
    """Match each VPC enrollment directory with its trial directory.

    VPC names them ``<base>_enrolls<suffix>`` and ``<base>_trials_<kind><suffix>``,
    for example ``libri_dev_enrolls_mcadams`` and
    ``libri_dev_trials_mixed_mcadams``. Pairing is by the shared prefix before
    ``_enrolls``, plus the anonymization suffix after it.

    Args:
        discovered: Output of :func:`find_embedding_dirs`.

    Returns:
        One ``(name, enrollment utt-level dir, trial utt-level dir)`` per pair.
    """
    pairs: list[tuple[str, Path, Path]] = []
    for name, levels in discovered.items():
        if "_enrolls" not in name or "utt" not in levels:
            continue
        base, _, suffix = name.partition("_enrolls")
        for other, other_levels in discovered.items():
            if "utt" not in other_levels or "_trials" not in other:
                continue
            other_base, _, rest = other.partition("_trials")
            if other_base == base and rest.endswith(suffix):
                pairs.append((other, levels["utt"], other_levels["utt"]))
    return pairs


def benchmark_vpc_run(
    results_dir: str | Path,
    distractor_dataset: str | None = None,
    conversation_lengths: tuple[int, ...] = (1, 3),
    n_runs: int = 5,
    n_folds: int = 5,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Compute the legal metrics for every dataset pair in a VPC run.

    Args:
        results_dir: A VPC results directory, for example
            ``exp/asv_anon_mcadams``, or its ``cosine_out/emb_xvect`` subtree.
        distractor_dataset: Name of a dataset whose speakers enlarge the
            enrollment population, typically ``train-clean-360<suffix>``.
        conversation_lengths: Values of ``L``.
        n_runs: Random draws per point.
        n_folds: Folds for Singling Out.
        seed: Base seed.

    Returns:
        One row per (dataset, metric, L, population size).
    """
    discovered = find_embedding_dirs(results_dir)
    distractors = None
    if distractor_dataset:
        if distractor_dataset not in discovered:
            raise KeyError(
                f"no embeddings for {distractor_dataset!r}. Available: "
                f"{sorted(discovered)}"
            )
        distractors = load_vpc_embeddings(discovered[distractor_dataset]["utt"])

    rows: list[dict[str, object]] = []
    for _name, enroll_dir, trial_dir in pair_datasets(discovered):
        rows.extend(
            evaluate_vpc_dataset(
                load_vpc_embeddings(enroll_dir),
                load_vpc_embeddings(trial_dir),
                conversation_lengths=conversation_lengths,
                distractor_utt=distractors,
                n_runs=n_runs,
                n_folds=n_folds,
                seed=seed,
            )
        )
    return rows
