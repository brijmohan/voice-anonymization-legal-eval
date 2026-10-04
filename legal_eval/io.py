"""Readers and writers for the on-disk formats used by the evaluation.

The framework consumes Kaldi-style data directories, because that is what the
upstream anonymization and x-vector extraction toolkits produce:

``spk2utt``
    One line per speaker: ``<speaker-id> <utt-id> <utt-id> ...``
``xvector.h5``
    HDF5 file with one dataset per utterance id, holding that utterance's
    embedding. This is what Sidekit's ``extract_xvectors.py`` writes.

Score matrices are stored as ``.npy`` plus a JSON sidecar holding the row and
column speaker orders, so that a matrix is self-describing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_spk2utt(path: str | Path, min_utterances: int = 0) -> dict[str, list[str]]:
    """Read a Kaldi ``spk2utt`` file.

    Args:
        path: Path to the ``spk2utt`` file.
        min_utterances: Drop speakers with fewer than this many utterances.

    Returns:
        Mapping from speaker id to the list of its utterance ids, in file order.
    """
    spk2utt: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields:
                continue
            utterances = fields[1:]
            if len(utterances) >= min_utterances:
                spk2utt[fields[0]] = utterances
    return spk2utt


def load_utt2spk(path: str | Path) -> dict[str, str]:
    """Read a Kaldi ``utt2spk`` file into a mapping from utterance to speaker."""
    utt2spk: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if fields:
                utt2spk[fields[0]] = fields[1]
    return utt2spk


def load_xvectors(
    path: str | Path,
    utterances: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load utterance-level embeddings from an HDF5 or ``.npz`` file.

    Args:
        path: HDF5 file with one dataset per utterance id, or a ``.npz``
            archive with one array per utterance id.
        utterances: Restrict loading to these utterance ids. Loading only what
            is needed matters: the Common Voice x-vector files in the paper hold
            close to a million utterances.

    Returns:
        Mapping from utterance id to a 1-D embedding.
    """
    path = Path(path)
    wanted = None if utterances is None else list(dict.fromkeys(utterances))

    if path.suffix == ".npz":
        with np.load(path) as archive:
            keys = archive.files if wanted is None else wanted
            return {key: np.squeeze(archive[key]) for key in keys}

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Reading HDF5 x-vectors requires h5py. Install it with "
            "`pip install h5py`, or convert your embeddings to .npz."
        ) from exc

    # The large read cache matters when pulling scattered utterances out of a
    # file with hundreds of thousands of datasets.
    with h5py.File(path, "r", rdcc_nbytes=1024**2 * 512, rdcc_nslots=1_000_003) as handle:
        keys = list(handle.keys()) if wanted is None else wanted
        return {key: np.squeeze(handle[key][()]) for key in keys}


def utterances_of(spk2utt: Mapping[str, Sequence[str]]) -> list[str]:
    """Flatten a ``spk2utt`` mapping into a list of utterance ids."""
    return [utt for utterances in spk2utt.values() for utt in utterances]


@dataclass(frozen=True)
class ScoreMatrix:
    """A cosine score matrix together with the speaker order of its axes.

    Attributes:
        scores: Array of shape ``(n_enroll, n_test)``. ``scores[i, j]`` is the
            cosine similarity between enrollment speaker ``enroll_speakers[i]``
            and test item ``test_speakers[j]``.
        enroll_speakers: Row labels.
        test_speakers: Column labels. For the worst-case analysis a speaker may
            appear in several columns, so these are not necessarily unique.
        metadata: Free-form provenance (conversation length, attacker, seed).
    """

    scores: np.ndarray
    enroll_speakers: list[str]
    test_speakers: list[str]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        expected = (len(self.enroll_speakers), len(self.test_speakers))
        if self.scores.shape != expected:
            raise ValueError(
                f"score matrix has shape {self.scores.shape}, but there are "
                f"{expected[0]} enrollment and {expected[1]} test labels"
            )

    @property
    def enroll_index(self) -> dict[str, int]:
        """Mapping from enrollment speaker id to row index."""
        return {spk: i for i, spk in enumerate(self.enroll_speakers)}


def save_score_matrix(matrix: ScoreMatrix, path: str | Path) -> None:
    """Write a score matrix as ``<path>`` plus a ``<path>.json`` sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix.scores)
    sidecar = path.with_suffix(path.suffix + ".json")
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "enroll_speakers": matrix.enroll_speakers,
                "test_speakers": matrix.test_speakers,
                "metadata": matrix.metadata,
            },
            handle,
        )


def load_score_matrix(path: str | Path) -> ScoreMatrix:
    """Read a score matrix written by :func:`save_score_matrix`."""
    path = Path(path)
    scores = np.load(path)
    sidecar = path.with_suffix(path.suffix + ".json")
    with open(sidecar, encoding="utf-8") as handle:
        meta = json.load(handle)
    return ScoreMatrix(
        scores=scores,
        enroll_speakers=meta["enroll_speakers"],
        test_speakers=meta["test_speakers"],
        metadata=meta.get("metadata", {}),
    )


def write_results(results: Mapping[str, object], path: str | Path) -> None:
    """Write sweep results as JSON, converting numpy scalars to Python floats."""

    def default(obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"cannot serialise {type(obj)!r}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=default)


def read_results(path: str | Path) -> dict[str, object]:
    """Read a results JSON file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
