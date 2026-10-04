"""Loading the published results behind the paper's figure.

The paper's Figure 1 is nine panels: three metrics by three conversation
lengths, each with four attacker curves. The values behind it are shipped in
``data/paper_results/`` as tidy CSV, one row per plotted point:

``attacker,L,enrollment_speakers,curve_type,<metric>_mean,<metric>_std,chance``

The EER is loaded in its raw form, the same convention
:func:`~legal_eval.sweeps.eer_sweep` returns, so that the single inversion to
the ``1 - EER`` the paper plots happens in the plotting layer for both sources.

These are summary statistics, not per-run values, so they load into
:class:`PublishedCurve` rather than
:class:`~legal_eval.sweeps.SweepResult`. Both expose ``speaker_counts``,
``mean()`` and ``std()``, so the plotting functions accept either.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

#: Attacker labels as the CSVs spell them, mapped to this package's keys.
ATTACKER_KEYS = {
    "original": "original",
    "INFORMED": "informed",
    "SEMI-INFORMED": "semi_informed",
    "IGNORANT": "ignorant",
}

#: Metric name -> (CSV file, mean column, std column).
_METRIC_FILES = {
    "linkability": ("linkability_results.csv", "linkability_mean", "linkabilitystd"),
    "singling_out": ("singling_out_results.csv", "singling_out_mean", "singling_out_std"),
    # The raw EER, matching what eer_sweep returns. The figure plots 1-EER, and
    # the plotting layer does that inversion for both sources. The same values
    # already inverted are in one_minus_eer_results.csv, for reading directly.
    "eer": ("eer_results.csv", "eer_mean", "eer_std"),
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "paper_results"


@dataclass
class PublishedCurve:
    """One published curve: a metric, for one attacker and conversation length.

    Attributes:
        metric: ``"linkability"``, ``"singling_out"`` or ``"eer"``.
        attacker: One of the values of :data:`ATTACKER_KEYS`.
        conversation_length: The ``L`` this curve was computed at.
        means: Mapping from speaker count to the published mean.
        stds: Mapping from speaker count to the published standard deviation.
        chance: Mapping from speaker count to the chance level recorded in the CSV.
    """

    metric: str
    attacker: str
    conversation_length: int
    means: dict[int, float] = field(default_factory=dict)
    stds: dict[int, float] = field(default_factory=dict)
    chance: dict[int, float] = field(default_factory=dict)

    @property
    def speaker_counts(self) -> list[int]:
        """The sorted speaker counts this curve covers."""
        return sorted(self.means)

    def mean(self) -> dict[int, float]:
        """Mean per speaker count. Named to match :class:`SweepResult`."""
        return {n: self.means[n] for n in self.speaker_counts}

    def std(self) -> dict[int, float]:
        """Standard deviation per speaker count."""
        return {n: self.stds[n] for n in self.speaker_counts}

    @property
    def values(self) -> dict[int, list[float]]:
        """The mean as a single-element list per point.

        Only summary statistics were published, so there is nothing per-run to
        return. This exists so a published curve can stand in for a
        :class:`SweepResult` where one is expected.
        """
        return {n: [self.means[n]] for n in self.speaker_counts}


def load_paper_results(
    metric: str | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, dict[int, dict[str, PublishedCurve]]]:
    """Load the published results as ``metric -> L -> attacker -> curve``.

    The returned structure is exactly what
    :func:`~legal_eval.plotting.plot_paper_figure` expects.

    Args:
        metric: Load only this metric. ``None`` loads all three.
        data_dir: Directory holding the CSVs. Defaults to the shipped
            ``data/paper_results``.

    Returns:
        Nested mapping of published curves.

    Raises:
        ValueError: If ``metric`` is not one of the three.
        FileNotFoundError: If a CSV is missing, with a pointer to the docs.
    """
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    wanted = list(_METRIC_FILES) if metric is None else [metric]
    for name in wanted:
        if name not in _METRIC_FILES:
            raise ValueError(
                f"unknown metric {name!r}, expected one of {sorted(_METRIC_FILES)}"
            )

    results: dict[str, dict[int, dict[str, PublishedCurve]]] = {}
    for name in wanted:
        filename, mean_column, std_column = _METRIC_FILES[name]
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. The published results ship with the "
                "repository; see docs/reproduction.md if you are working from a "
                "partial checkout."
            )
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                attacker = ATTACKER_KEYS.get(row["attacker"], row["attacker"])
                length = int(row["L"])
                curve = (
                    results.setdefault(name, {})
                    .setdefault(length, {})
                    .setdefault(
                        attacker,
                        PublishedCurve(
                            metric=name, attacker=attacker, conversation_length=length
                        ),
                    )
                )
                count = int(row["enrollment_speakers"])
                curve.means[count] = float(row[mean_column])
                curve.stds[count] = float(row[std_column])
                curve.chance[count] = float(row["chance"])
    return results
