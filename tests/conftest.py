"""Shared test configuration."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _headless_matplotlib():
    """Force a non-interactive backend before any figure is drawn.

    Tests run without a display, and the plotting module selects Agg lazily on
    first use. Selecting it once up front keeps that deterministic and avoids a
    backend being chosen by whatever imports matplotlib first.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
