"""Shared helpers for report figures. Figures use matplotlib's default style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only written to disk, never shown

import matplotlib.pyplot as plt  # noqa: E402  (must come after selecting the backend)


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Write `figure` as a PNG and release its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
