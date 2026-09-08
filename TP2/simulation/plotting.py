"""Dibuja los resúmenes de `simulation.analysis` como barras con barras de error.

La estadística vive en `simulation.analysis`; acá solo se decide cómo se ve. La
paleta es de un solo tono porque el gráfico compara magnitudes de una sola
serie: el color no codifica identidad, así que no hace falta leyenda.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from simulation.analysis import DEFAULT_CONFIDENCE, GroupSummary

_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_BAR = "#2a78d6"
_ERROR_BAR = "#0d366b"
_RUN_POINT = "#0d366b"
_GRID = "#e3e2df"

_BAR_HEIGHT = 0.48
_POINT_JITTER = 0.2


def plot_group_comparison(
    summaries: Sequence[GroupSummary],
    output: str | Path,
    *,
    title: str,
    axis_label: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Path:
    """Escribe un gráfico de barras horizontales con intervalos de confianza.

    Las barras van horizontales porque las etiquetas son nombres largos, y de
    menor a mayor de arriba hacia abajo para que se lea como un ranking. Además
    de la media se dibuja cada corrida individual, para que la dispersión sea
    visible y no quede escondida detrás de un solo bigote.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not summaries:
        raise ValueError("no hay grupos para graficar.")
    positions = range(len(summaries))
    figure, axis = plt.subplots(figsize=(9, 0.52 * len(summaries) + 2.2))
    figure.patch.set_facecolor(_SURFACE)
    axis.set_facecolor(_SURFACE)

    axis.barh(
        positions,
        [summary.mean for summary in summaries],
        height=_BAR_HEIGHT,
        color=_BAR,
        zorder=2,
    )
    axis.errorbar(
        [summary.mean for summary in summaries],
        positions,
        xerr=[summary.confidence_margin(confidence) for summary in summaries],
        fmt="none",
        ecolor=_ERROR_BAR,
        elinewidth=2,
        capsize=5,
        capthick=2,
        zorder=4,
    )
    _draw_individual_runs(axis, summaries)
    _annotate_means(axis, summaries, confidence)

    axis.set_yticks(list(positions), [summary.label for summary in summaries])
    axis.invert_yaxis()  # el grupo más rápido queda arriba
    axis.set_xlabel(axis_label, color=_TEXT_SECONDARY)
    axis.set_title(title, color=_TEXT_PRIMARY, fontsize=13, loc="left", pad=14)
    _apply_recessive_chrome(axis, summaries, confidence)

    figure.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, facecolor=_SURFACE)
    plt.close(figure)
    return output_path


def _draw_individual_runs(axis, summaries: Sequence[GroupSummary]) -> None:
    """Superpone cada corrida para que se vea la dispersión real, no solo el resumen."""
    for position, summary in enumerate(summaries):
        offsets = _spread_offsets(summary.runs)
        axis.scatter(
            summary.values,
            [position + offset for offset in offsets],
            s=22,
            facecolors=_SURFACE,
            edgecolors=_RUN_POINT,
            linewidths=1.2,
            zorder=3,
        )


def _spread_offsets(count: int) -> list[float]:
    """Reparte los puntos de una misma barra en vertical para que no se tapen."""
    if count == 1:
        return [0.0]
    step = 2 * _POINT_JITTER / (count - 1)
    return [-_POINT_JITTER + index * step for index in range(count)]


def _annotate_means(axis, summaries: Sequence[GroupSummary], confidence: float) -> None:
    """Etiqueta directa al final de cada barra: evita tener que leer contra la grilla."""
    for position, summary in enumerate(summaries):
        axis.annotate(
            f"{summary.mean:.2f}",
            (summary.mean + summary.confidence_margin(confidence), position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=_TEXT_SECONDARY,
        )


def _apply_recessive_chrome(axis, summaries: Sequence[GroupSummary], confidence: float) -> None:
    """Deja la grilla y los ejes en segundo plano y reserva lugar para las etiquetas."""
    axis.xaxis.grid(True, color=_GRID, linewidth=1, zorder=0)
    axis.set_axisbelow(True)
    axis.yaxis.grid(False)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(_GRID)
    axis.tick_params(colors=_TEXT_SECONDARY, length=0)
    widest = max(
        summary.mean + summary.confidence_margin(confidence) for summary in summaries
    )
    axis.set_xlim(0, widest * 1.12)
