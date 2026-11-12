"""Figuras. No contienen lógica de modelos: reciben arreglos ya calculados."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

Array = np.ndarray
FIGURE_DPI = 150
_CELL_INCHES = 0.8
_MOSAIC_CELL_INCHES = 0.6
_MOSAIC_MINIMUM_WIDTH_INCHES = 5.0
_MAX_ANNOTATED_POINTS = 64


def save_image_rows(
    rows: Sequence[Array],
    row_names: Sequence[str],
    column_titles: Sequence[str],
    image_shape: tuple[int, int],
    is_binary: bool,
    path: Path,
) -> None:
    """Grilla de imágenes: una fila por elemento de `rows` (cada uno N x D)."""
    columns = len(rows[0])
    figure, axes = plt.subplots(
        len(rows),
        columns,
        squeeze=False,
        figsize=(max(3.0, columns * _CELL_INCHES), max(1.5, len(rows) * _CELL_INCHES * 1.5)),
    )
    for row_index, (images, name) in enumerate(zip(rows, row_names)):
        for column_index, image in enumerate(images):
            axis = axes[row_index, column_index]
            _draw_image(axis, image.reshape(image_shape), is_binary)
            if row_index == 0:
                axis.set_title(column_titles[column_index], fontsize=7)
        axes[row_index, 0].text(
            -0.15, 0.5, name, transform=axes[row_index, 0].transAxes,
            ha="right", va="center", fontsize=8,
        )
    _save(figure, path)


def save_image_mosaic(
    images: Array,
    grid_shape: tuple[int, int],
    image_shape: tuple[int, int],
    is_binary: bool,
    title: str,
    path: Path,
) -> None:
    """Mosaico compacto de `filas x columnas` imágenes, sin títulos por celda."""
    rows, columns = grid_shape
    # El ancho mínimo evita que el título se corte en mosaicos de pocas columnas.
    width = max(_MOSAIC_MINIMUM_WIDTH_INCHES, columns * _MOSAIC_CELL_INCHES)
    cell_height = width / columns * image_shape[0] / image_shape[1]
    figure, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=(width, rows * cell_height + 0.4)
    )
    for axis, image in zip(axes.ravel(), images):
        _draw_image(axis, image.reshape(image_shape), is_binary)
    for axis in axes.ravel()[len(images):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=10)
    _save(figure, path)


def save_latent_scatter(
    points: Array,
    labels: Sequence[str],
    groups: Sequence[str],
    axis_names: tuple[str, str],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    unique_groups = sorted(set(groups))
    group_array = np.asarray(groups)
    for group in unique_groups:
        selected = points[group_array == group]
        axis.scatter(selected[:, 0], selected[:, 1], label=group, s=28)
    if len(points) <= _MAX_ANNOTATED_POINTS:
        for label, point in zip(labels, points):
            axis.annotate(label, point, xytext=(4, 4), textcoords="offset points")
    if len(unique_groups) > 1:
        axis.legend(frameon=False, fontsize=8)
    axis.set(xlabel=axis_names[0], ylabel=axis_names[1], title="Espacio latente")
    _hide_outer_spines(axis)
    _save(figure, path)


def save_learning_curves(history: Sequence[dict[str, float]], path: Path) -> None:
    """Un panel por cada columna del historial, con la época en el eje x."""
    columns = [name for name in history[0] if name != "epoch"]
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(
        1, len(columns), squeeze=False, figsize=(3.6 * len(columns), 3.0)
    )
    for axis, column in zip(axes[0], columns):
        values = [row[column] for row in history]
        axis.plot(epochs, values)
        if column in ("loss", "reconstruction") and min(values) > 0:
            axis.set_yscale("log")
        axis.set(xlabel="Época", ylabel=column)
        _hide_outer_spines(axis)
    _save(figure, path)


def save_line_chart(
    x_values: Sequence[float],
    series: dict[str, Sequence[float]],
    x_label: str,
    y_label: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5.4, 3.6))
    for name, values in series.items():
        axis.plot(x_values, values, marker="o", label=name)
    axis.set(xlabel=x_label, ylabel=y_label)
    axis.legend(frameon=False)
    _hide_outer_spines(axis)
    _save(figure, path)


def save_curves(
    curves: dict[str, tuple[Sequence[float], Sequence[float]]],
    x_label: str,
    y_label: str,
    path: Path,
) -> None:
    """Como `save_line_chart`, pero cada curva con sus propios valores de x."""
    figure, axis = plt.subplots(figsize=(5.4, 3.6))
    for name, (x_values, y_values) in curves.items():
        axis.plot(x_values, y_values, marker="o", label=name)
    axis.set(xlabel=x_label, ylabel=y_label)
    axis.legend(frameon=False, fontsize="small")
    _hide_outer_spines(axis)
    _save(figure, path)


def _draw_image(axis: plt.Axes, image: Array, is_binary: bool) -> None:
    if is_binary:
        axis.imshow(image, cmap="gray_r", vmin=0.0, vmax=1.0)
    else:
        # Espectrogramas: frecuencia creciente hacia arriba.
        axis.imshow(image, cmap="magma", vmin=0.0, vmax=1.0, origin="lower", aspect="auto")
    axis.axis("off")


def _hide_outer_spines(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
