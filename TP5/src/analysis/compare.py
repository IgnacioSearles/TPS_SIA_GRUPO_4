"""Compara las configuraciones de un experimento ya corrido.

Uso, desde TP5:
    python -m src.analysis.compare results/ej1a_optimizers
    python -m src.analysis.compare results/ej1b_denoising --metric denoising_output_mean_error@0.1

Genera en la carpeta del experimento:
    comparison_<métrica>.png   media y desvío entre semillas de cada configuración
    comparison_curves.png      curvas de aprendizaje promedio por configuración
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation.context import safe_filename  # noqa: E402
from src.experiments.runner import EXPERIMENT_SUMMARY_FILE, HISTORY_FILE  # noqa: E402
from src.plotting import FIGURE_DPI  # noqa: E402

DEFAULT_METRIC = "max_pixel_errors"
DEFAULT_CURVE = "loss"


def compare(experiment_directory: Path, metric: str, curve: str) -> None:
    rows = _read_csv(experiment_directory / EXPERIMENT_SUMMARY_FILE)
    runs_by_configuration: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        runs_by_configuration[row["configuration"]].append(row)
    _plot_metric(runs_by_configuration, metric, experiment_directory)
    _plot_curves(runs_by_configuration, curve, experiment_directory)


def _plot_metric(
    runs_by_configuration: dict[str, list[dict[str, str]]], metric: str, directory: Path
) -> None:
    names, values = [], []
    for name, runs in runs_by_configuration.items():
        present = _metric_values(runs, metric)
        # Una métrica puede faltar en algunas corridas (p. ej. nunca alcanzó la meta).
        names.append(name if len(present) == len(runs) else f"{name} ({len(present)}/{len(runs)})")
        values.append(present)
    if all(len(present) == 0 for present in values):
        raise SystemExit(f"Ninguna corrida tiene valor para '{metric}'")
    positions = np.arange(len(names))
    # Barras horizontales: los nombres de configuración suelen ser largos.
    figure, axis = plt.subplots(figsize=(8.5, 0.5 * len(names) + 1.4))
    # Las configuraciones sin ningún valor quedan listadas, sin barra.
    means = [v.mean() if len(v) else np.nan for v in values]
    deviations = [v.std() if len(v) else np.nan for v in values]
    axis.barh(positions, means, xerr=deviations, capsize=4)
    for position, runs in zip(positions, values):
        axis.scatter(runs, np.full(len(runs), position), color="black", s=10, zorder=3)
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlabel(f"{metric}: media ± desvío entre semillas")
    _finish(figure, axis, directory / f"comparison_{safe_filename(metric)}.png")


def _plot_curves(
    runs_by_configuration: dict[str, list[dict[str, str]]], curve: str, directory: Path
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for name, runs in runs_by_configuration.items():
        histories = [_read_csv(directory / run["run"] / HISTORY_FILE) for run in runs]
        epochs = [float(row["epoch"]) for row in histories[0]]
        mean_curve = np.mean(
            [[float(row[curve]) for row in history] for history in histories], axis=0
        )
        axis.plot(epochs, mean_curve, label=name)
    if curve == DEFAULT_CURVE:
        axis.set_yscale("log")
    axis.set(xlabel="Época", ylabel=f"{curve} (media entre semillas)")
    axis.legend(frameon=False, fontsize=8)
    _finish(figure, axis, directory / "comparison_curves.png")


def _metric_values(runs: list[dict[str, str]], metric: str) -> np.ndarray:
    """Valores de la métrica en las corridas que la tienen."""
    if metric not in runs[0]:
        raise SystemExit(f"La métrica '{metric}' no existe. Disponibles: {', '.join(runs[0])}")
    return np.array([float(run[metric]) for run in runs if run[metric] != ""])


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"No se encontró {path}; correr primero el experimento")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _finish(figure: plt.Figure, axis: plt.Axes, path: Path) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    print(f"Figura: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara configuraciones de un experimento")
    parser.add_argument("experiment", type=Path, help="carpeta results/<experimento>")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="columna de summary.csv")
    parser.add_argument("--curve", default=DEFAULT_CURVE, help="columna de history.csv")
    arguments = parser.parse_args()
    compare(arguments.experiment, arguments.metric, arguments.curve)
