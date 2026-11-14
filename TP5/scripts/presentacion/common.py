"""Rutas, estilo y utilidades compartidas por las figuras de la presentación."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "reports" / "presentacion"
AUDIO = OUTPUT / "audio"
TILES = OUTPUT / "espectrogramas"

PIXEL_TARGET = 1  # la consigna: a lo sumo un píxel incorrecto por letra
SPECTROGRAM_CMAP = "magma"  # el mismo que usan las figuras de src/plotting.py

# Estilo por defecto de matplotlib; solo se agranda el texto para que se lea en una diapositiva.
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

Array = np.ndarray


def save(figure: plt.Figure, name: str) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{name}.png"
    figure.savefig(path)
    plt.close(figure)
    print(f"  {path.relative_to(ROOT)}")
    return path


def read_summary(experiment: str) -> dict[str, list[dict[str, str]]]:
    """Corridas de un experimento agrupadas por configuración, en el orden del CSV."""
    path = RESULTS / experiment / "summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path}: correr antes configs/{experiment}.json")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[row["configuration"]].append(row)
    return dict(grouped)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


CONFIDENCE_Z = 1.96  # intervalo de 95 % con aproximación normal


def mean_and_interval(values) -> tuple[float, float]:
    """Media y semiancho del intervalo de confianza del 95 % de la media (error estándar × 1,96)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("Hacen falta al menos dos valores para estimar la dispersión")
    return float(values.mean()), float(CONFIDENCE_Z * values.std(ddof=1) / np.sqrt(len(values)))


def glyph(axis: plt.Axes, pattern: Array, shape: tuple[int, int] = (7, 5)) -> None:
    """Un carácter del font: negro = píxel encendido."""
    axis.imshow(np.reshape(pattern, shape), cmap="gray_r", vmin=0.0, vmax=1.0)
    bare(axis)


def bare(axis: plt.Axes) -> None:
    """Sin marcas ni bordes: para imágenes que no son gráficos."""
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def spectrogram(axis: plt.Axes, image: Array) -> None:
    axis.imshow(image, cmap=SPECTROGRAM_CMAP, vmin=0.0, vmax=1.0, origin="lower", aspect="auto")
    axis.axis("off")


def save_spectrogram_tile(image: Array, name: str) -> Path:
    """Un espectrograma solo, sin ejes: la diapositiva pone el rótulo y el botón de audio."""
    TILES.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(1.6, 2.0))
    spectrogram(axis, image)
    figure.subplots_adjust(0, 0, 1, 1)
    path = TILES / f"{name}.png"
    figure.savefig(path, bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return path
