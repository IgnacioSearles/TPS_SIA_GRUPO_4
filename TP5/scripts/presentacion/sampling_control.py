"""Control de muestreo de NSynth: N(0,I) contra gaussiana ajustada, para AE y VAE por igual.

Lee las distancias por muestra que escribe `python -m src.analysis.nsynth_sampling`
(copiadas a reports/ej2_search/nsynth_sampling_control/) y agrega como referencia
la distancia entre notas reales vecinas, igual que la figura e2_distancia_muestras.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from scripts.presentacion.audio import real_neighbor_distances
from scripts.presentacion.common import ROOT, Array, read_csv, save
from src.data.factory import build_dataset
from src.options import Options

CONTROL = ROOT / "reports" / "ej2_search" / "nsynth_sampling_control"
VAE_CONFIG = ROOT / "configs" / "ej2_nsynth_sampling_control_vae.json"
SAMPLE_COUNT = 256
# Una muestra está "lejos" si supera la distancia que el 99 % de las notas reales tiene a su vecina.
FAR_PERCENTILE = 99
# (carpeta del método, rótulo, color): AE en rojo y VAE en azul, como en e2_distancia_muestras.
METHODS = (
    ("vae_fitted", "VAE · gaussiana ajustada", "C0"),
    ("vae_standard", "VAE · N(0, I)", "C0"),
    ("ae_fitted", "AE · gaussiana ajustada", "C3"),
    ("ae_standard", "AE · N(0, I)", "C3"),
)
REFERENCE_LABEL = "referencia: nota real\ny su vecina más parecida"


def method_distances(method: str) -> Array:
    rows = read_csv(CONTROL / method / "samples.csv")
    if len(rows) != SAMPLE_COUNT:
        raise ValueError(f"{method}: se esperaban {SAMPLE_COUNT} muestras y hay {len(rows)}")
    return np.array([float(row["nearest_train_mse"]) for row in rows])


def reference_distances() -> Array:
    """Cada nota de entrenamiento contra su vecina más parecida (todas, para que el percentil 99 sea estable)."""
    config = json.loads(VAE_CONFIG.read_text())
    dataset = build_dataset(Options(config["dataset"], "dataset"), ROOT)
    return real_neighbor_distances(dataset.samples, len(dataset.samples))


def draw_row(axis: plt.Axes, position: int, label: str, color: str, scaled: Array, jitter) -> None:
    filled = "ajustada" not in label
    axis.scatter(scaled, position + jitter.uniform(-0.18, 0.18, len(scaled)), s=14, alpha=0.45,
                 color=color if filled else "none", edgecolors="none" if filled else color, linewidths=0.8)
    median = np.median(scaled)
    axis.plot([median, median], [position - 0.3, position + 0.3], color="black", lw=2.5)
    axis.text(median, position + 0.34, f"mediana {median:.0f}", ha="center", va="bottom", fontsize=11)


def make_all() -> None:
    reference = 1000 * reference_distances()
    far = np.percentile(reference, FAR_PERCENTILE)
    rows = [(label, color, 1000 * method_distances(method)) for method, label, color in METHODS]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.axvline(far, color="0.3", ls="--", lw=1.2)
    jitter = np.random.default_rng(0)
    for position, (label, color, scaled) in enumerate(rows):
        draw_row(axis, position, label, color, scaled, jitter)
        beyond = int((scaled > far).sum())
        axis.text(1.01, position, f"{beyond}/{len(scaled)} lejos", transform=axis.get_yaxis_transform(),
                  va="center", fontsize=11)
        print(f"    {label}: mediana {np.median(scaled):.1f}, lejos {beyond}/{len(scaled)}")
    draw_row(axis, len(rows), REFERENCE_LABEL, "0.45", reference, jitter)
    axis.text(far, len(rows) + 0.55, f" lejos: más que el {FAR_PERCENTILE} % de las notas reales",
              fontsize=10, color="0.3", va="bottom")
    print(f"    umbral lejos: {far:.1f}")
    axis.set_yticks(range(len(rows) + 1), [label for label, _, _ in rows] + [REFERENCE_LABEL])
    axis.set_ylim(-0.5, len(rows) + 0.9)
    axis.set_xlim(left=0)
    axis.set_xlabel("diferencia con la nota real más parecida del dataset (MSE × 1000)")
    save(figure, "e2_sorteo_ajustado")
