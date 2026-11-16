"""Qué decodifican AE y VAE lejos de sus códigos: la gaussiana ajustada, estirada por un factor s.

s = 1 es el sorteo ajustado de e2_sorteo_ajustado; s = 2 queda el doble de lejos, medido con
la nube de cada modelo (sus coordenadas tienen escalas distintas). Mismos vectores base para los dos.

Las muestras lejanas del VAE salen más suaves, y una salida suave queda cerca de cualquier nota
suave. Por eso la distancia se mide relativa a la energía de la propia muestra: MSE con la nota real
más parecida dividido por la media de sus valores al cuadrado.
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from scripts.presentacion.audio import NoteRenderer, nearest_distances
from scripts.presentacion.common import OUTPUT, RESULTS, Array, save
from src.experiments.runner import load_run

RUNS = {"ae": RESULTS / "ej2_nsynth_sampling_control_ae" / "base__seed=7",
        "vae": RESULTS / "ej2_nsynth_sampling_control_vae" / "base__seed=7"}
FITTED = RESULTS / "nsynth_sampling_control"
SCALES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SHOWN_SCALE = 3.0
SHOWN_COUNT = 6
SAMPLE_COUNT = 256
GENERATION_SEED = 2026111900  # el mismo del control de muestreo
FAR_PERCENTILE = 99
LABELS = {"ae": ("autoencoder", "C3"), "vae": ("VAE", "C0")}


def relative_distances(references: Array, outputs: Array) -> Array:
    return nearest_distances(references, outputs) / np.mean(outputs ** 2, axis=1)


def real_relative_distances(samples: Array) -> Array:
    """Cada nota real contra su vecina más parecida (sin contarse a sí misma), con la misma normalización."""
    squared = np.sum(samples ** 2, axis=1)
    distances = (squared[:, None] + squared[None, :] - 2 * samples @ samples.T) / samples.shape[1]
    np.fill_diagonal(distances, np.inf)
    return np.maximum(distances.min(axis=1), 0) / np.mean(samples ** 2, axis=1)


def stretched_codes(name: str, epsilon: Array, scale: float) -> Array:
    with np.load(FITTED / f"fitted_gaussian_{name}.npz") as fitted:
        return fitted["mean"] + scale * epsilon @ fitted["square_root"].T


def far_fractions(models: dict, references: Array, epsilon: Array, cutoff: float) -> dict[str, list[float]]:
    rows, fractions = [], {name: [] for name in models}
    quiet = np.percentile(references.mean(axis=1), 5)
    for scale in SCALES:
        for name, model in models.items():
            outputs = model.decode(stretched_codes(name, epsilon, scale).astype(model.precision))
            relative = relative_distances(references, outputs)
            fractions[name].append(100 * float(np.mean(relative > cutoff)))
            rows.append({"model": name, "scale": scale, "far_percent": fractions[name][-1],
                         "median_relative_distance": float(np.median(relative)),
                         "quieter_than_95pct_of_real": int(np.sum(outputs.mean(axis=1) < quiet))})
            print(f"    {name} s={scale}: lejos {fractions[name][-1]:.0f} %")
    with (OUTPUT / "e2_lejos.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return fractions


def far_figure(fractions: dict[str, list[float]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.4))
    for name, values in fractions.items():
        label, color = LABELS[name]
        axis.plot(SCALES, values, marker="o", color=color, lw=2)
        axis.text(SCALES[-1] + 0.1, values[-1], label, color=color, va="center", fontsize=13)
    axis.axvline(1, color="0.6", ls=":", lw=1)
    axis.text(1, 101, "sorteo ajustado", ha="center", va="bottom", fontsize=10, color="0.4")
    axis.set_xticks(SCALES, [f"{scale:g}" for scale in SCALES])
    axis.set_ylim(0, 100)
    axis.set_xlim(0.3, SCALES[-1] + 0.7)
    axis.set_xlabel("cuánto nos alejamos de las notas (× el ancho de la nube de códigos)")
    axis.set_ylabel("muestras que no se parecen\na ninguna nota real (%)")
    save(figure, "e2_lejos")


def far_tiles(models: dict, renderer: NoteRenderer, epsilon: Array) -> None:
    for name, model in models.items():
        codes = stretched_codes(name, epsilon[:SHOWN_COUNT], SHOWN_SCALE).astype(model.precision)
        for index, values in enumerate(model.decode(codes)):
            renderer.write(values, f"lejos_{name}_{index}")


def make_all() -> None:
    loaded = {name: load_run(path) for name, path in RUNS.items()}
    models = {name: model for name, (model, _) in loaded.items()}
    dataset = loaded["vae"][1]
    references = dataset.samples
    epsilon = np.random.default_rng(GENERATION_SEED).standard_normal((SAMPLE_COUNT, models["vae"].latent_size))
    cutoff = float(np.percentile(real_relative_distances(references), FAR_PERCENTILE))
    print(f"    umbral relativo ({FAR_PERCENTILE} % de las notas reales): {cutoff:.2f}")
    far_figure(far_fractions(models, references, epsilon, cutoff))
    far_tiles(models, NoteRenderer(dataset.audio_codec), epsilon)
