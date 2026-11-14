"""Ejercicio 2 con otras activaciones: el mismo AE y el mismo VAE con ReLU, leaky ReLU, tanh y sigmoide.

ReLU es la corrida base (configs/ej2_*_nsynth.json). Las demás salen de
configs/ej2_*_activations_he.json (leaky ReLU, inicialización He) y
configs/ej2_*_activations_xavier.json (tanh y sigmoide, inicialización Xavier):
misma semilla, datos, capas, épocas y optimizador; cambia solo la activación
y la inicialización que le corresponde.

Una muestra buena tiene que parecerse a alguna nota real y, a la vez, las
muestras tienen que variar tanto como las notas reales: un modelo que siempre
devuelve la misma nota promedio queda cerca de todo pero no genera nada nuevo.
Por eso se miden las dos cosas.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.presentacion.audio import AE_RUN, SAMPLE_SEED, VAE_RUN, NoteRenderer, typical_nearest_distance
from scripts.presentacion.common import OUTPUT, RESULTS, Array, mean_and_interval, save
from src.experiments.runner import load_run

ACTIVATIONS = {"relu": "ReLU", "leaky_relu": "leaky ReLU", "tanh": "tanh", "sigmoid": "sigmoide"}
INITIALIZER_OF = {"relu": "he", "leaky_relu": "he", "tanh": "xavier", "sigmoid": "xavier"}
BASE_RUNS = {"ae": AE_RUN, "vae": VAE_RUN}
MODEL_NAMES = {"ae": "Autoencoder", "vae": "VAE"}
MODEL_COLORS = {"ae": "C3", "vae": "C0"}
PRIOR_SAMPLES = 200
TILE_SAMPLES = 4
ACTIVE_DIMENSION_KL = 0.01  # nats: por debajo, la dimensión copia a la prior y no lleva información
Interval = tuple[float, float]  # media y semiancho del intervalo de 95 %


@dataclass(frozen=True)
class ActivationResult:
    model: str
    activation: str
    reconstruction_error: Interval  # entre notas
    nearest_note_distance: Interval  # entre muestras
    sample_diversity: Interval  # entre muestras
    active_dimensions: int | None


def make_all() -> None:
    renderer = None
    results = []
    data_reference = None
    for model_key in BASE_RUNS:
        for activation in ACTIVATIONS:
            model, dataset = load_run(run_path(model_key, activation))
            if data_reference is None:
                renderer = NoteRenderer(dataset.audio_codec)
                data_reference = (typical_nearest_distance(dataset.samples), data_diversity(dataset.samples))
            results.append(evaluate(model_key, activation, model, dataset.samples))
            if model_key == "vae":
                write_sample_tiles(model, activation, renderer)
    write_table(results)
    comparison_figure(results, *data_reference)


def run_path(model_key: str, activation: str) -> Path:
    if activation == "relu":
        return BASE_RUNS[model_key]
    experiment = f"ej2_{model_key}_activations_{INITIALIZER_OF[activation]}"
    return RESULTS / experiment / f"activation={activation}__seed={_seed_of(BASE_RUNS[model_key])}"


def _seed_of(run: Path) -> str:
    return run.name.split("seed=")[1]


def evaluate(model_key: str, activation: str, model, samples: Array) -> ActivationResult:
    per_note_error = np.mean((model.reconstruct(samples) - samples) ** 2, axis=1)
    generated = model.decode(_prior_points(model.latent_size, PRIOR_SAMPLES))
    nearest = [float(np.min(np.mean((samples - values) ** 2, axis=1))) for values in generated]
    active = None
    if model_key == "vae":
        mean, log_variance = model.encode_distribution(samples)
        divergence = 0.5 * np.mean(mean**2 + np.exp(log_variance) - 1.0 - log_variance, axis=0)
        active = int(np.sum(divergence > ACTIVE_DIMENSION_KL))
    result = ActivationResult(model_key, activation, mean_and_interval(per_note_error), mean_and_interval(nearest),
                              mean_and_interval(_distance_to_the_rest(generated)), active)
    print(f"    {MODEL_NAMES[model_key]} {activation}: reconstrucción {result.reconstruction_error[0]:.5f}, "
          f"muestra a nota más cercana {result.nearest_note_distance[0]:.4f}, "
          f"diversidad {result.sample_diversity[0]:.4f}"
          + (f", dimensiones activas {active}" if active is not None else ""))
    return result


def data_diversity(samples: Array) -> float:
    """La misma diversidad medida sobre notas reales, con la misma cantidad que de muestras."""
    chosen = np.random.default_rng(SAMPLE_SEED).choice(len(samples), PRIOR_SAMPLES, replace=False)
    return float(np.mean(_distance_to_the_rest(samples[chosen])))


def _distance_to_the_rest(points: Array) -> Array:
    """Para cada punto, el MSE medio contra todos los demás: cuánto se distingue del resto."""
    squared_norms = np.sum(points**2, axis=1)
    distances = (squared_norms[:, None] + squared_norms[None, :] - 2 * points @ points.T) / points.shape[1]
    np.fill_diagonal(distances, 0.0)
    return distances.sum(axis=1) / (len(points) - 1)


def _prior_points(latent_size: int, count: int) -> Array:
    return np.random.default_rng(SAMPLE_SEED).standard_normal((count, latent_size))


def write_sample_tiles(model, activation: str, renderer: NoteRenderer) -> None:
    """Los mismos primeros z que las muestras del deck, decodificados por el VAE de cada activación."""
    for index, values in enumerate(model.decode(_prior_points(model.latent_size, TILE_SAMPLES))):
        renderer.write(values, f"activacion_{activation}_{index}")


def write_table(results: list[ActivationResult]) -> None:
    path = OUTPUT / "e2_activaciones.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["modelo", "activacion", "mse_reconstruccion", "ic95_reconstruccion",
                         "mse_muestra_a_nota_mas_cercana", "ic95_muestra", "diversidad_muestras", "ic95_diversidad",
                         "dimensiones_activas"])
        for result in results:
            intervals = (result.reconstruction_error, result.nearest_note_distance, result.sample_diversity)
            writer.writerow([result.model, result.activation,
                             *(f"{value:.6f}" for interval in intervals for value in interval),
                             "" if result.active_dimensions is None else result.active_dimensions])


def comparison_figure(results: list[ActivationResult], typical_distance: float, diversity: float) -> None:
    """Reconstrucción, cercanía de las muestras a notas reales y diversidad de las muestras, por activación."""
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.0))
    panels = [
        (lambda row: row.reconstruction_error, "Error de reconstrucción\n(MSE; IC 95 % entre notas)", None),
        (lambda row: row.nearest_note_distance, "Muestra a la nota real más parecida\n(MSE; IC 95 % entre muestras)",
         (typical_distance, "nota real a su vecina")),
        (lambda row: row.sample_diversity, "Diversidad de las muestras\n(MSE medio entre ellas; IC 95 %)",
         (diversity, "notas reales")),
    ]
    positions = np.arange(len(ACTIVATIONS))
    for offset, model_key in zip((-0.2, 0.2), BASE_RUNS):
        rows = [next(result for result in results if result.model == model_key and result.activation == activation)
                for activation in ACTIVATIONS]
        for axis, (value_of, _, _) in zip(axes, panels):
            means, intervals = zip(*(value_of(row) for row in rows))
            axis.bar(positions + offset, means, 0.4, yerr=intervals, capsize=4, color=MODEL_COLORS[model_key],
                     label=MODEL_NAMES[model_key])
    for axis, (_, title, reference) in zip(axes, panels):
        if reference is not None:
            axis.axhline(reference[0], color="k", ls="--", lw=1.2, label=reference[1])
        axis.set_title(title)
        axis.set_xticks(positions, list(ACTIVATIONS.values()))
        axis.legend(fontsize=11)
    save(figure, "e2_activaciones")
