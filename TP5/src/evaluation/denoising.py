"""Capacidad del modelo de limpiar entradas corrompidas."""

from __future__ import annotations

import numpy as np

from src import plotting
from src.data.noise import Noise
from src.evaluation.context import EvaluationContext, Metrics, write_csv
from src.evaluation.metrics import PIXEL_ERROR_TARGET, error_unit, pixel_errors, sample_errors
from src.options import ConfigurationError, Options

Array = np.ndarray
_MAX_EXAMPLE_COLUMNS = 32


def evaluate_denoising(context: EvaluationContext, options: Options) -> Metrics:
    """Corrompe los datos con ruido nuevo y mide el error antes y después del modelo.

    Opciones: `levels` (lista de niveles del tipo `type`, o diccionario
    tipo → niveles para evaluar varios tipos), `type` (por defecto 'flip' en
    datos binarios y 'gaussian' en continuos), `trials` (repeticiones de todo
    el dataset por nivel) y `examples` (figura de ejemplo por nivel; por
    defecto true). El error se mide siempre contra el patrón limpio.

    En datos binarios agrega el error de los decodificadores óptimos (ver
    `optimal_errors`), el techo contra el que se compara el modelo.
    """
    dataset = context.dataset
    levels_by_type = _levels_by_type(options, "flip" if dataset.is_binary else "gaussian")
    trials = options.take("trials", 20)
    examples = options.take("examples", True)
    options.ensure_consumed()
    if not isinstance(trials, int) or trials < 1:
        raise ConfigurationError("evaluation.denoising.trials debe ser un entero positivo")

    targets = np.tile(dataset.samples, (trials, 1))
    shown = slice(0, min(len(dataset), _MAX_EXAMPLE_COLUMNS))
    rows = []
    for noise_type, levels in levels_by_type.items():
        for level in levels:
            noise = Noise(noise_type, level)
            noisy = noise.apply(targets, context.rng)
            cleaned = context.model.reconstruct(noisy)
            input_errors = sample_errors(dataset, targets, noisy)
            output_errors = sample_errors(dataset, targets, cleaned)
            row = {
                "noise_type": noise_type,
                "level": level,
                "input_mean_error": float(input_errors.mean()),
                "output_mean_error": float(output_errors.mean()),
                "output_max_error": float(output_errors.max()),
            }
            if dataset.is_binary:
                row["fraction_within_target"] = float(np.mean(output_errors <= PIXEL_ERROR_TARGET))
                fewest_pixels, most_within = optimal_errors(noisy, targets, dataset.samples, noise)
                row["optimal_mean_error"] = float(fewest_pixels.mean())
                row["optimal_fraction_within_target"] = float(
                    np.mean(most_within <= PIXEL_ERROR_TARGET)
                )
            rows.append(row)
            if examples:
                plotting.save_image_rows(
                    [dataset.samples[shown], noisy[shown], context.displayable(cleaned[shown])],
                    ["Original", "Con ruido", "Salida"],
                    dataset.labels[shown],
                    dataset.image_shape,
                    dataset.is_binary,
                    context.output_directory / f"denoising_{noise_type}_{level:g}.png",
                )

    write_csv(context.output_directory / "denoising.csv", rows)
    _save_chart(rows, error_unit(dataset), context)
    # Con un solo tipo, las métricas conservan el nombre sin tipo de las configs anteriores.
    several = len(levels_by_type) > 1
    return {
        f"denoising_{row['noise_type'] + '_' if several else ''}output_mean_error@{row['level']:g}":
            row["output_mean_error"]
        for row in rows
    }


def optimal_errors(noisy: Array, targets: Array, patterns: Array, noise: Noise) -> tuple[Array, Array]:
    """Píxeles mal de los dos mejores decodificadores posibles para entradas ruidosas.

    Ambos conocen los patrones limpios (equiprobables, como en la evaluación)
    y cómo actúa el ruido, y parten de la probabilidad de cada patrón dada la
    entrada. Ningún método puede superarlos en promedio, cada uno en su métrica:
      - el primero minimiza los píxeles mal esperados: cada píxel toma su valor
        más probable;
      - el segundo maximiza la probabilidad de quedar a ≤ 1 píxel del patrón:
        elige, entre los patrones y sus vecinos a un píxel, el que tiene más
        probabilidad acumulada a esa distancia (un punto medio entre dos letras
        a 2 píxeles cuenta para ambas).
    """
    log_likelihood = noise.log_likelihood(noisy, patterns)
    posterior = np.exp(log_likelihood - log_likelihood.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)

    fewest_pixels = posterior @ patterns  # probabilidad de que cada píxel esté prendido

    pixels = patterns.shape[1]
    neighbours = np.abs(patterns[:, None, :] - np.eye(pixels)[None, :, :]).reshape(-1, pixels)
    candidates = np.vstack([patterns, neighbours])
    distances = np.count_nonzero(candidates[:, None, :] != patterns[None, :, :], axis=2)
    reachable = posterior @ (distances <= PIXEL_ERROR_TARGET).T  # (muestras, candidatos)
    most_within = candidates[reachable.argmax(axis=1)]

    return pixel_errors(targets, fewest_pixels), pixel_errors(targets, most_within)


def _levels_by_type(options: Options, default_type: str) -> dict[str, list[float]]:
    levels = options.take("levels", [0.0, 0.05, 0.1, 0.2, 0.3])
    noise_type = options.take("type", None)
    if isinstance(levels, dict):
        if noise_type is not None:
            raise ConfigurationError(
                "evaluation.denoising: con levels por tipo no se usa 'type'"
            )
        levels_by_type = levels
    else:
        levels_by_type = {noise_type or default_type: levels}
    if not levels_by_type or not all(levels_by_type.values()):
        raise ConfigurationError("evaluation.denoising requiere niveles no vacíos")
    return levels_by_type


def _save_chart(rows: list[dict], unit: str, context: EvaluationContext) -> None:
    """Error a la salida contra error a la entrada: así los tipos de ruido son comparables."""
    curves: dict[str, tuple[list[float], list[float]]] = {}
    for noise_type in dict.fromkeys(row["noise_type"] for row in rows):
        typed = [row for row in rows if row["noise_type"] == noise_type]
        inputs = [row["input_mean_error"] for row in typed]
        curves[f"Salida ({noise_type})"] = (inputs, [row["output_mean_error"] for row in typed])
        if "optimal_mean_error" in typed[0]:
            curves[f"Óptimo ({noise_type})"] = (inputs, [row["optimal_mean_error"] for row in typed])
    everything = [row["input_mean_error"] for row in rows]
    curves["Entrada con ruido"] = (sorted(everything), sorted(everything))
    plotting.save_curves(
        curves,
        f"Promedio de {unit} a la entrada",
        f"Promedio de {unit}",
        context.output_directory / "denoising.png",
    )
