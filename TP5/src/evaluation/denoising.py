"""Capacidad del modelo de limpiar entradas corrompidas."""

from __future__ import annotations

import numpy as np

from src import plotting
from src.data.noise import Noise
from src.evaluation.context import EvaluationContext, Metrics, write_csv
from src.evaluation.metrics import PIXEL_ERROR_TARGET, error_unit, sample_errors
from src.options import ConfigurationError, Options

_MAX_EXAMPLE_COLUMNS = 32


def evaluate_denoising(context: EvaluationContext, options: Options) -> Metrics:
    """Corrompe los datos con ruido nuevo y mide el error antes y después del modelo.

    Opciones: `levels`, `trials` (repeticiones de todo el dataset por nivel) y
    `type` (por defecto 'flip' en datos binarios y 'gaussian' en continuos).
    El error se mide siempre contra el patrón limpio.
    """
    dataset = context.dataset
    levels = options.take("levels", [0.0, 0.05, 0.1, 0.2, 0.3])
    trials = options.take("trials", 20)
    noise_type = options.take("type", "flip" if dataset.is_binary else "gaussian")
    options.ensure_consumed()
    if not levels or not isinstance(trials, int) or trials < 1:
        raise ConfigurationError("evaluation.denoising requiere levels no vacío y trials positivo")

    targets = np.tile(dataset.samples, (trials, 1))
    shown = slice(0, min(len(dataset), _MAX_EXAMPLE_COLUMNS))
    rows = []
    for level in levels:
        noisy = Noise(noise_type, level).apply(targets, context.rng)
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
        rows.append(row)
        plotting.save_image_rows(
            [dataset.samples[shown], noisy[shown], context.displayable(cleaned[shown])],
            ["Original", "Con ruido", "Salida"],
            dataset.labels[shown],
            dataset.image_shape,
            dataset.is_binary,
            context.output_directory / f"denoising_{noise_type}_{level:g}.png",
        )

    write_csv(context.output_directory / "denoising.csv", rows)
    plotting.save_line_chart(
        levels,
        {
            "Entrada con ruido": [row["input_mean_error"] for row in rows],
            "Salida del modelo": [row["output_mean_error"] for row in rows],
        },
        f"Nivel de ruido ({noise_type})",
        f"Promedio de {error_unit(dataset)}",
        context.output_directory / "denoising.png",
    )
    return {f"denoising_output_mean_error@{row['level']:g}": row["output_mean_error"] for row in rows}
