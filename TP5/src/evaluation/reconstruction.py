"""Evaluaciones sobre los datos de entrenamiento: reconstrucción y espacio latente."""

from __future__ import annotations

import numpy as np

from src import plotting
from src.evaluation.context import EvaluationContext, Metrics, safe_filename, write_csv
from src.evaluation.metrics import sample_errors
from src.options import ConfigurationError, Options

Array = np.ndarray


def evaluate_reconstruction(context: EvaluationContext, options: Options) -> Metrics:
    """Original contra reconstrucción, con el error de cada muestra.

    Opciones: `max_examples` (columnas de la figura) y `audio_examples`
    (cuántas muestras exportar como WAV, original por el códec y reconstruida).
    """
    max_examples = options.take("max_examples", 32)
    audio_examples = options.take("audio_examples", 4)
    options.ensure_consumed()

    dataset = context.dataset
    reconstructions = context.model.reconstruct(dataset.samples)
    errors = sample_errors(dataset, dataset.samples, reconstructions)
    write_csv(
        context.output_directory / "reconstruction.csv",
        [{"label": label, "error": float(error)} for label, error in zip(dataset.labels, errors)],
    )
    shown = _evenly_spaced_indexes(len(dataset), max_examples)
    plotting.save_image_rows(
        [dataset.samples[shown], context.displayable(reconstructions[shown])],
        ["Original", "Reconstrucción"],
        [_short(dataset.labels[index]) for index in shown],
        dataset.image_shape,
        dataset.is_binary,
        context.output_directory / "reconstruction.png",
    )
    for index in _evenly_spaced_indexes(len(dataset), audio_examples):
        name = safe_filename(dataset.labels[index])
        context.write_audio(f"audio/reconstruction/{name}__original.wav", dataset.samples[index])
        context.write_audio(f"audio/reconstruction/{name}__model.wav", reconstructions[index])
    return {}


def evaluate_latent(context: EvaluationContext, options: Options) -> Metrics:
    """Ubicación de cada muestra en el espacio latente.

    Con más de dos dimensiones latentes se grafican las dos componentes
    principales; el CSV siempre guarda todas las coordenadas.
    """
    options.ensure_consumed()
    dataset = context.dataset
    codes = context.model.encode(dataset.samples)
    write_csv(
        context.output_directory / "latent.csv",
        [
            {"label": label, "group": group, **{f"z{i + 1}": float(v) for i, v in enumerate(code)}}
            for label, group, code in zip(dataset.labels, dataset.groups, codes)
        ],
    )
    points, axis_names = _two_dimensional_view(codes)
    plotting.save_latent_scatter(
        points, dataset.labels, dataset.groups, axis_names, context.output_directory / "latent.png"
    )
    return {}


def evaluate_latent_grid(context: EvaluationContext, options: Options) -> Metrics:
    """Decodifica una grilla regular del plano latente (requiere latent_size 2).

    Muestra qué genera el decodificador entre y alrededor de los datos.
    Opciones: `size` (celdas por lado) y `margin` (fracción extra alrededor
    de la caja que contiene a los datos codificados).
    """
    size = options.take("size", 12)
    margin = options.take("margin", 0.1)
    options.ensure_consumed()
    if context.model.latent_size != 2:
        raise ConfigurationError("evaluation.latent_grid requiere model.latent_size igual a 2")

    codes = context.model.encode(context.dataset.samples)
    low, high = codes.min(axis=0), codes.max(axis=0)
    padding = margin * (high - low)
    first_axis = np.linspace(low[0] - padding[0], high[0] + padding[0], size)
    # z2 decrece hacia abajo para coincidir con la orientación del gráfico latente.
    second_axis = np.linspace(high[1] + padding[1], low[1] - padding[1], size)
    grid = np.array([[first, second] for second in second_axis for first in first_axis])
    plotting.save_image_mosaic(
        context.model.decode(grid),
        (size, size),
        context.dataset.image_shape,
        context.dataset.is_binary,
        f"z1 ∈ [{first_axis[0]:.2f}, {first_axis[-1]:.2f}] →,  "
        f"z2 ∈ [{second_axis[-1]:.2f}, {second_axis[0]:.2f}] ↑",
        context.output_directory / "latent_grid.png",
    )
    return {}


def _two_dimensional_view(codes: Array) -> tuple[Array, tuple[str, str]]:
    if codes.shape[1] == 1:
        return np.column_stack([codes[:, 0], np.zeros(len(codes))]), ("z1", "")
    if codes.shape[1] == 2:
        return codes, ("z1", "z2")
    centered = codes - codes.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    explained = singular_values**2 / np.sum(singular_values**2)
    names = tuple(f"PC{i + 1} ({explained[i]:.0%} de la varianza)" for i in range(2))
    return centered @ components[:2].T, names


def _evenly_spaced_indexes(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    return sorted(set(np.linspace(0, total - 1, min(count, total)).round().astype(int).tolist()))


def _short(label: str, limit: int = 14) -> str:
    return label if len(label) <= limit else label[:limit] + "…"
