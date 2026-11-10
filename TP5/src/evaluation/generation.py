"""Generación de muestras nuevas: interpolación latente y muestreo."""

from __future__ import annotations

import numpy as np

from src import plotting
from src.evaluation.context import EvaluationContext, Metrics, safe_filename, write_csv
from src.options import ConfigurationError, Options

Array = np.ndarray


def evaluate_interpolation(context: EvaluationContext, options: Options) -> Metrics:
    """Recorre en línea recta el espacio latente entre pares de muestras.

    Opciones: `pairs` (lista de [origen, destino], por etiqueta o índice),
    `random_pairs` (pares adicionales al azar, de grupos distintos si los hay)
    y `steps` (puntos por recorrido, extremos incluidos). Sin `pairs` se usan
    3 pares al azar. En datos binarios el CSV indica qué patrones no están en
    el dataset; en audio se exporta un WAV por paso y otro con el recorrido completo.
    """
    explicit_pairs = options.take("pairs", [])
    random_pairs = options.take("random_pairs", 0 if explicit_pairs else 3)
    steps = options.take("steps", 9)
    options.ensure_consumed()
    if steps < 2:
        raise ConfigurationError("evaluation.interpolation.steps debe ser al menos 2")

    dataset = context.dataset
    pairs = [_resolve_pair(context, pair) for pair in explicit_pairs]
    pairs += _random_pairs(context, random_pairs)
    if not pairs:
        raise ConfigurationError("evaluation.interpolation no tiene pares para recorrer")

    codes = context.model.encode(dataset.samples)
    fractions = np.linspace(0.0, 1.0, steps)
    rows, row_names, csv_rows = [], [], []
    for origin, destination in pairs:
        path = codes[origin] + fractions[:, None] * (codes[destination] - codes[origin])
        decoded = context.model.decode(path)
        pair_name = f"{dataset.labels[origin]} → {dataset.labels[destination]}"
        novel = context.is_novel(decoded) if dataset.is_binary else [False] * steps
        rows.append(context.displayable(decoded))
        # En dos líneas: las etiquetas largas (audio) le quitarían lugar a las imágenes.
        row_names.append(f"{dataset.labels[origin]}\n→ {dataset.labels[destination]}")
        csv_rows += [
            {"pair": pair_name, "fraction": float(fraction), "novel": is_new}
            for fraction, is_new in zip(fractions, novel)
        ]
        _write_interpolation_audio(context, pair_name, decoded)

    plotting.save_image_rows(
        rows,
        row_names,
        [f"{fraction:.2f}" for fraction in fractions],
        dataset.image_shape,
        dataset.is_binary,
        context.output_directory / "interpolation.png",
    )
    write_csv(context.output_directory / "interpolation.csv", csv_rows)
    if not dataset.is_binary:
        return {}
    return {"interpolation_novel_patterns": sum(row["novel"] for row in csv_rows)}


def evaluate_sampling(context: EvaluationContext, options: Options) -> Metrics:
    """Decodifica puntos latentes sorteados de N(0, I).

    Es el prior de un VAE, así que allí produce muestras plausibles. En un
    autoencoder común nada obliga al latente a parecerse a N(0, I): sirve
    para mostrar el problema que el esquema variacional resuelve.
    Opción: `count`.
    """
    count = options.take("count", 16)
    options.ensure_consumed()

    dataset = context.dataset
    decoded = context.model.decode(context.rng.standard_normal((count, context.model.latent_size)))
    columns = int(np.ceil(np.sqrt(count)))
    plotting.save_image_mosaic(
        context.displayable(decoded),
        (int(np.ceil(count / columns)), columns),
        dataset.image_shape,
        dataset.is_binary,
        "Muestras decodificadas desde z ~ N(0, I)",
        context.output_directory / "samples.png",
    )
    for index, sample in enumerate(decoded):
        context.write_audio(f"audio/samples/sample_{index:02d}.wav", sample)
    if not dataset.is_binary:
        return {}
    return {"sampling_novel_patterns": sum(context.is_novel(decoded))}


def _resolve_pair(context: EvaluationContext, pair: object) -> tuple[int, int]:
    if not isinstance(pair, list) or len(pair) != 2:
        raise ConfigurationError(f"Cada par de interpolación debe ser [origen, destino]: {pair!r}")
    return context.dataset.index_of(pair[0]), context.dataset.index_of(pair[1])


def _random_pairs(context: EvaluationContext, count: int) -> list[tuple[int, int]]:
    groups = np.asarray(context.dataset.groups)
    pairs = []
    for _ in range(count):
        origin = int(context.rng.integers(len(groups)))
        different_group = np.flatnonzero(groups != groups[origin])
        candidates = different_group if len(different_group) else np.flatnonzero(
            np.arange(len(groups)) != origin
        )
        if len(candidates) == 0:
            raise ConfigurationError("Se necesitan al menos dos muestras para interpolar")
        pairs.append((origin, int(context.rng.choice(candidates))))
    return pairs


def _write_interpolation_audio(context: EvaluationContext, pair_name: str, decoded: Array) -> None:
    if context.dataset.audio_codec is None:
        return
    folder = f"audio/interpolation/{safe_filename(pair_name.replace('→', 'to'))}"
    waveforms = [
        context.write_audio(f"{folder}/step_{step:02d}.wav", sample)
        for step, sample in enumerate(decoded)
    ]
    # El recorrido completo, un paso tras otro, para escuchar la transición.
    context.write_waveform(f"{folder}/sequence.wav", np.concatenate(waveforms))
