"""Figuras del ejercicio 1: autoencoder y denoising autoencoder sobre font.h."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle

from scripts.presentacion.common import (
    PIXEL_TARGET, RESULTS, Array, bare, glyph, read_csv, read_summary, save,
)
from src.data.noise import Noise
from src.evaluation.metrics import binarize
from src.experiments.runner import load_run

BASELINE_RUN = RESULTS / "ej1a_baseline" / "base__seed=42"
DENOISING_RUN = RESULTS / "ej1b_denoising_latent_size" / "latent_size=8__seed=1"
SHAPE = (7, 5)
WRONG_PIXEL = to_rgb("C3")
NEW_LETTER_COUNT = 6
NEW_LETTER_MIN_DISTANCE = 3  # píxeles de diferencia con la letra más parecida del font
NEW_LETTER_MIN_CONFIDENCE = 0.25  # cada píxel a esta distancia de 0,5: la red "está segura"
NEW_LETTER_MAX_INK = 0.6  # descarta bloques casi llenos, parecidos a DEL
INTERPOLATION_PAIRS = [("a", "e"), ("o", "x"), ("l", "t")]
DENOISING_EXAMPLES = ["a", "e", "g", "k", "m", "s", "w", "z"]

OPTIMIZER_NAMES = {
    "optimizer=sgd,learning_rate=0.01": "SGD, η = 0,01",
    "optimizer=momentum,learning_rate=0.01,momentum=0.9": "Momentum, η = 0,01",
    "optimizer=rmsprop,learning_rate=0.001": "RMSProp, η = 0,001",
    "optimizer=adam,learning_rate=0.001": "Adam, η = 0,001",
    "optimizer=adam,learning_rate=0.01": "Adam, η = 0,01",
}
ACTIVATION_NAMES = {
    "activation=tanh": "tanh", "activation=sigmoid": "sigmoide",
    "activation=relu": "ReLU", "activation=leaky_relu": "leaky ReLU",
}


def make_all() -> None:
    model, dataset = load_run(BASELINE_RUN)
    font_figure(dataset.samples, dataset.labels)
    reconstruction_figure(model, dataset.samples, dataset.labels)
    success_bars("ej1a_architectures", _hidden_layers_name, "Capas ocultas del codificador", "e1_arquitecturas")
    success_bars("ej1a_latent_size", _latent_name, "Dimensión del espacio latente", "e1_latente_tamano")
    training_curves()
    latent_letters_figure(model, dataset.samples)
    decoded_map_figure(model, dataset.samples)
    new_letters_figure(model, dataset.samples, dataset.labels)
    interpolation_figure(model, dataset.samples, dataset.labels)
    denoising_examples_figure()
    denoising_levels_figure()
    denoising_latent_figure()


# ----------------------------------------------------------------------------- dataset y reconstrucción

def font_figure(samples: Array, labels: list[str]) -> None:
    figure, axes = plt.subplots(4, 8, figsize=(8, 6.4), gridspec_kw={"wspace": 0.45, "hspace": 0.55})
    for axis, pattern, label in zip(axes.flat, samples, labels):
        glyph(axis, pattern)
        axis.set_xlabel(label, fontsize=13)
    save(figure, "e1_font")


def reconstruction_figure(model, samples: Array, labels: list[str], per_band: int = 16) -> None:
    """Cada letra que entra, y debajo la que sale (binarizada), con los píxeles equivocados en rojo."""
    decoded = binarize(model.reconstruct(samples))
    bands = len(samples) // per_band
    figure = plt.figure(figsize=(0.95 * per_band, 3.4 * bands))
    for band, subfigure in enumerate(figure.subfigures(bands, 1, hspace=0.08)):
        inputs, outputs = subfigure.subplots(2, per_band, gridspec_kw={"wspace": 0.3, "hspace": 0.12})
        for column in range(per_band):
            index = band * per_band + column
            target, output = samples[index], decoded[index]
            glyph(inputs[column], target)
            image = np.repeat((1.0 - output)[:, None], 3, axis=1)
            image[target != output] = WRONG_PIXEL
            glyph(outputs[column], output)
            outputs[column].images[0].remove()
            outputs[column].imshow(image.reshape(*SHAPE, 3))
            outputs[column].set_xlabel(labels[index], fontsize=20)
        inputs[0].set_ylabel("entra", fontsize=24, rotation=0, ha="right", va="center", labelpad=12)
        outputs[0].set_ylabel("sale", fontsize=24, rotation=0, ha="right", va="center", labelpad=12)
    save(figure, "e1_reconstruccion")


# ----------------------------------------------------------------------------- qué hace falta para aprender

def success_bars(experiment: str, name_of, ylabel: str, output: str) -> None:
    """Época en que cada configuración llega a ≤ 1 píxel por letra: media ± desvío entre las semillas que llegan.

    Al lado, cuántas semillas llegan; las que no llegan no tienen época y no entran en la media.
    """
    groups = read_summary(experiment)
    names = [name_of(configuration) for configuration in groups]
    seeds = len(next(iter(groups.values())))
    epochs = [_epochs_within_target(rows) for rows in groups.values()]
    means = [np.mean(values) if values else 0.0 for values in epochs]
    deviations = [np.std(values) if len(values) > 1 else 0.0 for values in epochs]
    figure, axis = plt.subplots(figsize=(9, 0.62 * len(names) + 1.2))
    positions = np.arange(len(names))[::-1]
    axis.barh(positions, means, xerr=deviations, color="C0", capsize=4, error_kw={"lw": 1.4})
    for position, values in zip(positions, epochs):
        axis.scatter(values, np.full(len(values), position), color="black", s=12, zorder=3)
    right_edges = [max([mean + deviation, *values]) for mean, deviation, values in zip(means, deviations, epochs)]
    limit = max(right_edges) * 1.35
    for position, values, right_edge in zip(positions, epochs, right_edges):
        text = f"{len(values)}/{seeds} semillas" if values else f"0/{seeds}: no llega"
        axis.text(right_edge + 0.02 * limit, position, text, va="center", fontsize=12,
                  color="black" if values else "C3")
    axis.set_yticks(positions, names)
    axis.set_xlim(0, limit)
    axis.set_xlabel(f"época en que las 32 letras quedan con ≤ {PIXEL_TARGET} píxel mal\n"
                    "(barra: media ± desvío; puntos: cada semilla)")
    axis.set_ylabel(ylabel)
    save(figure, output)


def _epochs_within_target(rows: list[dict[str, str]]) -> list[int]:
    return [int(row["first_epoch_within_target"]) for row in rows
            if row["first_epoch_within_target"] and int(row["max_pixel_errors"]) <= PIXEL_TARGET]


def _hidden_layers_name(configuration: str) -> str:
    value = configuration.split("=", 1)[1]
    return "ninguna" if value == "none" else value


def _latent_name(configuration: str) -> str:
    return configuration.split("=", 1)[1]


def training_curves() -> None:
    """Letras aprendidas a lo largo del entrenamiento: media ± desvío de 5 semillas (banda)."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    _median_curves(left, "ej1a_optimizers", OPTIMIZER_NAMES)
    _median_curves(right, "ej1a_activations", ACTIVATION_NAMES)
    left.set_title("Optimizador (tanh)")
    right.set_title("Activación (Adam, η = 0,01)")
    left.set_ylabel("letras con ≤ 1 píxel mal\n(media ± desvío, 5 semillas)")
    for axis in (left, right):
        axis.set_xlabel("época")
        axis.set_ylim(0, 33)
        axis.axhline(32, color="C7", lw=0.8, ls=":")
        axis.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=11)
    save(figure, "e1_optimizacion")


def _median_curves(axis: plt.Axes, experiment: str, names: dict[str, str]) -> None:
    for configuration, rows in read_summary(experiment).items():
        histories = [read_csv(RESULTS / experiment / row["run"] / "history.csv") for row in rows]
        epochs = [int(entry["epoch"]) for entry in histories[0]]
        learned = np.array([[float(entry["patterns_within_target"]) for entry in history] for history in histories])
        mean, deviation = learned.mean(axis=0), learned.std(axis=0)
        (line,) = axis.plot(epochs, mean, label=names[configuration], lw=2)
        axis.fill_between(epochs, mean - deviation, np.minimum(mean + deviation, 32), color=line.get_color(),
                          alpha=0.15, linewidth=0)


# ----------------------------------------------------------------------------- espacio latente

def latent_letters_figure(model, samples: Array) -> None:
    """Cada letra dibujada en el lugar del plano latente donde la pone el codificador."""
    codes = model.encode(samples)
    figure, axis = plt.subplots(figsize=(12, 7.2))
    axis.scatter(codes[:, 0], codes[:, 1], s=6, color="C0", zorder=1)
    for code, pattern in zip(codes, samples):
        image = np.kron(pattern.reshape(SHAPE), np.ones((4, 4)))
        box = AnnotationBbox(
            OffsetImage(1.0 - image, cmap="gray", norm=Normalize(0, 1), zoom=1.35), code,
            frameon=True, pad=0.12, bboxprops={"edgecolor": "C0", "linewidth": 1.2},
        )
        axis.add_artist(box)
    _pad_limits(axis, codes, 0.08)
    axis.set_xlabel("z₁")
    axis.set_ylabel("z₂")
    save(figure, "e1_latente_letras")


def decoded_map_figure(model, samples: Array, cells: int = 18) -> None:
    """Lo que decodifica cada punto de una grilla del plano; en naranja, patrones que no son del font."""
    codes = model.encode(samples)
    first_axis, second_axis = _grid_axes(codes, cells, margin=0.08)
    points = np.array([[first, second] for second in second_axis[::-1] for first in first_axis])
    decoded = model.decode(points)
    novel = [_hamming_to_font(pattern, samples) >= 1 for pattern in binarize(decoded)]

    canvas = np.ones((cells * 8 - 1, cells * 6 - 1))
    for index, pattern in enumerate(decoded):
        row, column = divmod(index, cells)
        canvas[row * 8 : row * 8 + 7, column * 6 : column * 6 + 5] = 1.0 - pattern.reshape(SHAPE)
    figure, axis = plt.subplots(figsize=(9.5, 10.5))
    axis.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    for index, is_new in enumerate(novel):
        if is_new:
            row, column = divmod(index, cells)
            axis.add_patch(Rectangle((column * 6 - 0.5, row * 8 - 0.5), 5, 7, fill=False, edgecolor="C1", lw=1.4))
    axis.axis("off")
    share = np.mean(novel)
    print(f"    mapa decodificado: {share:.0%} de las celdas no son letras del font")
    save(figure, "e1_mapa_decodificado")


def new_letters_figure(model, samples: Array, labels: list[str]) -> None:
    """Patrones nítidos y lejos de toda letra del font, ordenados por cuánto plano ocupan."""
    codes = model.encode(samples)
    first_axis, second_axis = _grid_axes(codes, 160, margin=0.0)
    points = np.array([[first, second] for second in second_axis for first in first_axis])
    probabilities = model.decode(points)
    confident = np.min(np.abs(probabilities - 0.5), axis=1) >= NEW_LETTER_MIN_CONFIDENCE
    patterns = binarize(probabilities)
    areas: Counter[bytes] = Counter()
    for pattern, is_clear in zip(patterns, confident):
        if is_clear and _hamming_to_font(pattern, samples) >= NEW_LETTER_MIN_DISTANCE:
            areas[pattern.astype(np.uint8).tobytes()] += 1
    chosen: list[Array] = []
    nearest_used: set[int] = set()
    for key, _ in areas.most_common():
        pattern = np.frombuffer(key, dtype=np.uint8).astype(float)
        nearest = int(np.argmin(np.count_nonzero(samples != pattern, axis=1)))
        if pattern.mean() <= NEW_LETTER_MAX_INK and nearest not in nearest_used:
            chosen.append(pattern)
            nearest_used.add(nearest)
        if len(chosen) == NEW_LETTER_COUNT:
            break
    print(f"    letras nuevas nítidas: {len(areas)} patrones distintos")

    figure, axes = plt.subplots(2, len(chosen), figsize=(1.9 * len(chosen), 4.6), gridspec_kw={"wspace": 0.5})
    for column, pattern in enumerate(chosen):
        distances = np.count_nonzero(samples != pattern, axis=1)
        nearest = int(np.argmin(distances))
        glyph(axes[0, column], pattern)
        glyph(axes[1, column], samples[nearest])
        axes[1, column].set_xlabel(f"«{labels[nearest]}», a {distances[nearest]} px", fontsize=12)
    axes[0, 0].set_ylabel("nueva", fontsize=13)
    axes[1, 0].set_ylabel("más parecida", fontsize=13)
    save(figure, "e1_letras_nuevas")


def interpolation_figure(model, samples: Array, labels: list[str], steps: int = 9) -> None:
    """Recorridos en línea recta entre dos letras; en naranja, los pasos que no son letras del font."""
    codes = model.encode(samples)
    fractions = np.linspace(0, 1, steps)
    figure, axes = plt.subplots(len(INTERPOLATION_PAIRS), steps, figsize=(1.25 * steps, 2.2 * len(INTERPOLATION_PAIRS)))
    for row, (origin, destination) in enumerate(INTERPOLATION_PAIRS):
        start, end = codes[labels.index(origin)], codes[labels.index(destination)]
        decoded = binarize(model.decode(start + fractions[:, None] * (end - start)))
        for column, pattern in enumerate(decoded):
            axis = axes[row, column]
            glyph(axis, pattern)
            if _hamming_to_font(pattern, samples) >= 1:
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("C1")
                    spine.set_linewidth(3)
        axes[row, 0].set_ylabel(f"{origin} → {destination}", fontsize=14)
    for column, fraction in enumerate(fractions):
        axes[-1, column].set_xlabel(f"{fraction:.2f}".replace(".", ","), fontsize=11)
    save(figure, "e1_interpolacion")


def _grid_axes(codes: Array, cells: int, margin: float) -> tuple[Array, Array]:
    low, high = codes.min(axis=0), codes.max(axis=0)
    padding = margin * (high - low)
    return (np.linspace(low[0] - padding[0], high[0] + padding[0], cells),
            np.linspace(low[1] - padding[1], high[1] + padding[1], cells))


def _hamming_to_font(pattern: Array, samples: Array) -> int:
    return int(np.min(np.count_nonzero(samples != pattern, axis=1)))


def _pad_limits(axis: plt.Axes, points: Array, margin: float) -> None:
    low, high = points.min(axis=0), points.max(axis=0)
    padding = margin * (high - low)
    axis.set_xlim(low[0] - padding[0], high[0] + padding[0])
    axis.set_ylim(low[1] - padding[1], high[1] + padding[1])


# ----------------------------------------------------------------------------- denoising

def denoising_examples_figure() -> None:
    """Original, entrada con 10 % y 20 % de píxeles invertidos, y la salida de la red."""
    model, dataset = load_run(DENOISING_RUN)
    rng = np.random.default_rng(3)
    indexes = [dataset.labels.index(label) for label in DENOISING_EXAMPLES]
    clean = dataset.samples[indexes]
    rows = [("original", clean)]
    for level in (0.1, 0.2):
        noisy = Noise("flip", level).apply(clean, rng)
        rows += [(f"{level:.0%} ruido".replace("%", " %"), noisy), ("salida", binarize(model.reconstruct(noisy)))]
    figure, axes = plt.subplots(len(rows), len(indexes), figsize=(1.3 * len(indexes), 1.85 * len(rows)),
                             gridspec_kw={"wspace": 0.35, "hspace": 0.25})
    for row, (name, patterns) in enumerate(rows):
        for column, pattern in enumerate(patterns):
            glyph(axes[row, column], pattern)
        axes[row, 0].set_ylabel(name, fontsize=24, rotation=0, ha="right", va="center", labelpad=14)
    save(figure, "e1b_ejemplos")


def denoising_levels_figure() -> None:
    _denoising_curves("ej1b_denoising", "training.noise.level",
                      lambda value: f"entrenada con {float(value):.0%}", "e1b_niveles")


def denoising_latent_figure() -> None:
    _denoising_curves("ej1b_denoising_latent_size", "model.latent_size",
                      lambda value: f"latente {value}", "e1b_latente")


def _denoising_curves(experiment: str, parameter: str, name_of, output: str) -> None:
    """Píxeles mal por letra a la salida según el ruido de prueba; la línea punteada es no usar la red."""
    figure, axis = plt.subplots(figsize=(9, 5))
    input_errors: dict[float, list[float]] = {}
    for rows in read_summary(experiment).values():
        tables = [read_csv(RESULTS / experiment / row["run"] / "denoising.csv") for row in rows]
        levels = [float(entry["level"]) for entry in tables[0]]
        output_error = np.array([[float(entry["output_mean_error"]) for entry in table] for table in tables])
        axis.errorbar(levels, output_error.mean(axis=0), yerr=output_error.std(axis=0), marker="o", lw=2,
                      capsize=4, label=name_of(rows[0][parameter]))
        for index, level in enumerate(levels):
            input_errors.setdefault(level, []).extend(float(table[index]["input_mean_error"]) for table in tables)
    levels = sorted(input_errors)
    axis.plot(levels, [statistics.mean(input_errors[level]) for level in levels], "k--", lw=1.5,
              label="entrada, sin la red")
    axis.set_xlabel("probabilidad de invertir cada píxel en la prueba")
    seeds = len(next(iter(read_summary(experiment).values())))
    axis.set_ylabel(f"píxeles mal por letra\n(media ± desvío, {seeds} semillas)")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.legend(fontsize=11)
    save(figure, output)
