"""Diagramas de las redes: nodos y conexiones dibujados con matplotlib, con la entrada y la salida reales."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from scripts.presentacion.common import RESULTS, SPECTROGRAM_CMAP, Array, save
from src.data.noise import Noise
from src.evaluation.metrics import binarize
from src.experiments.runner import load_run

AUTOENCODER_RUN = RESULTS / "ej1a_baseline" / "base__seed=42"
DENOISING_RUN = RESULTS / "ej1b_denoising_latent_size" / "latent_size=8__seed=1"
VARIATIONAL_RUN = RESULTS / "ej2_vae_nsynth" / "base__seed=7"
EXAMPLE_LETTER = "a"
EXAMPLE_NOTE = "keyboard_electronic_098-060-100"
DENOISING_LEVEL = 0.1

MAX_NODES = 16  # capas más anchas se dibujan con una elipsis en el medio
NODE_SPACING = 1.0
LAYER_SPACING = 2.6
NODE_RADIUS = 0.3
EDGE_COLOR = "0.82"
HIDDEN_COLOR = "C0"
LATENT_COLOR = "C1"
TEXT_SIZE = 17


@dataclass(frozen=True)
class Layer:
    size: int
    activation: str = ""
    latent: bool = False
    shown: int | None = None  # nodos a dibujar; por defecto min(size, MAX_NODES)
    name: str = ""
    labeled: bool = True  # tamaño y nombre debajo de la capa

    @property
    def nodes(self) -> int:
        return self.shown if self.shown is not None else min(self.size, MAX_NODES)


def make_all() -> None:
    autoencoder_figure()
    denoising_figure()
    variational_figure()


# ----------------------------------------------------------------------------- piezas

def draw_layer(axis: plt.Axes, layer: Layer, x: float, center: float = 0.0) -> Array:
    """Dibuja los nodos de una capa y devuelve sus posiciones; si la capa es ancha, deja un hueco con ⋮."""
    count = layer.nodes
    truncated = count < layer.size
    gap = 1.2 * NODE_SPACING if truncated else 0.0
    offsets = np.arange(count) * NODE_SPACING
    if truncated:
        offsets[count // 2 :] += gap
    offsets = offsets - offsets.mean() + center
    color = LATENT_COLOR if layer.latent else HIDDEN_COLOR
    for y in offsets:
        axis.add_patch(plt.Circle((x, y), NODE_RADIUS, facecolor=color if layer.latent else "white",
                                  edgecolor=color, lw=1.8, zorder=3))
    if truncated:
        axis.text(x, center, "⋮", ha="center", va="center", fontsize=TEXT_SIZE + 4, color="0.4")
    if not layer.labeled:
        return np.column_stack([np.full(count, x), offsets])
    bottom = offsets.min() - 0.9
    axis.text(x, bottom, f"{layer.size:,}".replace(",", "."), ha="center", va="top",
              fontsize=TEXT_SIZE, fontweight="bold", color=color if layer.latent else "black")
    caption = layer.name or layer.activation
    if caption:
        axis.text(x, bottom - 1.1, caption, ha="center", va="top", fontsize=TEXT_SIZE - 3, color="0.35")
    return np.column_stack([np.full(count, x), offsets])


def connect_dense(axis: plt.Axes, left: Array, right: Array) -> None:
    for start in left:
        for end in right:
            axis.plot(*zip(start, end), color=EDGE_COLOR, lw=0.6, zorder=1)


def connect_pairwise(axis: plt.Axes, left: Array, right: Array, color: str = EDGE_COLOR) -> None:
    for start, end in zip(left, right):
        axis.plot(*zip(start, end), color=color, lw=1.2, zorder=1)


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], text: str = "",
          text_offset: tuple[float, float] = (0.0, 0.5), style: str = "-|>", color: str = "0.25") -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=22, lw=2, color=color, zorder=2))
    if text:
        middle = ((start[0] + end[0]) / 2 + text_offset[0], (start[1] + end[1]) / 2 + text_offset[1])
        axis.text(*middle, text, ha="center", va="bottom", fontsize=TEXT_SIZE - 2, color=color)


def picture(axis: plt.Axes, image: Array, center: tuple[float, float], height: float, cmap: str,
            origin: str = "upper", caption: str = "") -> tuple[float, float, float, float]:
    """Una imagen (letra o espectrograma) ubicada en coordenadas del diagrama; devuelve su extensión."""
    rows, columns = image.shape
    width = height * columns / rows
    extent = (center[0] - width / 2, center[0] + width / 2, center[1] - height / 2, center[1] + height / 2)
    axis.imshow(image, cmap=cmap, vmin=0, vmax=1, extent=extent, origin=origin, zorder=3,
                interpolation="nearest")
    axis.add_patch(plt.Rectangle((extent[0], extent[2]), width, height, fill=False, edgecolor="0.6", lw=1, zorder=4))
    if caption:
        axis.text(center[0], extent[2] - 0.6, caption, ha="center", va="top", fontsize=TEXT_SIZE - 1)
    return extent


def brace(axis: plt.Axes, left: float, right: float, y: float, text: str) -> None:
    axis.plot([left, left, right, right], [y - 0.35, y, y, y - 0.35], color="0.45", lw=1.5)
    axis.text((left + right) / 2, y + 0.3, text, ha="center", va="bottom", fontsize=TEXT_SIZE, color="0.3")


def draw_stack(axis: plt.Axes, layers: list[Layer], first_x: float, spacing: float = LAYER_SPACING) -> list[Array]:
    """Capas consecutivas totalmente conectadas, separadas LAYER_SPACING."""
    positions = [draw_layer(axis, layer, first_x + index * spacing) for index, layer in enumerate(layers)]
    for left, right in zip(positions, positions[1:]):
        connect_dense(axis, left, right)
    return positions


def _finish(axis: plt.Axes) -> None:
    axis.set_aspect("equal")
    axis.axis("off")
    axis.autoscale_view()


def _glyph_image(pattern: Array) -> Array:
    return np.reshape(pattern, (7, 5))


# ----------------------------------------------------------------------------- figuras

def autoencoder_figure() -> None:
    model, dataset = load_run(AUTOENCODER_RUN)
    letter = dataset.samples[dataset.labels.index(EXAMPLE_LETTER)]
    output = model.reconstruct(letter[None, :])[0]
    layers = [Layer(35, "entrada"), Layer(16, "tanh"), Layer(8, "tanh"), Layer(2, "lineal", latent=True),
              Layer(8, "tanh"), Layer(16, "tanh"), Layer(35, "sigmoide")]

    figure, axis = plt.subplots(figsize=(15, 7.2))
    positions = draw_stack(axis, layers, first_x=0.0, spacing=4.4)
    _letters_at_ends(axis, positions, letter, output, input_caption="entrada", output_caption="salida")
    top = max(position[:, 1].max() for position in positions) + 1.2
    brace(axis, positions[0][0, 0], positions[3][0, 0], top, "codificador")
    brace(axis, positions[3][0, 0], positions[-1][0, 0], top, "decodificador")
    latent = positions[3]
    axis.text(latent[0, 0] + 0.55, latent[:, 1].max() + 0.9, "(z₁, z₂)", ha="center", fontsize=TEXT_SIZE,
              color=LATENT_COLOR, fontweight="bold")
    _finish(axis)
    save(figure, "e1_red")


def _letters_at_ends(axis: plt.Axes, positions: list[Array], letter: Array, output: Array,
                     input_caption: str, output_caption: str, distance: float = 3.2) -> tuple[tuple, tuple]:
    first_x, last_x = positions[0][0, 0], positions[-1][0, 0]
    left = picture(axis, _glyph_image(letter), (first_x - distance, 0.0), 5.6, "gray_r", caption=input_caption)
    right = picture(axis, _glyph_image(output), (last_x + distance, 0.0), 5.6, "gray_r", caption=output_caption)
    arrow(axis, (left[1] + 0.2, 0.0), (first_x - 1.0, 0.0))
    arrow(axis, (last_x + 1.0, 0.0), (right[0] - 0.2, 0.0))
    return left, right


def denoising_figure() -> None:
    """Entrenamiento del denoising: entra la letra con ruido y la pérdida la compara con la limpia."""
    model, dataset = load_run(DENOISING_RUN)
    clean = dataset.samples[dataset.labels.index(EXAMPLE_LETTER)]
    noisy = Noise("flip", DENOISING_LEVEL).apply(clean[None, :], np.random.default_rng(4))[0]
    output = model.reconstruct(noisy[None, :])[0]
    layers = [Layer(35), Layer(32, shown=14), Layer(16, shown=14), Layer(8, latent=True),
              Layer(16, shown=14), Layer(32, shown=14), Layer(35)]

    figure, axis = plt.subplots(figsize=(15, 8.4))
    positions = draw_stack(axis, layers, first_x=0.0, spacing=3.8)
    left, right = _letters_at_ends(axis, positions, noisy, output,
                                   input_caption="con 10 % de ruido", output_caption="salida",
                                   distance=4.4)
    height = left[3] - left[2]
    above = left[3] + 3.6 + height / 2
    clean_left = picture(axis, _glyph_image(clean), ((left[0] + left[1]) / 2, above), 5.6, "gray_r",
                         caption="")
    clean_right = picture(axis, _glyph_image(clean), ((right[0] + right[1]) / 2, above), 5.6, "gray_r",
                          caption="")
    axis.text((clean_left[0] + clean_left[1]) / 2, clean_left[3] + 0.4, "limpia", ha="center", va="bottom",
              fontsize=TEXT_SIZE - 1)
    axis.text((clean_right[0] + clean_right[1]) / 2, clean_right[3] + 0.4, "objetivo: la limpia", ha="center",
              va="bottom", fontsize=TEXT_SIZE - 1, color="C3")
    arrow(axis, ((left[0] + left[1]) / 2, clean_left[2] - 0.2), ((left[0] + left[1]) / 2, left[3] + 0.3))
    axis.text(left[0] - 0.4, (clean_left[2] + left[3]) / 2, "invertir\npíxeles\nal azar", ha="right",
              va="center", fontsize=TEXT_SIZE - 3, color="0.35")
    arrow(axis, ((right[0] + right[1]) / 2, right[3] + 0.3), ((right[0] + right[1]) / 2, clean_right[2] - 0.2),
          style="<|-|>", color="C3")
    axis.text(right[1] + 0.4, (clean_right[2] + right[3]) / 2, "pérdida", ha="left", va="center",
              fontsize=TEXT_SIZE - 1, color="C3")
    _finish(axis)
    save(figure, "e1b_red")


def variational_figure() -> None:
    """VAE: el codificador da μ y log σ², z = μ + σ·ε, y el decodificador vuelve al espectrograma."""
    model, dataset = load_run(VARIATIONAL_RUN)
    note = dataset.samples[dataset.labels.index(EXAMPLE_NOTE)]
    shape = dataset.image_shape
    output = model.reconstruct(note[None, :])[0]

    figure, axis = plt.subplots(figsize=(16, 8.4))
    encoder = draw_stack(axis, [Layer(8192, "entrada", shown=12), Layer(512, "ReLU", shown=12),
                                Layer(128, "ReLU", shown=12)], first_x=0.0)
    heads_x = encoder[-1][0, 0] + 1.4 * LAYER_SPACING
    head = Layer(16, latent=True, shown=4, labeled=False)
    mean = draw_layer(axis, head, heads_x, center=3.6)
    deviation = draw_layer(axis, head, heads_x, center=-3.6)
    _latent_label(axis, heads_x, mean[:, 1].max() + 0.7, "μ: 16", above=True)
    _latent_label(axis, heads_x, deviation[:, 1].min() - 0.7, "log σ²: 16", above=False)
    connect_dense(axis, encoder[-1], mean)
    connect_dense(axis, encoder[-1], deviation)

    latent_x = heads_x + 1.4 * LAYER_SPACING
    latent = draw_layer(axis, head, latent_x)
    _latent_label(axis, latent_x, latent[:, 1].max() + 0.7, "z: 16", above=True)
    connect_pairwise(axis, mean, latent, color=LATENT_COLOR)
    connect_pairwise(axis, deviation, latent, color=LATENT_COLOR)
    noise_y = latent[:, 1].min() - 3.0
    axis.text(latent_x, noise_y, "ε ~ N(0, I)", ha="center", va="center", fontsize=TEXT_SIZE, color="0.3",
              bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.6"})
    arrow(axis, (latent_x, noise_y + 0.8), (latent_x, latent[:, 1].min() - 0.5))
    _latent_label(axis, latent_x, deviation[:, 1].min() - 2.2, "z = μ + σ · ε", above=False)

    decoder = draw_stack(axis, [Layer(128, "ReLU", shown=12), Layer(512, "ReLU", shown=12),
                                Layer(8192, "sigmoide", shown=12)], first_x=latent_x + 1.4 * LAYER_SPACING)
    connect_dense(axis, latent, decoder[0])
    first_x, last_x = encoder[0][0, 0], decoder[-1][0, 0]
    left = picture(axis, note.reshape(shape), (first_x - 4.0, 0.0), 8.0, SPECTROGRAM_CMAP, origin="lower",
                   caption="entrada")
    right = picture(axis, output.reshape(shape), (last_x + 4.0, 0.0), 8.0, SPECTROGRAM_CMAP, origin="lower",
                    caption="salida")
    arrow(axis, (left[1] + 0.2, 0.0), (first_x - 1.0, 0.0))
    arrow(axis, (last_x + 1.0, 0.0), (right[0] - 0.2, 0.0))
    top = mean[:, 1].max() + 2.2
    brace(axis, first_x, heads_x, top, "codificador")
    brace(axis, decoder[0][0, 0], last_x, top, "decodificador")
    _finish(axis)
    save(figure, "e2_red")


def _latent_label(axis: plt.Axes, x: float, y: float, text: str, above: bool) -> None:
    axis.text(x, y, text, ha="center", va="bottom" if above else "top", fontsize=TEXT_SIZE,
              fontweight="bold", color=LATENT_COLOR)