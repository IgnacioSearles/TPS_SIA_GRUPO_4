"""Figuras y audios del ejercicio 2: autoencoder y VAE sobre notas de NSynth.

Cada espectrograma que se escucha en el deck se guarda como una imagen suelta
(reports/presentacion/espectrogramas/) con su WAV homónimo (audio/), para que
la diapositiva ponga el rótulo y el botón de reproducción debajo de cada uno.
"""

from __future__ import annotations

import io
import re
import shutil
import tarfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

from scripts.presentacion.common import (
    AUDIO, RESULTS, ROOT, Array, mean_and_interval, read_csv, save, save_spectrogram_tile, spectrogram,
)
from src.data.audio import SpectrogramCodec, read_waveform, write_waveform
from src.experiments.runner import load_run

AE_RUN = RESULTS / "ej2_ae_nsynth" / "base__seed=7"
VAE_RUN = RESULTS / "ej2_vae_nsynth" / "base__seed=7"
SONG = VAE_RUN / "song" / "techno.wav"
ARCHIVE = ROOT / "data" / "nsynth" / "nsynth-test.jsonwav.tar.gz"

FAMILIES = ["bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "vocal"]
FAMILY_NAMES = {"bass": "bajo", "brass": "metal", "flute": "flauta", "guitar": "guitarra", "keyboard": "teclado",
                "mallet": "percusión", "organ": "órgano", "reed": "lengüeta", "string": "cuerdas", "vocal": "voz"}
FAMILY_COLORS = {family: f"C{index}" for index, family in enumerate(FAMILIES)}
REPRESENTATION_NOTE = "keyboard_electronic_098-060-100"
RECONSTRUCTION_NOTES = ["brass_acoustic_016-064-100", "guitar_electronic_022-063-025", "reed_acoustic_023-049-050"]
# Misma altura (MIDI 60) y timbres opuestos: una nota que se apaga contra una sostenida.
INTERPOLATION_PAIRS = [
    ("guitar_acoustic_015-060-050", "organ_electronic_028-060-075"),
    ("bass_electronic_018-060-075", "vocal_synthetic_003-060-100"),
]
INTERPOLATION_STEPS = 8
SAMPLE_COUNT = 6
SAMPLE_SEED = 11
DISTANCE_SAMPLES = 500
DISTANCE_SEED = 0
PRIOR_POINTS = 1500
_PITCH = re.compile(r"-(\d+)-\d+$")


class NoteRenderer:
    """Espectrograma -> imagen suelta + WAV, con la misma decodificación (mel NNLS + Griffin-Lim)."""

    def __init__(self, codec: SpectrogramCodec) -> None:
        self._codec = codec

    def write(self, spectrogram_values: Array, name: str) -> None:
        image = np.asarray(spectrogram_values, dtype=float).reshape(self._codec.image_shape)
        save_spectrogram_tile(image, name)
        AUDIO.mkdir(parents=True, exist_ok=True)
        write_waveform(AUDIO / f"{name}.wav", self._codec.decode(image), self._codec.sample_rate)

    def waveform(self, spectrogram_values: Array) -> Array:
        return self._codec.decode(np.asarray(spectrogram_values, dtype=float).reshape(self._codec.image_shape))


def make_all() -> None:
    autoencoder, dataset = load_run(AE_RUN)
    variational, _ = load_run(VAE_RUN)
    renderer = NoteRenderer(dataset.audio_codec)
    labels = dataset.labels
    family_examples(dataset, renderer)
    representation_figure(dataset, renderer)
    latent_spaces_figure(autoencoder, variational, dataset)
    prior_samples(autoencoder, variational, dataset, renderer)
    prior_distance_figure(autoencoder, variational, dataset)
    reconstructions(autoencoder, variational, dataset, renderer)
    reconstruction_error_by_family_figure()
    for index, (origin, destination) in enumerate(INTERPOLATION_PAIRS, start=1):
        interpolation(variational, dataset, renderer, labels.index(origin), labels.index(destination), index)
    song_figure()


# ----------------------------------------------------------------------------- dataset y representación

def family_examples(dataset, renderer: NoteRenderer) -> None:
    """Una nota por familia, la más cercana al do central (MIDI 60)."""
    for family in FAMILIES:
        candidates = [index for index, group in enumerate(dataset.groups) if group == family]
        index = min(candidates, key=lambda candidate: abs(_pitch(dataset.labels[candidate]) - 60))
        renderer.write(dataset.samples[index], f"familia_{family}")
        print(f"    familia {family}: {dataset.labels[index]}")


def representation_figure(dataset, renderer: NoteRenderer) -> None:
    """Onda -> espectro lineal -> 128 bandas mel en [0, 1]; con el audio original y el pasado por el códec."""
    codec = dataset.audio_codec
    waveform = _read_original(REPRESENTATION_NOTE, codec)
    AUDIO.mkdir(parents=True, exist_ok=True)
    write_waveform(AUDIO / "representacion_original.wav", waveform, codec.sample_rate)
    encoded = codec.encode(waveform)
    write_waveform(AUDIO / "representacion_codec.wav", codec.decode(encoded), codec.sample_rate)

    frequencies, times, spectrum = stft(waveform, fs=codec.sample_rate, nperseg=codec.fft_size,
                                        noverlap=codec.fft_size - codec.hop_length)
    decibels = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-10))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    time = np.arange(len(waveform)) / codec.sample_rate
    axes[0].plot(time, waveform, lw=0.4, color="C0")
    axes[0].set_xlabel("tiempo (s)")
    axes[0].set_title("onda: 16.000 muestras")
    axes[1].pcolormesh(times, frequencies, decibels, vmin=decibels.max() - 70, vmax=decibels.max(),
                       cmap="magma", shading="auto")
    axes[1].set_ylim(0, 4000)
    axes[1].set_xlabel("tiempo (s)")
    axes[1].set_ylabel("Hz")
    axes[1].set_title("espectro: 513 frecuencias")
    spectrogram(axes[2], encoded)
    axes[2].set_title("mel: 128 bandas × 64 cuadros")
    save(figure, "e2_representacion")


def _read_original(label: str, codec: SpectrogramCodec) -> Array:
    with tarfile.open(ARCHIVE, "r|gz") as stream:
        for member in stream:
            if member.name.endswith(f"/{label}.wav"):
                raw = read_waveform(io.BytesIO(stream.extractfile(member).read()), codec.sample_rate)
                return raw[: codec.sample_count]
    raise FileNotFoundError(f"{label} no está en {ARCHIVE}")


# ----------------------------------------------------------------------------- AE contra VAE

def latent_spaces_figure(autoencoder, variational, dataset) -> None:
    """Códigos de cada modelo (PCA a 2D) junto a muestras de N(0, I), y sus normas."""
    rng = np.random.default_rng(0)
    prior = rng.standard_normal((PRIOR_POINTS, variational.latent_size))
    codes = {"Autoencoder": autoencoder.encode(dataset.samples), "VAE": variational.encode(dataset.samples)}

    figure, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=True, sharey=True)
    for axis, (name, latent) in zip(axes, codes.items()):
        center = latent.mean(axis=0)
        _, _, components = np.linalg.svd(latent - center, full_matrices=False)
        project = lambda points: (points - center) @ components[:2].T  # noqa: E731
        axis.scatter(*project(prior).T, s=5, color="0.75", label="z ~ N(0, I)")
        projected = project(latent)
        for family in FAMILIES:
            mask = np.array([group == family for group in dataset.groups])
            axis.scatter(*projected[mask].T, s=9, color=FAMILY_COLORS[family], label=FAMILY_NAMES[family])
        axis.set_title(name)
        axis.set_xlabel("PC1")
        axis.set_aspect("equal")
    axes[0].set_ylabel("PC2")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=2.5, fontsize=11)
    save(figure, "e2_latentes")

    figure, axis = plt.subplots(figsize=(9, 4.6))
    bins = np.linspace(0, 12, 61)
    axis.hist(np.linalg.norm(prior, axis=1), bins=bins, color="0.6", alpha=0.7, label="z ~ N(0, I)")
    for color, (name, latent) in zip(("C3", "C0"), codes.items()):
        norms = np.linalg.norm(latent, axis=1)
        axis.hist(norms, bins=bins, histtype="step", lw=2.2, color=color, label=f"{name} (media {norms.mean():.1f})")
    axis.set_xlabel("distancia del código de cada nota al origen (16 dimensiones)")
    axis.set_ylabel("notas")
    axis.legend()
    save(figure, "e2_normas")


def prior_samples(autoencoder, variational, dataset, renderer: NoteRenderer) -> None:
    """Los mismos z ~ N(0, I) por los dos decodificadores, y la nota real más parecida a cada muestra del VAE."""
    latent = np.random.default_rng(SAMPLE_SEED).standard_normal((SAMPLE_COUNT, variational.latent_size))
    for name, model in (("ae", autoencoder), ("vae", variational)):
        for index, values in enumerate(model.decode(latent)):
            renderer.write(values, f"muestra_{name}_{index}")
    generated = variational.decode(latent)
    for index, values in enumerate(generated):
        distances = np.mean((dataset.samples - values) ** 2, axis=1)
        nearest = int(np.argmin(distances))
        renderer.write(dataset.samples[nearest], f"cercana_{index}")
        print(f"    muestra {index}: más cercana {dataset.labels[nearest]} (MSE {distances[nearest]:.4f})")


def prior_distance_figure(autoencoder, variational, dataset) -> None:
    """Qué tan lejos cae cada muestra al azar de la nota real más parecida, contra dos notas reales vecinas."""
    latent = np.random.default_rng(DISTANCE_SEED).standard_normal((DISTANCE_SAMPLES, variational.latent_size))
    rows = [
        (f"autoencoder\n{DISTANCE_SAMPLES} puntos al azar", "C3",
         nearest_distances(dataset.samples, autoencoder.decode(latent))),
        (f"VAE\n{DISTANCE_SAMPLES} puntos al azar", "C0", nearest_distances(dataset.samples, variational.decode(latent))),
        ("referencia: una nota real\ny su vecina más parecida", "0.45",
         real_neighbor_distances(dataset.samples, DISTANCE_SAMPLES)),
    ]
    figure, axis = plt.subplots(figsize=(10, 4.4))
    jitter = np.random.default_rng(0)
    for position, (label, color, distances) in enumerate(rows):
        scaled = 1000 * distances  # milésimas: ticks cortos
        axis.scatter(scaled, position + jitter.uniform(-0.18, 0.18, len(scaled)), s=14, alpha=0.45, color=color,
                     edgecolors="none")
        median = np.median(scaled)
        axis.plot([median, median], [position - 0.3, position + 0.3], color="black", lw=2.5)
        axis.text(median, position + 0.34, f"mediana {median:.0f}", ha="center", va="bottom", fontsize=11)
        print(f"    {label.splitlines()[0]}: mediana MSE {median / 1000:.4f}")
    axis.set_yticks(range(len(rows)), [label for label, _, _ in rows])
    axis.set_ylim(-0.5, len(rows) - 0.2)
    axis.set_xlim(left=0)
    axis.set_xlabel("diferencia con la nota real más parecida del dataset (MSE × 1000)")
    save(figure, "e2_distancia_muestras")


def nearest_distances(samples: Array, outputs: Array) -> Array:
    return np.array([np.mean((samples - values) ** 2, axis=1).min() for values in outputs])


def typical_nearest_distance(samples: Array) -> float:
    return float(np.median(real_neighbor_distances(samples)))


def real_neighbor_distances(samples: Array, count: int = 300) -> Array:
    """Para notas reales al azar, la distancia a la nota real más parecida que no sea ella misma."""
    chosen = np.random.default_rng(1).choice(len(samples), count, replace=False)
    nearest = []
    for index in chosen:
        distances = np.mean((samples - samples[index]) ** 2, axis=1)
        distances[index] = np.inf
        nearest.append(distances.min())
    return np.array(nearest)


def reconstructions(autoencoder, variational, dataset, renderer: NoteRenderer) -> None:
    for note_index, label in enumerate(RECONSTRUCTION_NOTES):
        original = dataset.samples[dataset.labels.index(label)]
        renderer.write(original, f"reconstruccion_{note_index}_original")
        for name, model in (("ae", autoencoder), ("vae", variational)):
            output = model.reconstruct(original[None, :])[0]
            renderer.write(output, f"reconstruccion_{note_index}_{name}")
            print(f"    {label} {name}: MSE {np.mean((output - original) ** 2):.4f}")


def reconstruction_error_by_family_figure() -> None:
    """Versión chica para la diapositiva: media por familia con su intervalo de 95 % entre notas."""
    errors = {name: _reconstruction_errors(run) for name, run in (("AE", AE_RUN), ("VAE", VAE_RUN))}
    summary = {name: [mean_and_interval([error for label, error in by_label.items()
                                         if label.startswith(f"{family}_")]) for family in FAMILIES]
               for name, by_label in errors.items()}
    order = np.argsort([mean for mean, _ in summary["VAE"]])
    positions = np.arange(len(FAMILIES))
    figure, axis = plt.subplots(figsize=(5.4, 5.6))
    scaled = {name: 1000 * np.array(values)[order] for name, values in summary.items()}  # milésimas: ticks cortos
    for offset, name, color in ((0.14, "AE", "C3"), (-0.14, "VAE", "C0")):
        axis.errorbar(scaled[name][:, 0], positions + offset, xerr=scaled[name][:, 1], fmt="o", markersize=7,
                      color=color, capsize=3, label=name, zorder=2)
    axis.set_yticks(positions, [FAMILY_NAMES[FAMILIES[index]] for index in order])
    axis.set_xlabel("error de reconstrucción (MSE × 1000)\nmedia e intervalo de 95 % entre notas")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.1f}".replace(".", ",")))
    axis.legend(loc="lower right")
    save(figure, "e2_error_familias_chico")


def _reconstruction_errors(run) -> dict[str, float]:
    return {row["label"]: float(row["error"]) for row in read_csv(run / "reconstruction.csv")}


# ----------------------------------------------------------------------------- generación

def interpolation(variational, dataset, renderer: NoteRenderer, origin: int, destination: int, number: int) -> None:
    """Recorrido en línea recta entre dos notas: cada paso como imagen y WAV, más el recorrido entero."""
    codes = variational.encode(dataset.samples)
    fractions = np.linspace(0, 1, INTERPOLATION_STEPS)
    path = codes[origin] + fractions[:, None] * (codes[destination] - codes[origin])
    decoded = variational.decode(path)
    pieces = []
    for step, values in enumerate(decoded):
        name = f"interpolacion_{number}_{step}"
        renderer.write(values, name)
        pieces += [renderer.waveform(values) * 0.9, np.zeros(dataset.audio_codec.sample_rate // 8)]
    write_waveform(AUDIO / f"interpolacion_{number}_recorrido.wav", np.concatenate(pieces),
                   dataset.audio_codec.sample_rate)
    _latent_path_figure(codes, dataset, path, number)
    print(f"    interpolación {number}: {dataset.labels[origin]} -> {dataset.labels[destination]}")


def _latent_path_figure(codes: Array, dataset, path: Array, number: int) -> None:
    center = codes.mean(axis=0)
    _, _, components = np.linalg.svd(codes - center, full_matrices=False)
    projected = (codes - center) @ components[:2].T
    route = (path - center) @ components[:2].T
    figure, axis = plt.subplots(figsize=(5, 5))
    for family in FAMILIES:
        mask = np.array([group == family for group in dataset.groups])
        axis.scatter(*projected[mask].T, s=5, color=FAMILY_COLORS[family], alpha=0.35)
    axis.plot(*route.T, "-o", color="black", lw=2, ms=5)
    axis.scatter(*route[[0, -1]].T, s=90, color="black", zorder=3)
    axis.set_aspect("equal")
    axis.axis("off")
    save(figure, f"e2_camino_{number}")


def song_figure() -> None:
    """Espectrograma del tema techno compuesto con el decodificador del VAE."""
    if not SONG.is_file():
        raise FileNotFoundError(f"Falta {SONG}: python -m src.song {VAE_RUN.relative_to(ROOT)}")
    shutil.copy2(SONG, AUDIO / "tema.wav")
    rate, data = wavfile.read(SONG)
    mono = data.astype(float) if data.ndim == 1 else data.astype(float).mean(axis=1)
    frequencies, times, spectrum = stft(mono, fs=rate, nperseg=2048, noverlap=1536)
    decibels = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    figure, axis = plt.subplots(figsize=(14, 4))
    axis.pcolormesh(times, frequencies, decibels, vmin=decibels.max() - 80, vmax=decibels.max(),
                    cmap="magma", shading="auto")
    axis.set_yscale("symlog", linthresh=200)
    axis.set_ylim(30, rate / 2)
    axis.set_xlabel("tiempo (s)")
    axis.set_ylabel("Hz")
    save(figure, "e2_tema")


def _pitch(label: str) -> int:
    match = _PITCH.search(label)
    if match is None:
        raise ValueError(f"Etiqueta sin altura: {label}")
    return int(match.group(1))
