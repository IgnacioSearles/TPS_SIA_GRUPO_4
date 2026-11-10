"""Compone un tema techno de un minuto usando un VAE de audio ya entrenado.

Uso, desde TP5:
    python -m src.song results/ej2_vae_nsynth/base__seed=7

Todo timbre sale del decodificador del VAE. Cada instrumento del tema es un
punto (o un recorrido) del espacio latente:
  - bajo, lead y pad interpolan entre notas reales de distintas familias a lo
    largo del tema, de modo que el timbre se transforma compás a compás;
  - bombo, hi-hat y clap son sonidos decodificados a los que se les da forma
    con envolventes, filtros y barridos de altura.
El procesamiento posterior (envolventes, cambio de altura por remuestreo,
filtros, sidechain y mezcla) es DSP convencional; el VAE no genera ritmo.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from src.data.audio import write_waveform
from src.data.dataset import Dataset
from src.experiments.runner import load_run
from src.models.autoencoder import Autoencoder

Array = np.ndarray

TEMPO_BPM = 128
BARS = 32
STEPS_PER_BAR = 16
STEPS_PER_BEAT = 4
REFERENCE_PITCH = 60  # altura MIDI de las notas ancla; el resto se obtiene remuestreando
BASS_REFERENCE_PITCH = 48  # la más grave del dataset: menos estiramiento al bajar al registro del bajo
ONSET_THRESHOLD = 0.2  # fracción del pico a partir de la cual se considera que la nota empezó
MUSIC_GAIN = 1.7  # nivel de todo lo que no es bombo

# La menor: i - i - VI - VII, un acorde por compás.
BASS_ROOTS = [33, 33, 29, 31]
LEAD_CHORDS = [[57, 60, 64, 69], [57, 60, 64, 69], [53, 57, 60, 65], [55, 59, 62, 67]]
LEAD_PATTERN = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 2, 1, 3]
PAD_CHORDS = [[57, 60, 64], [57, 60, 64], [53, 57, 60], [55, 59, 62]]

# Secciones, como rangos de compases [inicio, fin).
INTRO = range(0, 8)
BUILD = range(8, 16)
BREAKDOWN = range(16, 20)
DROP = range(20, 28)
OUTRO = range(28, 32)

_NOTE_PITCH = re.compile(r"-(\d+)-\d+$")


@dataclass(frozen=True)
class Anchor:
    """Una nota real del dataset usada como extremo de una interpolación."""

    label: str
    latent: Array
    pitch: int


class LatentInstrument:
    """Convierte puntos del espacio latente en audio, con caché por punto."""

    def __init__(self, model: Autoencoder, dataset: Dataset) -> None:
        if dataset.audio_codec is None:
            raise ValueError("La corrida no es de audio: se necesita un dataset nsynth")
        self._model = model
        self._dataset = dataset
        self._codec = dataset.audio_codec
        self._latents = model.encode(dataset.samples)
        self._cache: dict[bytes, Array] = {}
        self.sample_rate = self._codec.sample_rate

    def anchor(self, family: str, pitch: int = REFERENCE_PITCH) -> Anchor:
        """La nota de `family` más cercana a `pitch`."""
        candidates = [
            (abs(_pitch_of(label) - pitch), index)
            for index, (label, group) in enumerate(zip(self._dataset.labels, self._dataset.groups))
            if group == family
        ]
        if not candidates:
            raise ValueError(f"El dataset no tiene notas de la familia '{family}'")
        _, index = min(candidates)
        label = self._dataset.labels[index]
        return Anchor(label, self._latents[index], _pitch_of(label))

    def brightest_anchor(self) -> Anchor:
        """La nota con más energía en las bandas agudas: materia prima de hi-hat y clap."""
        bands = self._dataset.image_shape[0]
        energy = self._dataset.samples.reshape(len(self._dataset), bands, -1).sum(axis=2)
        centroids = (energy * np.arange(bands)).sum(axis=1) / np.maximum(energy.sum(axis=1), 1e-9)
        index = int(np.argmax(centroids))
        label = self._dataset.labels[index]
        return Anchor(label, self._latents[index], _pitch_of(label))

    def render(self, latent: Array) -> Array:
        """Decodifica un punto latente a un segundo de audio."""
        key = np.round(latent, 6).tobytes()
        if key not in self._cache:
            spectrogram = self._model.decode(latent[None, :])[0]
            self._cache[key] = self._codec.decode(
                np.asarray(spectrogram, dtype=float).reshape(self._dataset.image_shape)
            )
        return self._cache[key]

    def morph(self, origin: Anchor, destination: Anchor, fraction: float, pitch: float) -> Array:
        """Audio del punto intermedio entre dos anclas, llevado a la altura `pitch`."""
        latent = origin.latent + fraction * (destination.latent - origin.latent)
        source_pitch = origin.pitch + fraction * (destination.pitch - origin.pitch)
        return repitch(trim_onset(self.render(latent)), pitch - source_pitch)


def trim_onset(waveform: Array) -> Array:
    """Descarta el silencio inicial, para que las notas cortas empiecen con sonido."""
    loud = np.flatnonzero(np.abs(waveform) >= ONSET_THRESHOLD * np.max(np.abs(waveform)))
    return waveform[loud[0] :] if len(loud) else waveform


def repitch(waveform: Array, semitones: float) -> Array:
    """Cambia la altura remuestreando (también cambia la duración)."""
    ratio = 2.0 ** (semitones / 12.0)
    positions = np.arange(int(len(waveform) / ratio)) * ratio
    return np.interp(positions, np.arange(len(waveform)), waveform)


def shaped(waveform: Array, length: int, decay_seconds: float, sample_rate: int) -> Array:
    """Recorta a `length` muestras con ataque corto y caída exponencial."""
    note = np.zeros(length)
    usable = min(length, len(waveform))
    note[:usable] = waveform[:usable]
    time = np.arange(length) / sample_rate
    attack = np.minimum(1.0, time / 0.003)
    release = np.minimum(1.0, (length - np.arange(length)) / (0.005 * sample_rate))
    return note * attack * np.exp(-time / decay_seconds) * release


def swept_kick(source: Array, sample_rate: int) -> Array:
    """Bombo: una nota grave leída con velocidad decreciente (barrido de altura hacia abajo)."""
    length = int(0.32 * sample_rate)
    time = np.arange(length) / sample_rate
    rate = 0.40 + 2.4 * np.exp(-time / 0.025)
    positions = np.cumsum(rate)
    body = np.interp(positions, np.arange(len(source)), source)
    return np.tanh(3.0 * body * np.exp(-time / 0.11))


def filtered(waveform: Array, band: tuple[float, float] | float, sample_rate: int) -> Array:
    """Pasa-altos si `band` es una frecuencia; pasa-banda si es un par. Normaliza a pico 1."""
    kind = "bandpass" if isinstance(band, tuple) else "highpass"
    result = sosfilt(butter(4, band, kind, fs=sample_rate, output="sos"), waveform)
    # El filtro descarta casi toda la energía de una nota tonal: se recupera el nivel.
    return result / max(float(np.max(np.abs(result))), 1e-9)


class Track:
    """Pista mono donde se suman sonidos en posiciones de la grilla de semicorcheas."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.step_samples = int(round(60.0 / TEMPO_BPM / STEPS_PER_BEAT * sample_rate))
        self.audio = np.zeros(BARS * STEPS_PER_BAR * self.step_samples)

    def add(self, bar: int, step: float, sound: Array, gain: float) -> None:
        start = int((bar * STEPS_PER_BAR + step) * self.step_samples)
        end = min(start + len(sound), len(self.audio))
        self.audio[start:end] += gain * sound[: end - start]


def compose(instrument: LatentInstrument) -> tuple[Array, list[str]]:
    rate = instrument.sample_rate
    drums, music = Track(rate), Track(rate)
    step = drums.step_samples

    bass = instrument.anchor("bass", BASS_REFERENCE_PITCH)
    guitar = instrument.anchor("guitar", BASS_REFERENCE_PITCH)
    keyboard, organ, brass = (instrument.anchor(f) for f in ("keyboard", "organ", "brass"))
    vocal, string = instrument.anchor("vocal"), instrument.anchor("string")
    bright = instrument.brightest_anchor()

    kick = swept_kick(trim_onset(instrument.render(bass.latent)), rate)
    noise = trim_onset(instrument.render(bright.latent))
    hat = shaped(filtered(noise, 3500.0, rate), step, 0.018, rate)
    open_hat = shaped(filtered(noise, 3500.0, rate), 2 * step, 0.07, rate)
    clap = shaped(filtered(noise, (900.0, 3000.0), rate), 2 * step, 0.05, rate)

    kick_bars = [*range(0, 16), *DROP, *OUTRO]
    for bar in kick_bars:
        for beat in range(4):
            drums.add(bar, beat * STEPS_PER_BEAT, kick, 1.0)
    for bar in [*range(2, 16), *DROP, *OUTRO]:
        for beat in range(4):
            music.add(bar, beat * STEPS_PER_BEAT + 2, open_hat if bar >= 8 else hat, 0.22)
        if bar >= 8:
            for sixteenth in range(STEPS_PER_BAR):
                music.add(bar, sixteenth, hat, 0.07)
    for bar in [*BUILD, *DROP]:
        for beat in (1, 3):
            music.add(bar, beat * STEPS_PER_BEAT, clap, 0.35)
    # Redoble que crece en el último compás del breakdown, antes del drop.
    for sixteenth in range(STEPS_PER_BAR):
        music.add(BREAKDOWN[-1], sixteenth, clap, 0.05 + 0.3 * sixteenth / STEPS_PER_BAR)

    # Bajo en las tres semicorcheas que deja libres el bombo; el timbre va de bajo a guitarra.
    bass_bars = [*range(4, 16), *DROP]
    for position, bar in enumerate(bass_bars):
        fraction = 0.4 * position / (len(bass_bars) - 1)
        root = BASS_ROOTS[bar % 4]
        note = shaped(instrument.morph(bass, guitar, fraction, root), step, 0.09, rate)
        for beat in range(4):
            for offbeat in (1, 2, 3):
                music.add(bar, beat * STEPS_PER_BEAT + offbeat, note, 0.7)

    # Lead arpegiado: teclado -> órgano en la subida y órgano -> metal en el drop,
    # con notas cada vez más largas (el arpegio "se abre").
    sections = ((BUILD, keyboard, organ, 0.30), (DROP, organ, brass, 0.42))
    for section, origin, destination, gain in sections:
        for position, bar in enumerate(section):
            fraction = position / (len(section) - 1)
            decay = 0.04 + 0.10 * fraction
            for sixteenth, chord_index in enumerate(LEAD_PATTERN):
                pitch = LEAD_CHORDS[bar % 4][chord_index]
                voice = instrument.morph(origin, destination, fraction, pitch)
                music.add(bar, sixteenth, shaped(voice, 2 * step, decay, rate), gain)

    # Breakdown sin bombo: acordes largos que pasan de voz a cuerdas.
    half_bar = STEPS_PER_BAR // 2
    pad_slots = [(bar, half) for bar in BREAKDOWN for half in (0, half_bar)]
    for position, (bar, start) in enumerate(pad_slots):
        fraction = position / (len(pad_slots) - 1)
        for pitch in PAD_CHORDS[bar % 4]:
            voice = instrument.morph(vocal, string, fraction, pitch)
            music.add(bar, start, shaped(voice, half_bar * step, 0.6, rate), 0.22)

    mix = drums.audio + MUSIC_GAIN * music.audio * _sidechain(drums, kick_bars)
    mix *= _fade_out(len(mix), OUTRO[-1] * STEPS_PER_BAR * step)
    mix = np.tanh(1.4 * mix / np.max(np.abs(mix)))
    notes = [
        f"bombo: {bass.label} con barrido de altura",
        f"hi-hat y clap: {bright.label} filtrado",
        f"bajo: {bass.label} -> {guitar.label} (hasta 40 %)",
        f"lead, subida: {keyboard.label} -> {organ.label}",
        f"lead, drop: {organ.label} -> {brass.label}",
        f"pad, breakdown: {vocal.label} -> {string.label}",
    ]
    return mix / np.max(np.abs(mix)) * 0.95, notes


def _sidechain(drums: Track, kick_bars: list[int]) -> Array:
    """Ganancia que baja el resto de la mezcla en cada golpe de bombo."""
    gain = np.ones(len(drums.audio))
    beat = STEPS_PER_BEAT * drums.step_samples
    duck = 1.0 - 0.7 * np.exp(-np.arange(beat) / (0.08 * drums.sample_rate))
    for bar in kick_bars:
        for beat_index in range(4):
            start = (bar * 4 + beat_index) * beat
            gain[start : start + beat] = duck
    return gain


def _fade_out(length: int, start: int) -> Array:
    envelope = np.ones(length)
    envelope[start:] = np.linspace(1.0, 0.0, length - start)
    return envelope


def _pitch_of(label: str) -> int:
    match = _NOTE_PITCH.search(label)
    if match is None:
        raise ValueError(f"La etiqueta '{label}' no tiene el formato de NSynth")
    return int(match.group(1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="carpeta de una corrida entrenada con nsynth")
    parser.add_argument("--output", type=Path, help="por defecto, <run>/song/techno.wav")
    arguments = parser.parse_args()
    output = arguments.output or arguments.run / "song" / "techno.wav"
    trained_model, training_dataset = load_run(arguments.run)
    latent_instrument = LatentInstrument(trained_model, training_dataset)
    song, timbre_notes = compose(latent_instrument)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_waveform(output, song, latent_instrument.sample_rate)
    print("\n".join(timbre_notes))
    print(f"{len(song) / latent_instrument.sample_rate:.1f} s -> {output}")
