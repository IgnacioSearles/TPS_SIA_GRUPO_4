"""Escala cromática reproducible, sin aumentos que revelen las notas omitidas.

Opcionalmente agrega alturas microtonales (MIDI fraccionario) entre semitonos para
que el modelo vea el pitch como una magnitud continua. Las alturas cercanas a los
objetivos reservados se excluyen con una ventana (ver StudySettings.exclusion_radius).
"""

from itertools import product

import numpy as np

from src.data.audio import SpectrogramCodec
from src.data.notes import AudioNote, StudySettings, assemble_notes, midi_frequency, midi_list
from src.options import ConfigurationError

ENVELOPES = list(product((0.01, 0.03, 0.07, 0.12), (0.05, 0.15, 0.30, 0.45, 0.60), (0.8, 1.0)))


def harmonic_note(pitch: float, codec: SpectrogramCodec, *, attack_seconds=0.05,
                  release_seconds=0.05, duration_fraction=1.0) -> np.ndarray:
    time = np.arange(codec.sample_count) / codec.sample_rate
    frequency = midi_frequency(pitch)
    if 6 * frequency >= codec.sample_rate / 2:
        raise ConfigurationError("Los seis armónicos deben quedar debajo de Nyquist")
    waveform = sum(np.sin(2 * np.pi * frequency * k * time) / k for k in range(1, 7))
    length = round(codec.sample_count * duration_fraction)
    attack = min(round(attack_seconds * codec.sample_rate), length // 2)
    release = min(round(release_seconds * codec.sample_rate), length - attack)
    envelope = np.zeros(codec.sample_count)
    envelope[:length] = 1
    if attack:
        envelope[:attack] = np.linspace(0, 1, attack)
    if release:
        envelope[length - release:length] = np.linspace(1, 0, release)
    waveform *= envelope
    return waveform / np.max(np.abs(waveform))


def note_label(pitch: float, variant: str) -> str:
    """Las alturas enteras conservan la etiqueta histórica (manifiestos comparables)."""
    if float(pitch).is_integer():
        return f"harmonic-{int(pitch):03d}-{variant}"
    return f"harmonic-{pitch:06.2f}-{variant}"


def microtone_variant_indices(step: int, count: int, total: int = len(ENVELOPES)) -> list[int]:
    """Subconjunto equiespaciado de envolventes, rotado por microtono para cubrirlas todas."""
    stride = total // count
    return [(step - 1) % stride + stride * j for j in range(count)]


def pitch_schedule(bounds, microtones_per_semitone, microtone_variants):
    """Pares (altura, índice de envolvente). Enteras: las 40; microtonales: un subconjunto."""
    schedule = []
    for pitch in range(bounds[0], bounds[1] + 1):
        schedule.extend((pitch, index) for index in range(len(ENVELOPES)))
        if pitch == bounds[1]:
            break
        for step in range(1, microtones_per_semitone):
            fractional = round(pitch + step / microtones_per_semitone, 4)
            schedule.extend((fractional, index) for index in
                            microtone_variant_indices(step, microtone_variants))
    return schedule


def load_synthetic_notes(codec, pitch_range=(48, 72), exclude_pitches=(), settings=None,
                         variants_per_pitch=1, microtones_per_semitone=1, microtone_variants=10):
    bounds = midi_list(pitch_range, "pitch_range")
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ConfigurationError("pitch_range debe ser [mínimo, máximo], con mínimo < máximo")
    if type(variants_per_pitch) is not int or variants_per_pitch not in (1, 40):
        raise ConfigurationError("variants_per_pitch debe ser 1 (control original) o 40")
    if type(microtones_per_semitone) is not int or not 1 <= microtones_per_semitone <= 10:
        raise ConfigurationError("microtones_per_semitone debe ser un entero entre 1 y 10")
    if (type(microtone_variants) is not int or microtone_variants < 1
            or len(ENVELOPES) % microtone_variants):
        raise ConfigurationError("microtone_variants debe dividir a 40 (1, 2, 4, 5, 8, 10, 20, 40)")
    if variants_per_pitch == 1:
        if microtones_per_semitone != 1:
            raise ConfigurationError("Los microtonos requieren variants_per_pitch = 40")
        notes = tuple(AudioNote(f"harmonic-{pitch:03d}-100", pitch, "harmonic", 100,
                                harmonic_note(pitch, codec))
                      for pitch in range(bounds[0], bounds[1] + 1))
    else:
        if codec.duration_seconds < 1:
            raise ConfigurationError("Las 40 variantes requieren al menos un segundo de audio")
        # El mismo producto cartesiano para TODAS las alturas: jamás transponer
        # un objetivo reservado ni usarlo para crear datos de entrenamiento.
        notes = []
        for pitch, index in pitch_schedule(bounds, microtones_per_semitone, microtone_variants):
            attack, release, duration = ENVELOPES[index]
            notes.append(AudioNote(
                note_label(pitch, f"v{index:02d}"), pitch, "harmonic", 100,
                harmonic_note(pitch, codec, attack_seconds=attack, release_seconds=release,
                              duration_fraction=duration).astype(np.float32),
                variant=f"v{index:02d}", parameters={"attack_seconds": attack,
                    "release_seconds": release, "duration_fraction": duration}))
        notes = tuple(notes)
    return assemble_notes(notes, "synthetic_notes", codec, exclude_pitches, settings)
