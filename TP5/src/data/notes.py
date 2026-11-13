"""Notas monofónicas y separación explícita entre entrenamiento y referencias."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.audio import SpectrogramCodec
from src.data.dataset import Dataset
from src.options import ConfigurationError, Options


def midi_frequency(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def midi_list(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or any(
        type(p) is not int or not 0 <= p <= 127 for p in values
    ) or len(set(values)) != len(values):
        raise ConfigurationError(f"{name} debe contener alturas MIDI únicas entre 0 y 127")
    return tuple(values)


@dataclass(frozen=True)
class StudySettings:
    omit_targets: bool = True
    targets: tuple[int, ...] | None = None
    evaluation_variants: tuple[str, ...] | None = None
    # Con omit_targets se excluye toda altura a <= exclusion_radius semitonos de un
    # objetivo. Mínimo 0.5: ninguna nota de entrenamiento cae dentro de la tolerancia
    # de acierto (50 cents). Menor que 1: los vecinos enteros siguen en entrenamiento.
    exclusion_radius: float = 0.5
    # No entran ni en controles completos ni en calibración/selección de modelos.
    reserved_targets: tuple[int, ...] = ()

    @classmethod
    def parse(cls, options: Options) -> StudySettings:
        omit = options.take("omit_targets", True)
        targets = options.take("targets", None)
        variants = options.take("evaluation_variants", None)
        radius = options.take("exclusion_radius", 0.5)
        reserved = midi_list(options.take("reserved_targets", []), "pitch_study.reserved_targets")
        options.ensure_consumed()
        if type(omit) is not bool:
            raise ConfigurationError("pitch_study.omit_targets debe ser booleano")
        if type(radius) not in (int, float) or not 0.5 <= radius < 1:
            raise ConfigurationError("pitch_study.exclusion_radius debe estar en [0.5, 1)")
        if targets is not None:
            targets = midi_list(targets, "pitch_study.targets")
            if not targets:
                raise ConfigurationError("pitch_study.targets no puede estar vacío")
        if variants is not None:
            if (not isinstance(variants, list) or not variants
                    or any(not isinstance(v, str) for v in variants)
                    or len(set(variants)) != len(variants)):
                raise ConfigurationError("evaluation_variants debe contener identificadores únicos")
            variants = tuple(variants)
        if targets is not None and set(reserved).intersection(targets):
            raise ConfigurationError("targets y reserved_targets deben ser disjuntos")
        return cls(omit, targets, variants, float(radius), reserved)


@dataclass(frozen=True)
class AudioNote:
    label: str
    pitch: float  # MIDI; entero salvo en los microtonos sintéticos
    instrument: str
    velocity: int
    waveform: np.ndarray
    variant: str = "base"
    parameters: dict[str, float] | None = None


@dataclass(frozen=True)
class PitchStudy:
    """Referencias de evaluación: el Trainer recibe únicamente Dataset.samples."""

    source: str
    notes: tuple[AudioNote, ...]
    targets: tuple[int, ...]
    omit_targets: bool
    evaluation_variants: tuple[str, ...] = ("base",)
    reserved_targets: tuple[int, ...] = ()

    def note(self, pitch: int, variant: str = "base") -> AudioNote:
        return next(note for note in self.notes if note.pitch == pitch and note.variant == variant)


def assemble_notes(notes: tuple[AudioNote, ...], source: str, codec: SpectrogramCodec,
                   excluded: tuple[int, ...], settings: StudySettings | None,
                   encoded: np.ndarray | None = None) -> Dataset:
    """Forma el conjunto completo y filtra las filas antes de entregarlas al Trainer."""
    pitches = sorted({note.pitch for note in notes})
    identities = {(note.pitch, note.variant) for note in notes}
    variants = sorted({note.variant for note in notes})
    if not pitches or len(identities) != len(notes) or len({note.label for note in notes}) != len(notes):
        raise ConfigurationError("Cada combinación de altura y variante debe tener una etiqueta única")
    evaluation_variants = tuple(variants)
    targets: tuple[int, ...] = ()
    if settings is not None:
        evaluation_variants = settings.evaluation_variants or tuple(variants)
        if not set(evaluation_variants).issubset(variants):
            raise ConfigurationError("evaluation_variants contiene variantes inexistentes")
        semitones = [p for p in pitches if float(p).is_integer()]
        targets = settings.targets or tuple(semitones[(len(semitones) - 1) * k // 4]
                                             for k in (1, 2, 3))
        if len(set(targets)) != len(targets):
            raise ConfigurationError("No hay suficientes notas para tres objetivos interiores")
        if set(targets).intersection(settings.reserved_targets):
            raise ConfigurationError("targets y reserved_targets deben ser disjuntos")
        omitted = set(excluded)
        omitted |= {p for p in pitches
                    if any(abs(p - t) <= settings.exclusion_radius + 1e-9
                           for t in settings.reserved_targets)}
        if settings.omit_targets:
            # Ventana alrededor de cada objetivo: también saca los microtonos cercanos.
            omitted |= {p for p in pitches
                        if any(abs(p - t) <= settings.exclusion_radius + 1e-9 for t in targets)}
        for target in targets:
            if target not in pitches or any(p not in pitches or p in omitted
                                            for p in (target - 1, target + 1)):
                raise ConfigurationError(f"El objetivo {target} necesita ambos vecinos en entrenamiento")
            if any((p, v) not in identities for v in variants
                   for p in (target - 1, target, target + 1)):
                raise ConfigurationError(f"El objetivo {target} necesita vecinos de la misma variante")
            if not settings.omit_targets and target in omitted:
                raise ConfigurationError("El control completo debe incluir las notas objetivo")
    else:
        omitted = set(excluded)
    kept = [i for i, note in enumerate(notes) if note.pitch not in omitted]
    if not kept:
        raise ConfigurationError("La exclusión dejó el dataset vacío")
    if encoded is None:
        # Las referencias omitidas se codifican recién durante la evaluación.
        samples = np.asarray([codec.encode(notes[i].waveform).ravel() for i in kept])
    else:
        samples = encoded[kept]
    return Dataset(samples=samples, labels=[notes[i].label for i in kept],
                   groups=[notes[i].instrument for i in kept], image_shape=codec.image_shape,
                   is_binary=False, audio_codec=codec,
                   pitch_study=None if settings is None else PitchStudy(source, notes, targets,
                                                                        settings.omit_targets,
                                                                        evaluation_variants,
                                                                        settings.reserved_targets))
