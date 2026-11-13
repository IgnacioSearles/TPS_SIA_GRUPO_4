"""Ternas de entrenamiento y configuración de la supervisión de interpolación."""

from dataclasses import dataclass

import numpy as np

from src.options import ConfigurationError, Options


@dataclass(frozen=True)
class InterpolationSettings:
    weight: float = 0.0
    batch_size: int = 32

    def __post_init__(self):
        if (type(self.weight) not in (int, float) or not np.isfinite(self.weight)
                or self.weight < 0):
            raise ConfigurationError("training.interpolation.weight debe ser finito y no negativo")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ConfigurationError("training.interpolation.batch_size debe ser positivo")

    @classmethod
    def parse(cls, options: Options):
        result = cls(options.take("weight", 0.0), options.take("batch_size", 32))
        options.ensure_consumed()
        return result


def build_interpolation_triplets(dataset):
    """Índices [inferior, centro, superior], separados por un semitono.

    El único universo elegible es Dataset.samples. La metadata completa nunca
    sirve como fuente de objetivos: ni las notas omitidas ni sus microtonos
    cercanos pueden entrar por esta vía. Cada terna comparte timbre y envolvente.
    """
    if dataset.pitch_study is None:
        raise ConfigurationError("training.interpolation requiere un dataset con pitch_study")
    indices = {label: index for index, label in enumerate(dataset.labels)}
    available = {(n.pitch, n.variant, n.instrument, n.velocity): indices[n.label]
                 for n in dataset.pitch_study.notes
                 if n.label in indices and float(n.pitch).is_integer()}
    triplets = []
    for (pitch, variant, instrument, velocity), center in sorted(available.items()):
        lower = available.get((pitch - 1, variant, instrument, velocity))
        upper = available.get((pitch + 1, variant, instrument, velocity))
        if lower is not None and upper is not None:
            triplets.append((lower, center, upper))
    if not triplets:
        raise ConfigurationError("No hay ternas completas de notas conocidas con la misma envolvente")
    return np.asarray(triplets, dtype=int)


def triplet_records(dataset, triplets):
    """Manifiesto auditable: etiquetas de las tres muestras realmente entrenadas."""
    notes = {n.label: n for n in dataset.pitch_study.notes}
    records = []
    for lower, center, upper in triplets:
        labels = [dataset.labels[index] for index in (lower, center, upper)]
        note = notes[labels[1]]
        records.append({"lower": labels[0], "center": labels[1], "upper": labels[2],
                        "center_pitch": note.pitch, "variant": note.variant,
                        "instrument": note.instrument, "velocity": note.velocity})
    return records
