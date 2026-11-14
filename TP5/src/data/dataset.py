"""Contenedor común de datos para todos los modelos y evaluaciones."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.notes import PitchStudy
    from src.data.generative import GenerativeStudy

import numpy as np

from src.data.audio import SpectrogramCodec


@dataclass(frozen=True)
class Dataset:
    """Muestras aplanadas en [0, 1] junto con cómo interpretarlas.

    `groups` es la categoría de cada muestra (para colorear gráficos);
    `audio_codec` existe solo si cada muestra es un espectrograma audible.
    """

    samples: np.ndarray
    labels: list[str]
    groups: list[str]
    image_shape: tuple[int, ...]
    is_binary: bool
    audio_codec: SpectrogramCodec | None = None
    pitch_study: PitchStudy | None = None
    generative_study: GenerativeStudy | None = None

    def __post_init__(self) -> None:
        if (len(self.image_shape) not in (2, 3)
                or any(type(v) is not int or v < 1 for v in self.image_shape)
                or (len(self.image_shape) == 3 and self.image_shape[2] != 3)):
            raise ValueError("image_shape debe ser (alto, ancho) o (alto, ancho, 3)")
        if len(self.image_shape) == 3 and (self.is_binary or self.audio_codec is not None):
            raise ValueError("RGB no admite datos binarios ni códec de audio")
        features = int(np.prod(self.image_shape))
        if self.samples.ndim != 2 or self.samples.shape[1] != features:
            raise ValueError("samples debe ser N x (producto de image_shape) según image_shape")
        if not len(self.samples) == len(self.labels) == len(self.groups):
            raise ValueError("samples, labels y groups deben tener el mismo largo")
        if len(self.samples) == 0:
            raise ValueError("El dataset está vacío")
        if self.generative_study is not None:
            study = self.generative_study
            if (study.test_samples.ndim != 2 or study.test_samples.shape[1] != features
                    or not len(study.test_samples) == len(study.test_labels) == len(study.test_records)
                    or not len(study.train_records) == len(self.samples) or not len(study.test_samples)):
                raise ValueError("El conjunto generativo de prueba no está alineado")
            if set(self.labels).intersection(study.test_labels):
                raise ValueError("Entrenamiento y prueba comparten etiquetas")
            if self.audio_codec is not None and (study.test_waveforms is None
                    or len(study.test_waveforms) != len(study.test_samples)):
                raise ValueError("Faltan formas de onda de referencia para el conjunto de prueba")

    def __len__(self) -> int:
        return len(self.samples)

    def index_of(self, reference: str | int) -> int:
        """Posición de una muestra dada por etiqueta exacta o por índice."""
        if isinstance(reference, int) and not isinstance(reference, bool):
            if not 0 <= reference < len(self):
                raise ValueError(f"Índice de muestra fuera de rango: {reference}")
            return reference
        if reference not in self.labels:
            raise ValueError(f"No existe una muestra con etiqueta {reference!r}")
        return self.labels.index(reference)

    def select(self, references: list[str | int]) -> Dataset:
        indexes = [self.index_of(reference) for reference in references]
        if len(set(indexes)) != len(indexes):
            raise ValueError("El subconjunto contiene muestras repetidas")
        study = self.generative_study
        if study is not None:
            study = replace(study, train_records=tuple(study.train_records[i] for i in indexes),
                            metadata={**study.metadata, "train_sha256": hashlib.sha256(
                                self.samples[indexes].astype("<f4").tobytes()).hexdigest()})
        return replace(
            self,
            samples=self.samples[indexes],
            labels=[self.labels[index] for index in indexes],
            groups=[self.groups[index] for index in indexes],
            generative_study=study,
        )
