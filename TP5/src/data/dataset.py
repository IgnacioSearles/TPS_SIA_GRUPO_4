"""Contenedor común de datos para todos los modelos y evaluaciones."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.notes import PitchStudy

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
    image_shape: tuple[int, int]
    is_binary: bool
    audio_codec: SpectrogramCodec | None = None
    pitch_study: PitchStudy | None = None

    def __post_init__(self) -> None:
        rows, columns = self.image_shape
        if self.samples.ndim != 2 or self.samples.shape[1] != rows * columns:
            raise ValueError("samples debe ser N x (filas * columnas) según image_shape")
        if not len(self.samples) == len(self.labels) == len(self.groups):
            raise ValueError("samples, labels y groups deben tener el mismo largo")
        if len(self.samples) == 0:
            raise ValueError("El dataset está vacío")

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
        return replace(
            self,
            samples=self.samples[indexes],
            labels=[self.labels[index] for index in indexes],
            groups=[self.groups[index] for index in indexes],
        )
