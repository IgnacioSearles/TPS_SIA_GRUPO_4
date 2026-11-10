"""Lo que toda evaluación recibe, y utilidades de salida compartidas."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.audio import write_waveform
from src.data.dataset import Dataset
from src.evaluation.metrics import binarize
from src.models.autoencoder import Autoencoder

Array = np.ndarray
Metrics = dict[str, float]
_UNSAFE_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\s`{}~]+')


@dataclass(frozen=True)
class EvaluationContext:
    model: Autoencoder
    dataset: Dataset
    output_directory: Path
    rng: np.random.Generator

    def displayable(self, decoded: Array) -> Array:
        """Salida del decodificador tal como se la juzga: binarizada si el dataset lo es."""
        return binarize(decoded) if self.dataset.is_binary else decoded

    def is_novel(self, decoded: Array) -> list[bool]:
        """Para datos binarios: si cada patrón generado no coincide con ninguno del dataset."""
        patterns = binarize(decoded)
        return [not np.any(np.all(self.dataset.samples == pattern, axis=1)) for pattern in patterns]

    def write_audio(self, relative_path: str, decoded: Array) -> Array | None:
        """Guarda como WAV un vector decodificado y devuelve su forma de onda.

        Si el dataset no es de audio no hace nada y devuelve None.
        """
        codec = self.dataset.audio_codec
        if codec is None:
            return None
        waveform = codec.decode(decoded.reshape(self.dataset.image_shape))
        self.write_waveform(relative_path, waveform)
        return waveform

    def write_waveform(self, relative_path: str, waveform: Array) -> None:
        codec = self.dataset.audio_codec
        if codec is None:
            raise ValueError("El dataset no es de audio")
        path = self.output_directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        write_waveform(path, waveform, codec.sample_rate)


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No hay filas para escribir en {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(text: str) -> str:
    """Versión de `text` válida como nombre de archivo en Windows y Unix."""
    cleaned = _UNSAFE_FILENAME_CHARACTERS.sub("_", text).strip("_.")
    return cleaned or "sin_nombre"
