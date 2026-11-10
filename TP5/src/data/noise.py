"""Corrupciones de entrada para entrenar y evaluar el Denoising Autoencoder.

Todas reciben muestras en [0, 1], devuelven una copia corrompida en [0, 1] y
se parametrizan con un único `level` cuyo significado depende del tipo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.options import ConfigurationError, Options, choose

Array = np.ndarray
NoiseFunction = Callable[[Array, float, np.random.Generator], Array]


def _flip(samples: Array, level: float, rng: np.random.Generator) -> Array:
    """Invierte cada píxel con probabilidad `level`."""
    return np.where(rng.random(samples.shape) < level, 1.0 - samples, samples)


def _mask(samples: Array, level: float, rng: np.random.Generator) -> Array:
    """Apaga cada píxel con probabilidad `level`."""
    return np.where(rng.random(samples.shape) < level, 0.0, samples)


def _salt_and_pepper(samples: Array, level: float, rng: np.random.Generator) -> Array:
    """Reemplaza cada píxel, con probabilidad `level`, por 0 o 1 al azar."""
    replacement = rng.integers(0, 2, samples.shape).astype(samples.dtype)
    return np.where(rng.random(samples.shape) < level, replacement, samples)


def _gaussian(samples: Array, level: float, rng: np.random.Generator) -> Array:
    """Suma ruido normal de desvío `level` y recorta a [0, 1]."""
    return np.clip(samples + rng.normal(0.0, level, samples.shape), 0.0, 1.0)


_PROBABILITY_NOISES: dict[str, NoiseFunction] = {
    "flip": _flip,
    "mask": _mask,
    "salt_and_pepper": _salt_and_pepper,
}
NOISES: dict[str, NoiseFunction] = {**_PROBABILITY_NOISES, "gaussian": _gaussian}


@dataclass(frozen=True)
class Noise:
    kind: str
    level: float

    def __post_init__(self) -> None:
        choose(NOISES, self.kind, "Tipo de ruido")
        if self.level < 0 or (self.kind in _PROBABILITY_NOISES and self.level > 1):
            raise ConfigurationError(f"Nivel de ruido inválido para '{self.kind}': {self.level}")

    def apply(self, samples: Array, rng: np.random.Generator) -> Array:
        return NOISES[self.kind](samples, self.level, rng)


def build_noise(options: Options) -> Noise:
    """Sección con `type` (por defecto 'flip') y `level` (obligatorio)."""
    noise = Noise(options.take("type", "flip"), options.require("level"))
    options.ensure_consumed()
    return noise
