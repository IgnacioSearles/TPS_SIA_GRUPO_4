"""Corrupciones de entrada para entrenar y evaluar el Denoising Autoencoder.

Todas reciben muestras en [0, 1], devuelven una copia corrompida en [0, 1] y
se parametrizan con un único `level` cuyo significado depende del tipo.
"""

from __future__ import annotations

import math
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


# Log-probabilidad de cada píxel ruidoso dado el píxel limpio (0 o 1), con nivel > 0.
# Los términos que no dependen del píxel limpio se omiten: no cambian qué patrón es
# más probable.
def _flip_log_likelihood(noisy: Array, clean: Array, level: float) -> Array:
    with np.errstate(divide="ignore"):
        return np.where(noisy != clean, np.log(level), np.log1p(-level))


def _mask_log_likelihood(noisy: Array, clean: Array, level: float) -> Array:
    with np.errstate(divide="ignore"):
        on = np.where(noisy == 0, np.log(level), np.log1p(-level))
    # Apagar no puede prender: un píxel prendido sobre uno apagado descarta el patrón.
    return np.where(clean == 1, on, np.where(noisy == 0, 0.0, -np.inf))


def _salt_and_pepper_log_likelihood(noisy: Array, clean: Array, level: float) -> Array:
    # La mitad de los reemplazos coinciden con el valor original: es invertir con level / 2.
    return _flip_log_likelihood(noisy, clean, level / 2)


def _gaussian_log_likelihood(noisy: Array, clean: Array, level: float) -> Array:
    # Dentro de (0, 1) es la densidad normal; en 0 y en 1 se acumula la probabilidad de
    # haber quedado fuera del intervalo antes de recortar: Φ(0) = 1/2 si el ruido tuvo que
    # alejarse del valor limpio y Φ(-1/σ) si tuvo que cruzar todo el intervalo.
    density = -((noisy - clean) ** 2) / (2 * level**2)
    tail = 0.5 * math.erfc(1 / (level * math.sqrt(2)))
    crossed = math.log(tail) if tail > 0 else -math.inf
    at_zero = np.where(clean == 0, math.log(0.5), crossed)
    at_one = np.where(clean == 1, math.log(0.5), crossed)
    return np.where(noisy <= 0, at_zero, np.where(noisy >= 1, at_one, density))


_LOG_LIKELIHOODS: dict[str, Callable[[Array, Array, float], Array]] = {
    "flip": _flip_log_likelihood,
    "mask": _mask_log_likelihood,
    "salt_and_pepper": _salt_and_pepper_log_likelihood,
    "gaussian": _gaussian_log_likelihood,
}


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

    def log_likelihood(self, noisy: Array, patterns: Array) -> Array:
        """log P(noisy | patrón), salvo una constante por fila, para patrones binarios.

        noisy: (n, píxeles); patterns: (k, píxeles). Devuelve (n, k).
        """
        noisy, patterns = noisy[:, None, :], patterns[None, :, :]
        if self.level == 0:
            per_pixel = np.where(noisy == patterns, 0.0, -np.inf)
        else:
            per_pixel = _LOG_LIKELIHOODS[self.kind](noisy, patterns, self.level)
        return per_pixel.sum(axis=2)


def build_noise(options: Options) -> Noise:
    """Sección con `type` (por defecto 'flip') y `level` (obligatorio)."""
    noise = Noise(options.take("type", "flip"), options.require("level"))
    options.ensure_consumed()
    return noise
