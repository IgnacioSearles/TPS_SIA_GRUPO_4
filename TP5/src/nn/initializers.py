"""Inicialización de matrices de pesos de forma (entradas, salidas)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.options import choose

Initializer = Callable[[np.random.Generator, int, int], np.ndarray]

SMALL_NORMAL_SCALE = 0.01


def _xavier(rng: np.random.Generator, inputs: int, outputs: int) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / (inputs + outputs)), (inputs, outputs))


def _xavier_sigmoid(rng: np.random.Generator, inputs: int, outputs: int) -> np.ndarray:
    """Xavier supone derivada 1 en el origen; la de la sigmoide es 1/4, así que se escala por 4."""
    return 4.0 * _xavier(rng, inputs, outputs)


def _he(rng: np.random.Generator, inputs: int, outputs: int) -> np.ndarray:
    return rng.normal(0.0, np.sqrt(2.0 / inputs), (inputs, outputs))


def _small_normal(rng: np.random.Generator, inputs: int, outputs: int) -> np.ndarray:
    return rng.normal(0.0, SMALL_NORMAL_SCALE, (inputs, outputs))


def _uniform(rng: np.random.Generator, inputs: int, outputs: int) -> np.ndarray:
    limit = 1.0 / np.sqrt(inputs)
    return rng.uniform(-limit, limit, (inputs, outputs))


INITIALIZERS: dict[str, Initializer] = {
    "xavier": _xavier,
    "xavier_sigmoid": _xavier_sigmoid,
    "he": _he,
    "small_normal": _small_normal,
    "uniform": _uniform,
}


def build_initializer(name: str) -> Initializer:
    return choose(INITIALIZERS, name, "Inicializador")
