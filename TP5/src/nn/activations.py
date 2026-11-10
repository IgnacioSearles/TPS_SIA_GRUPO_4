"""Funciones de activación y sus derivadas respecto de la preactivación."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.options import choose

Array = np.ndarray
LEAKY_RELU_SLOPE = 0.01


@dataclass(frozen=True)
class Activation:
    name: str
    apply: Callable[[Array], Array]
    # Recibe (preactivación, salida) para reutilizar lo ya calculado en forward.
    derivative: Callable[[Array, Array], Array]


def _stable_sigmoid(values: Array) -> Array:
    return np.exp(-np.logaddexp(0.0, -values))


ACTIVATIONS: dict[str, Activation] = {
    activation.name: activation
    for activation in (
        Activation("identity", lambda x: x, lambda x, y: np.ones_like(x)),
        Activation("tanh", np.tanh, lambda x, y: 1.0 - y**2),
        Activation("sigmoid", _stable_sigmoid, lambda x, y: y * (1.0 - y)),
        Activation("relu", lambda x: np.maximum(x, 0.0), lambda x, y: (x > 0).astype(x.dtype)),
        Activation(
            "leaky_relu",
            lambda x: np.where(x > 0, x, LEAKY_RELU_SLOPE * x),
            lambda x, y: np.where(x > 0, 1.0, LEAKY_RELU_SLOPE).astype(x.dtype),
        ),
        Activation("softplus", lambda x: np.logaddexp(0.0, x), lambda x, y: _stable_sigmoid(x)),
    )
}


def build_activation(name: str) -> Activation:
    return choose(ACTIVATIONS, name, "Activación")
