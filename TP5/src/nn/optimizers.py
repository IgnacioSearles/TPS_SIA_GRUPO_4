"""Optimizadores de primer orden que actualizan los parámetros en el lugar."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from src.options import ConfigurationError, Options, choose

Array = np.ndarray


class Optimizer(ABC):
    def __init__(self, learning_rate: float) -> None:
        if not learning_rate > 0:
            raise ConfigurationError(f"learning_rate debe ser positivo, no {learning_rate}")
        self._learning_rate = learning_rate

    @abstractmethod
    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> None:
        """Aplica una actualización; `gradients` está alineado con `parameters`."""


class GradientDescent(Optimizer):
    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> None:
        for parameter, gradient in zip(parameters, gradients):
            parameter -= self._learning_rate * gradient


class Momentum(Optimizer):
    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        super().__init__(learning_rate)
        self._momentum = _validated_decay(momentum, "momentum")
        self._velocities: list[Array] | None = None

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> None:
        if self._velocities is None:
            self._velocities = [np.zeros_like(parameter) for parameter in parameters]
        for parameter, gradient, velocity in zip(parameters, gradients, self._velocities):
            velocity *= self._momentum
            velocity -= self._learning_rate * gradient
            parameter += velocity


class RootMeanSquarePropagation(Optimizer):
    def __init__(self, learning_rate: float, decay: float = 0.9, epsilon: float = 1e-8) -> None:
        super().__init__(learning_rate)
        self._decay = _validated_decay(decay, "decay")
        self._epsilon = epsilon
        self._squares: list[Array] | None = None

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> None:
        if self._squares is None:
            self._squares = [np.zeros_like(parameter) for parameter in parameters]
        for parameter, gradient, square in zip(parameters, gradients, self._squares):
            square *= self._decay
            square += (1.0 - self._decay) * gradient**2
            parameter -= self._learning_rate * gradient / (np.sqrt(square) + self._epsilon)


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        self._beta1 = _validated_decay(beta1, "beta1")
        self._beta2 = _validated_decay(beta2, "beta2")
        self._epsilon = epsilon
        self._first_moments: list[Array] | None = None
        self._second_moments: list[Array] | None = None
        self._steps = 0

    def step(self, parameters: Sequence[Array], gradients: Sequence[Array]) -> None:
        if self._first_moments is None or self._second_moments is None:
            self._first_moments = [np.zeros_like(parameter) for parameter in parameters]
            self._second_moments = [np.zeros_like(parameter) for parameter in parameters]
        self._steps += 1
        # Las correcciones de sesgo se pliegan en escalares: es algebraicamente
        # igual a m̂ / (sqrt(v̂) + ε) y evita copias de matrices grandes.
        # Escalares de Python (no de NumPy) para no promover pesos float32 a float64.
        second_correction_root = math.sqrt(1.0 - self._beta2**self._steps)
        step_size = self._learning_rate * second_correction_root / (1.0 - self._beta1**self._steps)
        scaled_epsilon = self._epsilon * second_correction_root
        for parameter, gradient, first, second in zip(
            parameters, gradients, self._first_moments, self._second_moments
        ):
            first *= self._beta1
            first += (1.0 - self._beta1) * gradient
            second *= self._beta2
            second += (1.0 - self._beta2) * gradient * gradient
            update = np.sqrt(second)
            update += scaled_epsilon
            np.divide(first, update, out=update)
            update *= step_size
            parameter -= update


def _validated_decay(value: float, name: str) -> float:
    if not 0 <= value < 1:
        raise ConfigurationError(f"{name} debe estar en [0, 1), no {value}")
    return value


OPTIMIZERS: dict[str, type[Optimizer]] = {
    "sgd": GradientDescent,
    "momentum": Momentum,
    "rmsprop": RootMeanSquarePropagation,
    "adam": Adam,
}


def build_optimizer(options: Options) -> Optimizer:
    """Sección `training.optimizer`: `type`, `learning_rate` y los hiperparámetros del tipo."""
    optimizer_class = choose(OPTIMIZERS, options.take("type", "adam"), "Optimizador")
    hyperparameters = {key: options.take(key, None) for key in options.keys() if key != "type"}
    hyperparameters.setdefault("learning_rate", 0.001)
    try:
        return optimizer_class(**hyperparameters)
    except TypeError as error:
        raise ConfigurationError(f"Hiperparámetros inválidos para el optimizador: {error}") from error
