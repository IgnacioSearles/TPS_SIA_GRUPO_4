"""Pérdidas de reconstrucción definidas sobre los logits de la última capa.

Trabajar con logits (y no con la salida ya activada) permite que la entropía
cruzada sea numéricamente estable aunque la sigmoide sature.
Cada pérdida devuelve un valor por muestra, sumado sobre las características.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.nn.activations import Activation
from src.options import ConfigurationError, choose

Array = np.ndarray


class ReconstructionLoss(ABC):
    def __init__(self, output_activation: Activation) -> None:
        self._output_activation = output_activation

    @abstractmethod
    def per_sample(self, logits: Array, targets: Array) -> Array:
        """Pérdida de cada muestra (vector de largo N)."""

    @abstractmethod
    def logit_gradient(self, logits: Array, targets: Array) -> Array:
        """Derivada de la suma de `per_sample` respecto de los logits."""


class BinaryCrossEntropy(ReconstructionLoss):
    def __init__(self, output_activation: Activation) -> None:
        if output_activation.name != "sigmoid":
            raise ConfigurationError("La pérdida 'bce' requiere output_activation 'sigmoid'")
        super().__init__(output_activation)

    def per_sample(self, logits: Array, targets: Array) -> Array:
        return np.sum(np.logaddexp(0.0, logits) - targets * logits, axis=1)

    def logit_gradient(self, logits: Array, targets: Array) -> Array:
        return self._output_activation.apply(logits) - targets


class SquaredError(ReconstructionLoss):
    def per_sample(self, logits: Array, targets: Array) -> Array:
        return np.sum((self._output_activation.apply(logits) - targets) ** 2, axis=1)

    def logit_gradient(self, logits: Array, targets: Array) -> Array:
        outputs = self._output_activation.apply(logits)
        return 2.0 * (outputs - targets) * self._output_activation.derivative(logits, outputs)


LOSSES: dict[str, type[ReconstructionLoss]] = {"bce": BinaryCrossEntropy, "mse": SquaredError}


def build_loss(name: str, output_activation: Activation) -> ReconstructionLoss:
    return choose(LOSSES, name, "Pérdida")(output_activation)
