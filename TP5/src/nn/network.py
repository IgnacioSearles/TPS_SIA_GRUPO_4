"""Perceptrón multicapa con retropropagación manual."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.nn.activations import Activation
from src.nn.initializers import Initializer

Array = np.ndarray


class DenseLayer:
    """Capa totalmente conectada: salida = activación(entrada @ pesos + sesgos)."""

    def __init__(self, weights: Array, activation: Activation) -> None:
        self.weights = weights
        self.biases = np.zeros(weights.shape[1], dtype=weights.dtype)
        self.weight_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.biases)
        self._activation = activation
        self._inputs: Array | None = None
        self._preactivation: Array | None = None
        self._outputs: Array | None = None

    def forward(self, inputs: Array) -> Array:
        self._inputs = inputs
        self._preactivation = inputs @ self.weights + self.biases
        self._outputs = self._activation.apply(self._preactivation)
        return self._outputs

    def backward(self, output_gradient: Array) -> Array:
        """Guarda los gradientes de la capa y devuelve el gradiente de su entrada."""
        if self._inputs is None:
            raise RuntimeError("backward requiere un forward previo")
        delta = output_gradient * self._activation.derivative(self._preactivation, self._outputs)
        self.weight_gradient = self._inputs.T @ delta
        self.bias_gradient = delta.sum(axis=0)
        return delta @ self.weights.T


class MultilayerPerceptron:
    """Secuencia de capas densas. La última capa usa `output_activation`."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        hidden_activation: Activation,
        output_activation: Activation,
        initializer: Initializer,
        rng: np.random.Generator,
        precision: type[np.floating] = np.float64,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Una red necesita al menos tamaño de entrada y de salida")
        if any(not isinstance(size, int) or size < 1 for size in layer_sizes):
            raise ValueError(f"Los tamaños de capa deben ser enteros positivos: {layer_sizes}")
        self.layer_sizes = list(layer_sizes)
        last_index = len(layer_sizes) - 2
        self._layers = [
            DenseLayer(
                initializer(rng, inputs, outputs).astype(precision),
                output_activation if index == last_index else hidden_activation,
            )
            for index, (inputs, outputs) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
        ]

    def forward(self, inputs: Array) -> Array:
        outputs = inputs
        for layer in self._layers:
            outputs = layer.forward(outputs)
        return outputs

    def backward(self, output_gradient: Array) -> Array:
        """Retropropaga desde la salida y devuelve el gradiente de la entrada."""
        gradient = output_gradient
        for layer in reversed(self._layers):
            gradient = layer.backward(gradient)
        return gradient

    def parameters(self) -> list[Array]:
        return [array for layer in self._layers for array in (layer.weights, layer.biases)]

    def gradients(self) -> list[Array]:
        """Gradientes del último backward, alineados con `parameters()`."""
        return [
            array for layer in self._layers for array in (layer.weight_gradient, layer.bias_gradient)
        ]
