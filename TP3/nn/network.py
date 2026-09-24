"""Sequential container and config-driven model construction."""

from typing import Any

import numpy as np

from nn.layers import Dense, Parameter
from nn.registry import build


class Sequential:
    """Chains layers: forward runs them in order, backward in reverse."""

    def __init__(self, layers: list[Any]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> list[Parameter]:
        return [param for layer in self.layers for param in layer.params()]


def build_model(model_cfg: dict[str, Any], rng: np.random.Generator) -> Sequential:
    """Build a Sequential from the `model` section of a config.

    `layers` lists layer sizes: [2, 1] is a simple perceptron, [2, 3, 1] adds
    one hidden layer. Every Dense is followed by `activation`, except the last
    one, which is followed by `output_activation`.
    """
    if "layers" not in model_cfg:
        raise ValueError("model config is missing 'layers', e.g. [2, 1]")
    sizes = model_cfg["layers"]
    if len(sizes) < 2:
        raise ValueError(f"model.layers needs at least input and output sizes, got {sizes}")

    init = build("initializer", model_cfg.get("initializer", "uniform"), rng=rng)
    hidden_activation = model_cfg.get("activation", "identity")
    output_activation = model_cfg.get("output_activation", hidden_activation)
    last_index = len(sizes) - 2

    layers: list[Any] = []
    for index, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(Dense(n_in, n_out, init, name=f"dense_{index}"))
        activation = output_activation if index == last_index else hidden_activation
        layers.append(build("activation", activation))
    return Sequential(layers)
