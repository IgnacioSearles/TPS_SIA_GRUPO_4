"""Trainable layers. Every layer follows the forward / backward / params protocol."""

from typing import Callable

import numpy as np


class Parameter:
    """A trainable tensor and the gradient of the loss with respect to it."""

    def __init__(self, name: str, value: np.ndarray):
        # Unique across the network (e.g. "dense_0.W"); optimizers key their state by it.
        self.name = name
        self.value = value
        self.grad = np.zeros_like(value)


class Dense:
    """Fully connected layer: y = x @ W + b."""

    def __init__(self, n_in: int, n_out: int, init: Callable[[tuple[int, ...]], np.ndarray], name: str):
        self.W = Parameter(f"{name}.W", init((n_in, n_out)))
        self.b = Parameter(f"{name}.b", np.zeros(n_out))
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W.value + self.b.value

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("Dense.backward called before forward")
        batch_size = len(grad)
        # Averaging over the batch here keeps the learning rate independent of batch size.
        self.W.grad = self._x.T @ grad / batch_size
        self.b.grad = grad.mean(axis=0)
        return grad @ self.W.value.T

    def params(self) -> list[Parameter]:
        return [self.W, self.b]
