"""Optimizers update parameters in place from their gradients."""

import numpy as np

from nn.layers import Parameter
from nn.registry import register


@register("optimizer", "sgd")
class SGD:
    """Plain gradient descent: w <- w - lr * dL/dw."""

    def __init__(self, lr: float = 0.01):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = lr

    def step(self, params: list[Parameter]) -> None:
        for param in params:
            param.value -= self.lr * param.grad


@register("optimizer", "momentum")
class Momentum:
    """Classical momentum: update = momentum * previous_update - lr * gradient.

    Starts with zero updates; no Nesterov lookahead or (1-momentum) scaling.
    State belongs to each Parameter object and survives minibatch boundaries.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not np.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be finite and in [0, 1)")
        self.lr = lr
        self.momentum = momentum
        self.updates: dict[Parameter, np.ndarray] = {}

    def step(self, params: list[Parameter]) -> None:
        for param in params:
            if param not in self.updates:
                self.updates[param] = np.zeros_like(param.value)
            update = self.updates[param]
            update *= self.momentum
            update -= self.lr * param.grad
            param.value += update
