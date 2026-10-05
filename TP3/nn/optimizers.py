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


@register("optimizer", "adam")
class Adam:
    """Adam (Kingma & Ba, 2014): momentum plus a per-weight step size.

    m: running mean of the gradient (direction, as in momentum).
    v: running mean of the squared gradient (how large this weight's gradients are).
    Both start at zero, so they are divided by (1 - beta^t) to undo that bias early on.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        for name, beta in (("beta1", beta1), ("beta2", beta2)):
            if not np.isfinite(beta) or not 0 <= beta < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: dict[Parameter, np.ndarray] = {}
        self.v: dict[Parameter, np.ndarray] = {}
        self.t = 0

    def step(self, params: list[Parameter]) -> None:
        self.t += 1
        m_correction = 1 - self.beta1 ** self.t
        v_correction = 1 - self.beta2 ** self.t
        for param in params:
            if param not in self.m:
                self.m[param] = np.zeros_like(param.value)
                self.v[param] = np.zeros_like(param.value)
            m, v = self.m[param], self.v[param]
            m *= self.beta1
            m += (1 - self.beta1) * param.grad
            v *= self.beta2
            v += (1 - self.beta2) * param.grad ** 2
            param.value -= self.lr * (m / m_correction) / (np.sqrt(v / v_correction) + self.epsilon)
