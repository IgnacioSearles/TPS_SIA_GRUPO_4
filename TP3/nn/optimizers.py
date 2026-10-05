"""Optimizers update parameters in place from their gradients.

All optimizers accept `weight_decay` (L2 regularization strength λ). It adds the
gradient of (λ/2)·||W||² to every weight matrix, so the effective gradient is
dL/dW + λ·W. Biases are not penalized: they shift activations and do not grow
the model's capacity the way large weights do.
"""

import numpy as np

from nn.layers import Parameter
from nn.registry import register


def _validate_weight_decay(weight_decay: float) -> float:
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and >= 0")
    return weight_decay


def _regularized_gradient(param: Parameter, weight_decay: float) -> np.ndarray:
    """dL/dparam plus the L2 term for weight matrices (parameters named '<layer>.W')."""
    if weight_decay and param.name.endswith(".W"):
        return param.grad + weight_decay * param.value
    return param.grad


@register("optimizer", "sgd")
class SGD:
    """Plain gradient descent: w <- w - lr * (dL/dw + weight_decay * w)."""

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = lr
        self.weight_decay = _validate_weight_decay(weight_decay)

    def step(self, params: list[Parameter]) -> None:
        for param in params:
            param.value -= self.lr * _regularized_gradient(param, self.weight_decay)


@register("optimizer", "momentum")
class Momentum:
    """Classical momentum: update = momentum * previous_update - lr * gradient.

    Starts with zero updates; no Nesterov lookahead or (1-momentum) scaling.
    State belongs to each Parameter object and survives minibatch boundaries.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not np.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be finite and in [0, 1)")
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = _validate_weight_decay(weight_decay)
        self.updates: dict[Parameter, np.ndarray] = {}

    def step(self, params: list[Parameter]) -> None:
        for param in params:
            if param not in self.updates:
                self.updates[param] = np.zeros_like(param.value)
            update = self.updates[param]
            update *= self.momentum
            update -= self.lr * _regularized_gradient(param, self.weight_decay)
            param.value += update


@register("optimizer", "adam")
class Adam:
    """Adam (Kingma & Ba, 2014): momentum plus a per-weight step size.

    m: running mean of the gradient (direction, as in momentum).
    v: running mean of the squared gradient (how large this weight's gradients are).
    Both start at zero, so they are divided by (1 - beta^t) to undo that bias early on.
    weight_decay is classic (coupled) L2: it enters the gradient, not a separate AdamW step.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.0):
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
        self.weight_decay = _validate_weight_decay(weight_decay)
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
            gradient = _regularized_gradient(param, self.weight_decay)
            m, v = self.m[param], self.v[param]
            m *= self.beta1
            m += (1 - self.beta1) * gradient
            v *= self.beta2
            v += (1 - self.beta2) * gradient ** 2
            param.value -= self.lr * (m / m_correction) / (np.sqrt(v / v_correction) + self.epsilon)
