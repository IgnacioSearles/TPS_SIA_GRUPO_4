"""Activation layers. `backward` is a vector-Jacobian product: dL/dz given dL/da."""

import numpy as np

from nn.layers import Parameter
from nn.registry import register


@register("activation", "identity")
class Identity:
    """a = z. Turns a Dense layer into a linear perceptron."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad

    def params(self) -> list[Parameter]:
        return []


@register("activation", "step")
class Step:
    """a = sign(z), with outputs in {-1, 1}.

    The true derivative is zero almost everywhere, so backward passes the
    gradient straight through. Combined with MSE this reproduces Rosenblatt's
    rule exactly: dw = lr * (y - y_hat) * x.
    """

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z >= 0, 1.0, -1.0)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad

    def params(self) -> list[Parameter]:
        return []


@register("activation", "sigmoid")
class Sigmoid:
    """a = 1 / (1 + exp(-z)). Suitable for differentiable binary MLPs."""

    def __init__(self) -> None:
        self._a: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        # The split form avoids overflowing exp for large-magnitude inputs.
        self._a = np.empty_like(z, dtype=float)
        positive = z >= 0
        self._a[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        self._a[~positive] = exp_z / (1.0 + exp_z)
        return self._a

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._a is None:
            raise RuntimeError("Sigmoid.backward called before forward")
        return grad * self._a * (1.0 - self._a)

    def params(self) -> list[Parameter]:
        return []


@register("activation", "tanh")
class Tanh:
    """a = tanh(z). Its output range [-1, 1] matches bipolar targets."""

    def __init__(self) -> None:
        self._a: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._a = np.tanh(z)
        return self._a

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._a is None:
            raise RuntimeError("Tanh.backward called before forward")
        return grad * (1.0 - self._a ** 2)

    def params(self) -> list[Parameter]:
        return []


@register("activation", "relu")
class ReLU:
    """a = max(0, z)."""

    def __init__(self) -> None:
        self._z: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._z = z
        return np.maximum(0.0, z)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._z is None:
            raise RuntimeError("ReLU.backward called before forward")
        return grad * (self._z > 0)

    def params(self) -> list[Parameter]:
        return []
