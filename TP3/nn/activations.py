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
