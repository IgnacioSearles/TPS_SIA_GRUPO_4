"""Loss functions. `backward` returns dL/dy_pred per sample (NOT divided by batch size)."""

import numpy as np

from nn.registry import register


@register("loss", "mse")
class MSE:
    """L = mean over the batch of 0.5 * sum((y_pred - y_true)^2).

    The 0.5 factor makes the gradient exactly (y_pred - y_true), which keeps
    the step perceptron update identical to Rosenblatt's rule.
    """

    def __init__(self) -> None:
        self._error: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        self._error = y_pred - y_true
        return float(0.5 * np.sum(self._error ** 2) / len(y_pred))

    def backward(self) -> np.ndarray:
        if self._error is None:
            raise RuntimeError("MSE.backward called before forward")
        return self._error
