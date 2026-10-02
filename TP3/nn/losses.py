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


@register("loss", "cross_entropy")
class CrossEntropy:
    """L = mean over the batch of -sum(y_true * log(y_pred)).

    Expects y_pred to be a probability distribution per row (a softmax output) and
    y_true one-hot. Only the true class's term survives: L = -log(p_true).
    """

    # Smallest positive float: clipping here only guards log(0) and keeps y/p exact.
    _FLOOR = np.finfo(float).tiny

    def __init__(self) -> None:
        self._y_pred: np.ndarray | None = None
        self._y_true: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        self._y_pred = np.maximum(y_pred, self._FLOOR)
        self._y_true = y_true
        return float(-np.sum(y_true * np.log(self._y_pred)) / len(y_pred))

    def backward(self) -> np.ndarray:
        if self._y_pred is None or self._y_true is None:
            raise RuntimeError("CrossEntropy.backward called before forward")
        return -self._y_true / self._y_pred
