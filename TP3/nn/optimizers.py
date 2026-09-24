"""Optimizers update parameters in place from their gradients."""

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
