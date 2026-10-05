"""Weight initializers: callables `init(shape) -> np.ndarray` backed by the shared rng."""

import numpy as np

from nn.registry import register


@register("initializer", "uniform")
class Uniform:
    """Samples weights uniformly from [low, high)."""

    def __init__(self, rng: np.random.Generator, low: float = -0.5, high: float = 0.5):
        if low >= high:
            raise ValueError(f"Expected low < high, got low={low}, high={high}")
        self.rng = rng
        self.low = low
        self.high = high

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        return self.rng.uniform(self.low, self.high, size=shape)


@register("initializer", "he")
class HeUniform:
    """Fan-in scaled uniform initialization for ReLU layers (He et al., 2015).

    Var(w) = 2 / fan_in compensates for ReLU zeroing half of its inputs, so the
    signal keeps its scale through deep ReLU stacks (Xavier assumes symmetric
    activations such as tanh).
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("He requires a positive (fan_in, fan_out) shape")
        limit = np.sqrt(6.0 / shape[0])
        return self.rng.uniform(-limit, limit, size=shape)


@register("initializer", "xavier")
class XavierUniform:
    """Fan-in/fan-out scaled uniform initialization for dense layers."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("Xavier requires a positive (fan_in, fan_out) shape")
        limit = np.sqrt(6.0 / sum(shape))
        return self.rng.uniform(-limit, limit, size=shape)
