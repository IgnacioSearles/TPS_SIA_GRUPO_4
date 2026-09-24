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
