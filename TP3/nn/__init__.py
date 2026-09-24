"""Minimal neural network library built from scratch with NumPy.

Importing the component modules registers their classes in the registry,
so `build("activation", "step")` works as soon as `nn` is imported.
"""

from nn import activations, initializers, losses, optimizers  # noqa: F401  (registration side effect)
from nn.layers import Dense, Parameter
from nn.network import Sequential, build_model
from nn.registry import build, register

__all__ = ["Dense", "Parameter", "Sequential", "build", "build_model", "register"]
