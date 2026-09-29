"""Sequential container, config-driven model construction and weight persistence."""

from pathlib import Path
from typing import Any

import numpy as np

from nn.layers import Dense, Parameter
from nn.registry import Spec, build


class Sequential:
    """Chains layers: forward runs them in order, backward in reverse."""

    def __init__(self, layers: list[Any]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> list[Parameter]:
        return [param for layer in self.layers for param in layer.params()]


def resolve_activations(model_cfg: dict[str, Any], n_dense: int) -> list[Spec]:
    """Return one activation spec per Dense layer.

    `activation` is either a single spec (a name or a {"name": ...} dict) used
    for every hidden layer, with `output_activation` for the last layer, or a
    list with exactly one spec per Dense layer.
    """
    activation = model_cfg.get("activation", "identity")

    if isinstance(activation, list):
        if "output_activation" in model_cfg:
            raise ValueError("Use either a per-layer 'activation' list or 'output_activation', not both")
        if len(activation) != n_dense:
            raise ValueError(
                f"'activation' list has {len(activation)} entries but the model has {n_dense} Dense layers"
            )
        return activation

    output_activation = model_cfg.get("output_activation", activation)
    return [activation] * (n_dense - 1) + [output_activation]


def build_model(model_cfg: dict[str, Any], rng: np.random.Generator) -> Sequential:
    """Build a Sequential from the `model` section of a config.

    `layers` lists layer sizes: [2, 1] is a simple perceptron, [2, 3, 1] adds
    one hidden layer. Each Dense is followed by its activation, as resolved by
    `resolve_activations`.
    """
    if "layers" not in model_cfg:
        raise ValueError("model config is missing 'layers', e.g. [2, 1]")
    sizes = model_cfg["layers"]
    if len(sizes) < 2:
        raise ValueError(f"model.layers needs at least input and output sizes, got {sizes}")

    init = build("initializer", model_cfg.get("initializer", "uniform"), rng=rng)
    activations = resolve_activations(model_cfg, n_dense=len(sizes) - 1)

    layers: list[Any] = []
    for index, (n_in, n_out, activation) in enumerate(zip(sizes[:-1], sizes[1:], activations)):
        layers.append(Dense(n_in, n_out, init, name=f"dense_{index}"))
        layers.append(build("activation", activation))
    return Sequential(layers)


def save_weights(net: Sequential, path: str | Path) -> Path:
    """Persist all trainable parameters as a portable NumPy .npz archive keyed by parameter name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{param.name: param.value for param in net.params()})
    return path


def load_weights(net: Sequential, path: str | Path) -> Sequential:
    """Restore parameters saved by `save_weights` into a net with the same architecture."""
    with np.load(Path(path), allow_pickle=False) as saved:
        for param in net.params():
            if param.name not in saved.files:
                raise ValueError(f"Saved weights have no parameter '{param.name}'")
            value = saved[param.name]
            if value.shape != param.value.shape:
                raise ValueError(f"Wrong shape for {param.name}: saved {value.shape}, model {param.value.shape}")
            param.value[...] = value
    return net
