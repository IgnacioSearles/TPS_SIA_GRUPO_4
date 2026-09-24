"""Finite-difference gradient checks: analytical backward vs numerical derivative."""

import numpy as np
import pytest

from nn import build, build_model

EPSILON = 1e-6
TOLERANCE = 1e-6


def relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    return float(np.max(np.abs(analytical - numerical) / np.maximum(1e-8, np.abs(analytical) + np.abs(numerical))))


def numerical_gradient(f, x: np.ndarray) -> np.ndarray:
    """Central differences of the scalar function f with respect to every entry of x (mutated in place)."""
    grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + EPSILON
        plus = f()
        x[index] = original - EPSILON
        minus = f()
        x[index] = original
        grad[index] = (plus - minus) / (2 * EPSILON)
    return grad


def test_mse_gradient():
    rng = np.random.default_rng(0)
    y_pred, y_true = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    loss = build("loss", "mse")

    loss.forward(y_pred, y_true)
    # backward is per-sample (not averaged), so scale by batch size to compare with dL/dy_pred.
    analytical = loss.backward() / len(y_pred)
    numerical = numerical_gradient(lambda: loss.forward(y_pred, y_true), y_pred)

    assert relative_error(analytical, numerical) < TOLERANCE


# Step is excluded: its straight-through backward is intentionally not its true derivative.
@pytest.mark.parametrize("activation_name", ["identity", "sigmoid", "tanh", "relu"])
def test_activation_gradient(activation_name):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(4, 3))
    upstream = rng.normal(size=(4, 3))
    activation = build("activation", activation_name)

    activation.forward(z)
    analytical = activation.backward(upstream)
    numerical = numerical_gradient(lambda: float(np.sum(activation.forward(z) * upstream)), z)

    assert relative_error(analytical, numerical) < TOLERANCE


def test_network_parameter_gradients():
    rng = np.random.default_rng(0)
    net = build_model({"layers": [3, 4, 2], "activation": "identity"}, rng)
    loss = build("loss", "mse")
    X, Y = rng.normal(size=(5, 3)), rng.normal(size=(5, 2))

    loss.forward(net.forward(X), Y)
    net.backward(loss.backward())

    for param in net.params():
        numerical = numerical_gradient(lambda: loss.forward(net.forward(X), Y), param.value)
        assert relative_error(param.grad, numerical) < TOLERANCE, param.name


def test_multilayer_network_parameter_gradients_with_tanh():
    rng = np.random.default_rng(4)
    net = build_model({"layers": [2, 3, 2, 1], "activation": "tanh"}, rng)
    loss = build("loss", "mse")
    X, Y = rng.normal(size=(4, 2)), rng.normal(size=(4, 1))

    loss.forward(net.forward(X), Y)
    net.backward(loss.backward())

    for param in net.params():
        numerical = numerical_gradient(lambda: loss.forward(net.forward(X), Y), param.value)
        assert relative_error(param.grad, numerical) < TOLERANCE, param.name
