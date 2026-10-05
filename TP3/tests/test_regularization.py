import numpy as np
import pytest

from nn import build
from nn.layers import Parameter


def weight_and_bias(value: float = 2.0) -> tuple[Parameter, Parameter]:
    weight = Parameter("dense_0.W", np.full((2, 2), value))
    bias = Parameter("dense_0.b", np.full(2, value))
    return weight, bias


@pytest.mark.parametrize("spec", [
    {"name": "sgd", "lr": 0.1},
    {"name": "momentum", "lr": 0.1, "momentum": 0.9},
    {"name": "adam", "lr": 0.1},
])
def test_weight_decay_shrinks_weights_but_not_biases_without_a_loss_gradient(spec):
    weight, bias = weight_and_bias()
    optimizer = build("optimizer", {**spec, "weight_decay": 0.5})

    optimizer.step([weight, bias])  # Both gradients are zero: only the L2 term acts.

    assert np.all(weight.value < 2.0)
    np.testing.assert_array_equal(bias.value, 2.0)


def test_sgd_weight_decay_adds_lambda_times_weight_to_the_gradient():
    weight, _ = weight_and_bias(2.0)
    weight.grad = np.full((2, 2), 1.0)

    build("optimizer", {"name": "sgd", "lr": 0.1, "weight_decay": 0.5}).step([weight])

    # w - lr * (grad + λ w) = 2 - 0.1 * (1 + 0.5 * 2) = 1.8
    np.testing.assert_allclose(weight.value, 1.8)


@pytest.mark.parametrize("name", ["sgd", "momentum", "adam"])
def test_zero_weight_decay_matches_the_unregularized_optimizer(name):
    rng = np.random.default_rng(0)
    first, second = Parameter("a.W", rng.normal(size=(3, 3))), None
    second = Parameter("a.W", first.value.copy())
    first.grad = second.grad = rng.normal(size=(3, 3))

    build("optimizer", {"name": name, "lr": 0.01}).step([first])
    build("optimizer", {"name": name, "lr": 0.01, "weight_decay": 0.0}).step([second])

    np.testing.assert_array_equal(first.value, second.value)


def test_negative_weight_decay_is_rejected():
    with pytest.raises(ValueError, match="weight_decay"):
        build("optimizer", {"name": "momentum", "weight_decay": -1e-4})


def test_he_initializer_has_variance_two_over_fan_in():
    fan_in = 400
    weights = build("initializer", "he", rng=np.random.default_rng(0))((fan_in, 300))

    assert weights.var() == pytest.approx(2 / fan_in, rel=0.02)
