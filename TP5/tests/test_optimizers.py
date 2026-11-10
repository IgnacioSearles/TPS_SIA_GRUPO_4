import numpy as np
import pytest

from src.nn.optimizers import build_optimizer
from src.options import ConfigurationError, Options

TARGET = np.array([3.0, -2.0])


@pytest.mark.parametrize(
    "settings",
    [
        {"type": "sgd", "learning_rate": 0.1},
        {"type": "momentum", "learning_rate": 0.05, "momentum": 0.8},
        {"type": "rmsprop", "learning_rate": 0.01},
        {"type": "adam", "learning_rate": 0.1},
    ],
)
def test_optimizer_minimizes_quadratic(settings):
    optimizer = build_optimizer(Options(settings, "optimizer"))
    parameter = np.zeros(2)
    for _ in range(1000):
        optimizer.step([parameter], [2.0 * (parameter - TARGET)])
    np.testing.assert_allclose(parameter, TARGET, atol=0.05)


def test_unknown_hyperparameter_is_rejected():
    with pytest.raises(ConfigurationError):
        build_optimizer(Options({"type": "sgd", "learning_rate": 0.1, "beta1": 0.9}, "optimizer"))


def test_non_positive_learning_rate_is_rejected():
    with pytest.raises(ConfigurationError):
        build_optimizer(Options({"type": "adam", "learning_rate": 0}, "optimizer"))
