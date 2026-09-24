"""Validation tasks from the handout that the current components can already solve."""

import numpy as np

from experiments.config import load_config
from experiments.runner import run_training

SILENT = {"callbacks": []}


def test_step_perceptron_learns_and():
    X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    Y = np.array([[-1], [-1], [-1], [1]], dtype=float)
    config = load_config(experiment_defaults={
        **SILENT,
        "model": {"layers": [2, 1], "activation": "step"},
        "optimizer": {"lr": 0.1},
        "training": {"epochs": 50, "batch_size": 1},
    })

    result = run_training(config, X, Y)

    np.testing.assert_array_equal(result.net.forward(X), Y)


def test_linear_perceptron_fits_identity():
    X = np.linspace(-1, 1, 50).reshape(-1, 1)
    Y = X.copy()
    config = load_config(experiment_defaults={
        **SILENT,
        "model": {"layers": [1, 1], "activation": "identity"},
        "optimizer": {"lr": 0.1},
        "training": {"epochs": 200, "batch_size": 10},
    })

    result = run_training(config, X, Y)

    assert result.history[-1]["loss"] < 1e-6
