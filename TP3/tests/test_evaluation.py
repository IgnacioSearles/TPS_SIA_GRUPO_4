import json

import numpy as np

from experiments.evaluation import classification_metrics, probability_metrics, save_results, save_weights
from nn import build_model


def test_binary_classification_metrics_with_bipolar_labels():
    metrics = classification_metrics(
        np.array([[-1.0], [-1.0], [1.0], [1.0]]),
        np.array([[-0.9], [0.2], [0.4], [0.8]]),
    )

    assert metrics["labels"] == [-1.0, 1.0]
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == 0.75


def test_multiclass_metrics_with_one_hot_targets():
    metrics = classification_metrics(
        np.eye(3)[[0, 1, 2]],
        np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.4, 0.5, 0.1]]),
    )

    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    assert metrics["accuracy"] == 2 / 3


def test_probability_metrics_compare_continuous_values_without_a_threshold():
    metrics = probability_metrics(
        np.array([[0.2], [0.8]]),
        np.array([[0.3], [0.6]]),
    )

    assert np.isclose(metrics["mae"], 0.15)
    assert np.isclose(metrics["rmse"], np.sqrt(0.025))
    assert "confusion_matrix" not in metrics


def test_save_weights_and_results(tmp_path):
    net = build_model({"layers": [1, 1], "activation": "identity"}, np.random.default_rng(0))
    weights_path = save_weights(net, tmp_path / "model.npz")
    results_path = save_results([{"epoch": 1, "loss": 0.2, "epoch_time": 0.01}], {"accuracy": 1.0}, tmp_path / "results.json")

    with np.load(weights_path) as saved:
        assert set(saved.files) == {"dense_0.W", "dense_0.b"}
    assert json.loads(results_path.read_text()) == {
        "history": [{"epoch": 1, "loss": 0.2, "epoch_time": 0.01}],
        "metrics": {"accuracy": 1.0},
    }
