import json

import numpy as np
import pandas as pd
import pytest

from experiments.config import load_config
from experiments.digits.baseline import DEFAULTS, ValidationRecorder, digit_metrics, one_hot, run
from nn.initializers import XavierUniform
from nn.layers import Parameter


def test_one_hot_preserves_digit_column_and_rejects_invalid_labels():
    targets = one_hot(np.array([9, 0, 5]))
    assert targets.shape == (3, 10)
    np.testing.assert_array_equal(targets.argmax(axis=1), [9, 0, 5])
    np.testing.assert_array_equal(targets.sum(axis=1), [1, 1, 1])
    for invalid in [[], [10], [-1], [1.5], [[1]], [float("nan")]]:
        with pytest.raises(ValueError):
            one_hot(np.array(invalid))


def test_missing_classes_and_unpredicted_supported_class():
    result = digit_metrics(np.array([0, 5, 9]), one_hot(np.array([0, 0, 9])))
    matrix = np.array(result["confusion_matrix"])
    assert matrix.shape == (10, 10)
    assert matrix[5, 0] == 1 and matrix[9, 9] == 1
    assert result["per_class"][8]["recall"] is None
    assert result["per_class"][8]["f1"] is None
    assert result["per_class"][5]["recall"] == 0
    assert result["per_class"][5]["precision"] is None
    assert result["macro_labels"] == [0, 5, 9]
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0 + 1) / 3)


def test_recorder_restores_best_epoch_not_last_and_breaks_ties_by_loss():
    class Net:
        parameter = Parameter("weight", np.array([1.0]))
        outputs = one_hot(np.array([0, 1])) * 0.7

        def forward(self, X):
            return self.outputs.copy()

        def params(self):
            return [self.parameter]

    net = Net()
    X, labels = np.zeros((2, 1)), np.array([0, 1])
    recorder = ValidationRecorder(net, X, labels, X, labels)
    recorder.on_epoch_end({"epoch": 1, "loss": 0.1})
    net.parameter.value[:] = 2
    net.outputs = one_hot(labels) * 0.9  # Same accuracy, smaller loss.
    recorder.on_epoch_end({"epoch": 2, "loss": 0.01})
    net.parameter.value[:] = 3
    net.outputs = one_hot(np.array([1, 0]))
    recorder.on_epoch_end({"epoch": 3, "loss": 1.0})
    recorder.on_train_end([])
    assert recorder.best_epoch == 2
    np.testing.assert_array_equal(net.parameter.value, [2])


def test_xavier_reproducibility_scale_and_shape():
    shape = (784, 32)
    first = XavierUniform(np.random.default_rng(42))(shape)
    np.testing.assert_array_equal(first, XavierUniform(np.random.default_rng(42))(shape))
    assert abs(first).max() <= np.sqrt(6 / sum(shape))
    assert first.std() == pytest.approx(np.sqrt(2 / sum(shape)), rel=0.03)
    with pytest.raises(ValueError):
        XavierUniform(np.random.default_rng(42))((0, 10))


def test_small_end_to_end_run_persists_disjoint_split_and_model(tmp_path):
    labels = np.tile([0, 5, 9], 10)
    X = np.zeros((len(labels), 784))
    X[np.arange(len(labels)), labels] = 1
    dataset = tmp_path / "digits.csv"
    pd.DataFrame({"label": labels, "image": [json.dumps(row.tolist()) for row in X]}).to_csv(dataset, index=False)
    config = load_config(experiment_defaults=DEFAULTS)
    config["data"] = {"path": str(dataset), "validation_ratio": 0.2, "split_seed": 42}
    config["training"] = {"epochs": 3, "batch_size": 8}
    config["output"] = {"directory": str(tmp_path / "results")}
    result = run(config)
    assert result["external_test_evaluated"] is False
    assert result["weights_reload_verified"] is True
    assert result["validation"]["per_class"][8]["recall"] is None
    with np.load(tmp_path / "results" / "split_indices.npz") as split:
        assert not np.intersect1d(split["train"], split["validation"]).size
        np.testing.assert_array_equal(np.sort(np.r_[split["train"], split["validation"]]), np.arange(len(X)))
    assert (tmp_path / "results" / "learning_curves.png").is_file()
    config["data"]["path"] = str(tmp_path / "digits_test.csv")
    with pytest.raises(ValueError, match="only digits.csv"):
        run(config)
