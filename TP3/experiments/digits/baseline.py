"""Exercise 2 development experiment; never loads the external test dataset.

Run from TP3: python -m experiments.digits.baseline [config.json]
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.splits import stratified_holdout
from datasets.digit_dataset_loader import load_digit_arrays
from experiments.config import config_from_cli
from nn.network import build_model, load_weights, save_weights
from nn.registry import build
from training.callbacks import Callback, ProgressPrinter
from training.metrics import classification_metrics
from training.trainer import train

DEFAULTS = {
    "data": {"path": "datasets/digits.csv", "validation_ratio": 0.2, "split_seed": 42},
    "model": {"layers": [784, 32, 10], "activation": "tanh",
              "output_activation": "sigmoid", "initializer": "xavier"},
    "optimizer": {"name": "sgd", "lr": 0.1},
    "training": {"epochs": 50, "batch_size": 128},
    "output": {"directory": "results/digits_baseline"},
}


def one_hot(labels: np.ndarray) -> np.ndarray:
    """Targets zeta: column d is one exactly when the digit is d."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, np.arange(10)).all():
        raise ValueError("Expected a non-empty vector of integer digit labels 0..9")
    return np.eye(10)[labels.astype(int)]


def digit_metrics(labels: np.ndarray, outputs: np.ndarray) -> dict:
    """Fixed 10x10 matrix; undefined rates are null, macro uses supported classes.

    For supported classes without predictions precision is null and F1 is zero.
    Classes without true examples have null recall/F1 and are excluded from macro.
    """
    targets = one_hot(labels)
    if outputs.shape != targets.shape or not np.isfinite(outputs).all():
        raise ValueError("Expected finite outputs of shape (number of examples, 10)")
    result = classification_metrics(targets, outputs)
    matrix = np.zeros((10, 10), dtype=int)
    matrix[np.ix_(result["labels"], result["labels"])] = result["confusion_matrix"]
    rows = []
    for digit in range(10):
        support = int(matrix[digit].sum())
        predicted = int(matrix[:, digit].sum())
        correct = int(matrix[digit, digit])
        rows.append({"label": digit, "support": support,
                     "precision": correct / predicted if predicted else None,
                     "recall": correct / support if support else None,
                     "f1": 2 * correct / (support + predicted) if support else None})
    supported = [row for row in rows if row["support"]]
    return {"accuracy": result["accuracy"], "labels": list(range(10)),
            "confusion_matrix": matrix.tolist(), "per_class": rows,
            "macro_f1": float(np.mean([row["f1"] for row in supported])),
            "macro_labels": [row["label"] for row in supported],
            "loss": float(0.5 * np.sum((outputs - targets) ** 2) / len(labels))}


class ValidationRecorder(Callback):
    """Select by validation accuracy, optionally stop after validation stalls."""

    def __init__(self, net, X_train, labels_train, X_validation, labels_validation,
                 patience=None, min_delta=0.0):
        self.net = net
        self.X_train, self.labels_train = X_train, labels_train
        self.X_validation, self.labels_validation = X_validation, labels_validation
        self.best_key = None
        self.best_epoch = None
        self.best_weights = None
        self.patience = patience
        self.min_delta = min_delta
        self.epochs_without_accuracy_gain = 0
        self.stop_requested = False

    def on_epoch_end(self, logs):
        train_outputs = self.net.forward(self.X_train)
        validation = digit_metrics(self.labels_validation, self.net.forward(self.X_validation))
        logs.update(train_accuracy=float(np.mean(train_outputs.argmax(axis=1) == self.labels_train)),
                    validation_loss=validation["loss"], validation_accuracy=validation["accuracy"],
                    validation_macro_f1=validation["macro_f1"])
        if not np.isfinite(list(logs.values())).all():
            raise FloatingPointError("Non-finite training/validation metrics")
        key = (validation["accuracy"], -validation["loss"])
        if self.best_key is None or key > self.best_key:
            self.best_key, self.best_epoch = key, int(logs["epoch"])
            self.best_weights = [param.value.copy() for param in self.net.params()]
        if self.patience is not None:
            accuracy = validation["accuracy"]
            if accuracy > getattr(self, "best_stopping_accuracy", -1.0) + self.min_delta:
                self.best_stopping_accuracy = accuracy
                self.epochs_without_accuracy_gain = 0
            else:
                self.epochs_without_accuracy_gain += 1
                if self.epochs_without_accuracy_gain >= self.patience:
                    self.stop_requested = True

    def on_train_end(self, history):
        for param, value in zip(self.net.params(), self.best_weights):
            param.value[...] = value


def write_plots(history, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    for ax, keys, title in zip(axes, [("loss", "validation_loss"),
                                    ("train_accuracy", "validation_accuracy")],
                               ["Error cuadrático promedio", "Accuracy"]):
        for key, label in zip(keys, ["Entrenamiento", "Validación"]):
            ax.plot(epochs, [row[key] for row in history], label=label)
        ax.set(xlabel="Época", title=title)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)

    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    for actual in range(10):
        for predicted in range(10):
            ax.text(predicted, actual, str(matrix[actual, predicted]), ha="center", va="center",
                    color="white" if matrix[actual, predicted] > matrix.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Dígito predicho", ylabel="Dígito real",
           title="Validación — mejor época (filas vacías: sin ejemplos)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def run(config):
    data_path = Path(config["data"]["path"])
    if data_path.name != "digits.csv":
        raise ValueError("Exercise 2 development uses only digits.csv; external test stays reserved")
    if config["model"]["layers"][0] != 784 or config["model"]["layers"][-1] != 10:
        raise ValueError("Digit models require 784 inputs and 10 outputs")
    if config["loss"] != "mse":
        raise ValueError("This baseline reports and trains with mse")
    X, labels = load_digit_arrays(data_path)
    train_idx, validation_idx = stratified_holdout(
        labels, config["data"]["validation_ratio"], config["data"]["split_seed"])
    X_train, X_validation = X[train_idx], X[validation_idx]
    labels_train, labels_validation = labels[train_idx], labels[validation_idx]
    targets = one_hot(labels_train)
    rng = np.random.default_rng(config["seed"])
    net = build_model(config["model"], rng)
    initial = digit_metrics(labels_validation, net.forward(X_validation))
    counts = np.bincount(labels, minlength=10)
    missing = np.flatnonzero(counts == 0).tolist()
    print(f"Train: {len(train_idx)}; validation: {len(validation_idx)}; missing classes: {missing}", flush=True)
    early_stopping = config["training"].get("early_stopping", {})
    recorder = ValidationRecorder(
        net, X_train, labels_train, X_validation, labels_validation,
        patience=early_stopping.get("patience"),
        min_delta=early_stopping.get("min_delta", 0.0),
    )
    # Separate batch randomness from architecture-dependent initialization draws.
    shuffle_seed = config["training"].get("shuffle_seed")
    batch_rng = rng if shuffle_seed is None else np.random.default_rng(shuffle_seed)
    history = train(net, build("loss", config["loss"]), build("optimizer", config["optimizer"]),
                    X_train, targets, config["training"]["epochs"], config["training"]["batch_size"],
                    batch_rng, callbacks=[recorder, ProgressPrinter(every=5)])
    predictions = net.forward(X_validation)
    validation = digit_metrics(labels_validation, predictions)
    train_metrics = digit_metrics(labels_train, net.forward(X_train))
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    weights_path = save_weights(net, output / "best_weights.npz")
    restored = load_weights(build_model(config["model"], np.random.default_rng(0)), weights_path)
    np.testing.assert_array_equal(restored.forward(X_validation), predictions)
    np.savez(output / "split_indices.npz", train=train_idx, validation=validation_idx)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    pd.DataFrame({"row_index": validation_idx, "label": labels_validation,
                  "predicted": predictions.argmax(axis=1),
                  **{f"output_{d}": predictions[:, d] for d in range(10)}}).to_csv(
                      output / "validation_predictions.csv", index=False)
    summary = {"dataset": str(data_path.resolve()),
               "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
               "counts": counts.tolist(), "missing_classes": missing,
               "train_counts": np.bincount(labels_train, minlength=10).tolist(),
               "validation_counts": np.bincount(labels_validation, minlength=10).tolist(),
               "selection": "highest validation accuracy, tie: lowest validation loss",
               "best_epoch": recorder.best_epoch, "epochs_ran": len(history),
               "initial_validation": initial,
               "train": train_metrics, "validation": validation,
               "weights_reload_verified": True, "external_test_evaluated": False}
    for name, value in [("summary.json", summary), ("config.json", config)]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_plots(history, validation, output)
    print(f"Best epoch: {recorder.best_epoch}; validation accuracy: {validation['accuracy']:.4%}", flush=True)
    print(f"Results: {output.resolve()}", flush=True)
    return summary


if __name__ == "__main__":
    run(config_from_cli(DEFAULTS))
