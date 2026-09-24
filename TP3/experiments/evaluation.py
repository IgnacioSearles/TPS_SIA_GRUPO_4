"""Classification metrics and optional persistence for experiments."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from nn.network import Sequential
from training.callbacks import EpochLogs


def probability_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Measure how closely continuous predictions match target probabilities."""
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError(f"Expected non-empty matching shapes, got {y_true.shape} and {y_pred.shape}")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Probabilities must be finite")
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def _as_class_labels(values: np.ndarray, *, name: str) -> np.ndarray:
    """Convert scalar binary targets/predictions or one-hot rows to class labels."""
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError(f"{name} must be a vector or a 2-D matrix, got shape {values.shape}")
    if values.shape[1] == 1:
        return values[:, 0]
    return np.argmax(values, axis=1)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return accuracy, confusion matrix and precision/recall/F1 for class outputs.

    Single-output binary predictions are thresholded halfway between the two
    sorted target labels. Multi-output predictions and one-hot targets use argmax.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    true = _as_class_labels(y_true, name="y_true")
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        labels = np.unique(true)
        if len(labels) != 2:
            raise ValueError("Binary classification requires exactly two target labels")
        threshold = (float(labels[0]) + float(labels[1])) / 2.0
        predicted = np.where(y_pred[:, 0] >= threshold, labels[1], labels[0])
    else:
        predicted = _as_class_labels(y_pred, name="y_pred")
        labels = np.unique(np.concatenate((true, predicted)))

    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_index = {label: index for index, label in enumerate(labels.tolist())}
    for actual, estimated in zip(true, predicted):
        matrix[label_index[actual], label_index[estimated]] += 1

    per_class = []
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"label": label.item() if hasattr(label, "item") else label,
                          "precision": precision, "recall": recall, "f1": f1, "support": int(matrix[index, :].sum())})

    return {
        "accuracy": float(np.mean(true == predicted)),
        "labels": [item["label"] for item in per_class],
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "macro_avg": {
            "precision": float(np.mean([item["precision"] for item in per_class])),
            "recall": float(np.mean([item["recall"] for item in per_class])),
            "f1": float(np.mean([item["f1"] for item in per_class])),
        },
    }


def save_weights(net: Sequential, path: str | Path) -> Path:
    """Persist all trainable parameters as a portable NumPy .npz archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{param.name: param.value for param in net.params()})
    return path


def save_results(history: list[EpochLogs], metrics: dict[str, Any] | None, path: str | Path) -> Path:
    """Persist training history and optional evaluation metrics as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump({"history": history, "metrics": metrics}, file, indent=2)
    return path
