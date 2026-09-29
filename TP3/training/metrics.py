"""Evaluation metrics on NumPy arrays: regression error, class predictions and thresholded scores.

For thresholded scores, a row is predicted positive when `score >= threshold`.
Sweeps are vectorized: they count positives above each threshold with a binary
search on sorted scores.
"""

from typing import Any

import numpy as np


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


def _validate(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten inputs to 1-D and check they describe the same non-empty binary problem."""
    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError(f"Expected non-empty matching sizes, got {labels.shape} and {scores.shape}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("y_true must only contain 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    positives = labels.astype(bool)
    if positives.all() or not positives.any():
        raise ValueError("y_true must contain both classes")
    return positives, scores


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """numerator / denominator, with 0 where the denominator is 0 (e.g. precision with no flagged rows)."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def f_beta(precision: np.ndarray, recall: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Weighted harmonic mean of precision and recall; beta > 1 values recall more."""
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    beta_squared = beta ** 2
    return _safe_ratio((1 + beta_squared) * precision * recall, beta_squared * precision + recall)


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """Confusion counts and rates for every threshold in `thresholds`."""
    positives, scores = _validate(y_true, scores)
    thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
    positive_scores = np.sort(scores[positives])
    negative_scores = np.sort(scores[~positives])
    # Rows with score >= t are the ones to the right of t's insertion point.
    tp = len(positive_scores) - np.searchsorted(positive_scores, thresholds, side="left")
    fp = len(negative_scores) - np.searchsorted(negative_scores, thresholds, side="left")
    fn = len(positive_scores) - tp
    tn = len(negative_scores) - fp
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "threshold": thresholds,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": _safe_ratio(fp, fp + tn),
        "accuracy": (tp + tn) / len(scores),
        "flagged_rate": (tp + fp) / len(scores),
        "f1": f_beta(precision, recall, 1.0),
    }


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """All sweep metrics at a single threshold, as plain floats."""
    sweep = threshold_sweep(y_true, scores, np.array([threshold]))
    return {name: float(values[0]) for name, values in sweep.items()}


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(false positive rate, true positive rate) at every distinct score, starting from (0, 0)."""
    positives, scores = _validate(y_true, scores)
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(positives[order])
    fp = np.cumsum(~positives[order])
    # Keep only the last row of each group of tied scores: ties are flagged together.
    last_of_group = np.r_[np.diff(scores[order]) != 0, True]
    tpr = np.r_[0.0, tp[last_of_group] / positives.sum()]
    fpr = np.r_[0.0, fp[last_of_group] / (~positives).sum()]
    return fpr, tpr


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve: chance that a random positive outscores a random negative."""
    fpr, tpr = roc_curve(y_true, scores)
    return float(np.trapezoid(tpr, fpr))


def precision_recall_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(precision, recall) at every distinct score, from the strictest threshold to the loosest."""
    positives, scores = _validate(y_true, scores)
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(positives[order])
    flagged = np.arange(1, len(scores) + 1)
    last_of_group = np.r_[np.diff(scores[order]) != 0, True]
    return tp[last_of_group] / flagged[last_of_group], tp[last_of_group] / positives.sum()


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, as in the standard definition)."""
    precision, recall = precision_recall_curve(y_true, scores)
    recall_steps = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_steps * precision))
