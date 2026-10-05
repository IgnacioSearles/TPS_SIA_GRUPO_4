"""Compare asymmetric probability losses without training on flagged_fraud.

Usage: python -m experiments.study_fraud_asymmetric experiments/configs/fraud_asymmetric.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nn  # noqa: F401 (component registration)
from experiments.fraud_data import FEATURE_COLUMNS, TEACHER_TARGET, kfold, load_fraud_dataset, prepare_fold
from nn.network import build_model
from nn.registry import build
from training.trainer import train_epoch


class AsymmetricMSE:
    """MSE/2 with separate penalties for predictions above and below BigModel."""

    def __init__(self, over_weight: float, under_weight: float):
        if not np.isfinite([over_weight, under_weight]).all() or min(over_weight, under_weight) <= 0:
            raise ValueError("over_weight and under_weight must be finite and positive")
        self.over_weight = over_weight
        self.under_weight = under_weight
        self.gradient: np.ndarray | None = None

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        if prediction.shape != target.shape or prediction.size == 0:
            raise ValueError("prediction and target need matching non-empty shapes")
        error = prediction - target
        weight = np.where(error >= 0, self.over_weight, self.under_weight)
        self.gradient = weight * error
        return float(0.5 * np.sum(weight * error ** 2) / len(target))

    def backward(self) -> np.ndarray:
        if self.gradient is None:
            raise RuntimeError("forward must run before backward")
        return self.gradient


def measures(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & y))
    fp = int(np.sum(predicted & ~y))
    fn = int(np.sum(~predicted & y))
    tn = int(np.sum(~predicted & ~y))
    return {
        "threshold": float(threshold), "flags": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def best_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Maximize F1; break a tie in favor of fewer alerts."""
    order = np.argsort(-scores, kind="stable")
    labels = y[order]
    tp = np.cumsum(labels)
    fp = np.arange(1, len(labels) + 1) - tp
    fn = labels.sum() - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    return float(scores[order[np.argmax(f1)]])


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    positives = y[order]
    if not positives.any():
        return 0.0
    precision_at_rank = np.cumsum(positives) / np.arange(1, len(y) + 1)
    return float(precision_at_rank[positives].mean())


def crossfit_threshold(y: np.ndarray, scores: np.ndarray, fold_ids: np.ndarray) -> tuple[dict, list[float]]:
    predictions = np.zeros(len(y), dtype=bool)
    thresholds = []
    for fold in np.unique(fold_ids):
        train = fold_ids != fold
        validation = ~train
        threshold = best_threshold(y[train], scores[train])
        thresholds.append(threshold)
        predictions[validation] = scores[validation] >= threshold
    # Binary predictions can be passed as 0/1 scores with threshold 0.5.
    return measures(y, predictions.astype(float), 0.5), thresholds


def save_precision_recall_plot(y: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    for name, scores in predictions.items():
        order = np.argsort(-scores, kind="stable")
        tp = np.cumsum(y[order])
        recall = tp / y.sum()
        precision = tp / np.arange(1, len(y) + 1)
        axis.plot(recall, precision, label=f"{name} (AP={average_precision(y, scores):.4f})")
    axis.set(xlabel="Recall frente a la regla BigModel", ylabel="Precisión",
             title="Pérdida simétrica y asimétrica")
    axis.set_xlim(0.6, 1.0)
    axis.set_ylim(0.4, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(config: dict) -> None:
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    data = load_fraud_dataset(config["dataset"])
    teacher = data[TEACHER_TARGET].to_numpy(dtype=float)
    folds = kfold(data, n_splits=int(config["folds"]), seed=int(config["seed"]))
    fold_ids = np.zeros(len(data), dtype=int)
    for number, fold in enumerate(folds, 1):
        fold_ids[fold.validation] = number

    predictions = {}
    curves = []
    for variant in config["variants"]:
        name = variant["name"]
        oof = np.full(len(data), np.nan)
        for fold_number, fold in enumerate(folds, 1):
            prepared = prepare_fold(data, fold)
            rng = np.random.default_rng(int(config["seed"]))
            net = build_model({"layers": [len(FEATURE_COLUMNS), 1], "activation": "sigmoid"}, rng)
            optimizer = build("optimizer", {"name": "sgd", "lr": float(config["learning_rate"])})
            loss = AsymmetricMSE(float(variant["over_weight"]), float(variant["under_weight"]))
            for epoch in range(1, int(config["epochs"]) + 1):
                train_epoch(net, loss, optimizer, prepared.X_train, prepared.teacher_train,
                            int(config["batch_size"]), rng)
                if epoch == 1 or epoch % int(config["log_every"]) == 0 or epoch == int(config["epochs"]):
                    estimate = net.forward(prepared.X_validation)[:, 0]
                    truth = prepared.teacher_validation[:, 0]
                    curves.append({"variant": name, "fold": fold_number, "epoch": epoch,
                                   "validation_mae": float(np.mean(np.abs(estimate - truth))),
                                   "validation_rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
                                   "validation_bias": float(np.mean(estimate - truth))})
            oof[fold.validation] = net.forward(prepared.X_validation)[:, 0]
            print(f"{name}: fold {fold_number}/{len(folds)} complete", flush=True)
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Incomplete out-of-fold predictions for {name}")
        predictions[name] = oof

    # The documented outcome is loaded only after all model fitting is finished.
    labels = pd.read_csv(config["dataset"], usecols=["flagged_fraud"])["flagged_fraud"].to_numpy(dtype=bool)
    if not np.array_equal(labels, teacher >= float(config["big_threshold"])):
        print("WARNING: flagged_fraud differs from BigModel's threshold rule", flush=True)
    rows = []
    for variant in config["variants"]:
        name = variant["name"]
        scores = predictions[name]
        crossfit, thresholds = crossfit_threshold(labels, scores, fold_ids)
        error = scores - teacher
        rows.append({
            "variant": name, "over_weight": variant["over_weight"], "under_weight": variant["under_weight"],
            "teacher_mae": float(np.mean(np.abs(error))),
            "teacher_rmse": float(np.sqrt(np.mean(error ** 2))),
            "teacher_bias": float(np.mean(error)),
            "teacher_high_bias": float(np.mean(error[teacher >= float(config["big_threshold"])])),
            "average_precision": average_precision(labels, scores),
            "at_085_precision": measures(labels, scores, 0.85)["precision"],
            "at_085_recall": measures(labels, scores, 0.85)["recall"],
            "at_085_f1": measures(labels, scores, 0.85)["f1"],
            "best_oof_threshold": best_threshold(labels, scores),
            "best_oof_f1": measures(labels, scores, best_threshold(labels, scores))["f1"],
            "crossfit_threshold_min": min(thresholds),
            "crossfit_threshold_max": max(thresholds),
            "crossfit_precision": crossfit["precision"],
            "crossfit_recall": crossfit["recall"],
            "crossfit_f1": crossfit["f1"],
        })
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    pd.DataFrame({"row": np.arange(len(data)), "fold": fold_ids, "teacher": teacher,
                  "flagged_fraud": labels.astype(int), **predictions}).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(curves).to_csv(output / "learning_curves.csv", index=False)
    save_precision_recall_plot(labels, predictions, output / "precision_recall.png")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="experiments/configs/fraud_asymmetric.json")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
