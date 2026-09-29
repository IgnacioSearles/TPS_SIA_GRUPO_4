"""Validation analysis and fraud-threshold recommendation for the distilled TinyModel.

Reads the outputs of `experiments.fraud.probability` (it never trains anything):
the out-of-fold predictions on the development set and the final model's test
predictions. The threshold is chosen on the out-of-fold predictions only; the
test set is used once, to check the already-chosen threshold.

Usage:
    python -m experiments.fraud.validation_analysis [path/to/config.json]

Every config key is optional (see DEFAULTS).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.config import Config, deep_merge, read_json
from analysis.plotting import save_figure
from experiments.fraud.data import FRAUD_LABEL, TEACHER_TARGET, load_labeled_fraud_dataset
from training.metrics import (
    average_precision,
    binary_metrics,
    f_beta,
    precision_recall_curve,
    roc_auc,
    roc_curve,
    threshold_sweep,
)

DEFAULTS: Config = {
    "dataset": "datasets/fraud_dataset.csv",
    "results_directory": "reports/fraud_probability",
    "output_directory": "reports/fraud_probability/validation",
    "threshold": {
        # beta > 1 favors catching fraud (recall); beta < 1 favors fewer false alarms (precision).
        "beta": 1.0,
        "grid_step": 0.001,
        # BigModel's own cutoff: flagged_fraud == (big_model_fraud_probability > 0.85).
        "reference": 0.85,
    },
}
TARGET_BINS = np.linspace(0, 1, 21)


@dataclass
class Predictions:
    """TinyModel predictions for one set of rows, joined with BigModel's probability and label."""

    name: str
    target: np.ndarray
    predicted: np.ndarray
    label: np.ndarray
    fold: np.ndarray | None = None


def load_predictions(config: Config) -> tuple[Predictions, Predictions]:
    """Out-of-fold (validation) and test predictions, with the fraud label rejoined by row."""
    results = Path(config["results_directory"])
    dataset = load_labeled_fraud_dataset(config["dataset"])
    sets = []
    for name, filename in (("validación (out-of-fold)", "out_of_fold_predictions.csv"),
                           ("test", "test_predictions.csv")):
        rows = pd.read_csv(results / filename)
        source = dataset.iloc[rows["row"].to_numpy()]
        # Fail fast if the predictions were produced from a different dataset or row order.
        if not np.allclose(source[TEACHER_TARGET].to_numpy(), rows["target_probability"].to_numpy()):
            raise ValueError(f"{filename} does not match {config['dataset']} row by row")
        sets.append(Predictions(
            name=name,
            target=rows["target_probability"].to_numpy(),
            predicted=rows["predicted_probability"].to_numpy(),
            label=source[FRAUD_LABEL].to_numpy(),
            fold=rows["fold"].to_numpy() if "fold" in rows else None,
        ))
    return sets[0], sets[1]


def choose_threshold(validation: Predictions, beta: float, grid_step: float) -> tuple[float, pd.DataFrame]:
    """Threshold that maximizes F-beta on validation predictions, plus the full sweep."""
    thresholds = np.round(np.arange(0, 1 + grid_step / 2, grid_step), 10)
    sweep = pd.DataFrame(threshold_sweep(validation.label, validation.predicted, thresholds))
    sweep["f_beta"] = f_beta(sweep["precision"].to_numpy(), sweep["recall"].to_numpy(), beta)
    best = sweep.loc[sweep["f_beta"].idxmax()]
    return float(best["threshold"]), sweep


def evaluate(predictions: Predictions, thresholds: dict[str, float], beta: float) -> dict[str, Any]:
    """Ranking metrics plus decision metrics at each named threshold."""
    at_threshold = {}
    for name, threshold in thresholds.items():
        metrics = binary_metrics(predictions.label, predictions.predicted, threshold)
        metrics["f_beta"] = float(f_beta(np.array(metrics["precision"]), np.array(metrics["recall"]), beta))
        at_threshold[name] = metrics
    error = predictions.predicted - predictions.target
    return {
        "samples": int(len(predictions.label)),
        "fraud_rate": float(predictions.label.mean()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "roc_auc": roc_auc(predictions.label, predictions.predicted),
        "average_precision": average_precision(predictions.label, predictions.predicted),
        "at_threshold": at_threshold,
    }


def binned_mean(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center, mean and standard deviation of y inside each non-empty bin of x."""
    frame = pd.DataFrame({"bin": pd.cut(x, bins, include_lowest=True), "x": x, "y": y})
    grouped = frame.groupby("bin", observed=True)
    return grouped["x"].mean().to_numpy(), grouped["y"].mean().to_numpy(), grouped["y"].std().fillna(0).to_numpy()


def plot_predicted_vs_target(sets: list[Predictions], threshold: float, reference: float, path: Path) -> None:
    figure, axes = plt.subplots(1, len(sets), figsize=(6 * len(sets), 5.5), sharey=True, layout="constrained")
    for axis, predictions in zip(axes, sets):
        axis.scatter(predictions.target, predictions.predicted, s=4, alpha=0.15, label="transacción")
        centers, means, _ = binned_mean(predictions.target, predictions.predicted, TARGET_BINS)
        axis.plot(centers, means, "o-", color="C1", markersize=4, label="media de TinyModel por bin")
        axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="predicción perfecta")
        axis.axvline(reference, color="C3", linestyle=":", label=f"corte de BigModel ({reference:g})")
        axis.axhline(threshold, color="C2", linestyle=":", label=f"umbral elegido ({threshold:.3f})")
        axis.set(xlim=(0, 1), ylim=(0, 1), title=predictions.name,
                 xlabel=f"{TEACHER_TARGET} (BigModel)", ylabel="probabilidad de TinyModel")
    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle("TinyModel vs. BigModel en datos no vistos")
    save_figure(figure, path)


def plot_error_by_target(sets: list[Predictions], path: Path) -> None:
    figure, (histogram_axis, bias_axis) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    bins = np.linspace(-0.6, 0.6, 61)
    for index, predictions in enumerate(sets):
        error = predictions.predicted - predictions.target
        histogram_axis.hist(error, bins=bins, density=True, histtype="step", linewidth=2,
                            color=f"C{index}", label=predictions.name)
        centers, means, stds = binned_mean(predictions.target, error, TARGET_BINS)
        bias_axis.plot(centers, means, "o-", color=f"C{index}", markersize=4, label=predictions.name)
        bias_axis.fill_between(centers, means - stds, means + stds, color=f"C{index}", alpha=0.15)
    histogram_axis.axvline(0, color="black", linewidth=1)
    histogram_axis.set(title="Distribución del error", xlabel="TinyModel − BigModel", ylabel="densidad")
    histogram_axis.legend()
    bias_axis.axhline(0, color="black", linewidth=1)
    bias_axis.set(title="Error medio (± 1 desvío) según la probabilidad de BigModel",
                  xlabel=f"{TEACHER_TARGET} (BigModel)", ylabel="TinyModel − BigModel")
    bias_axis.legend()
    save_figure(figure, path)


def plot_fold_errors(summary: dict[str, Any], path: Path) -> None:
    folds = pd.DataFrame(summary["cross_validation"]["results"])
    test_metrics = summary["final_model"]["test_metrics"]
    positions = np.arange(len(folds))
    baseline = summary["cross_validation"]["baseline_oof_rmse"]
    # Points (not bars) so the axis can zoom in on the small train/validation differences.
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(positions - 0.08, folds["train_rmse"], "o", markersize=9, label="RMSE train")
    axis.plot(positions + 0.08, folds["validation_rmse"], "s", markersize=9, label="RMSE validación")
    axis.axhline(test_metrics["rmse"], color="C3", linestyle="--", label=f"RMSE test ({test_metrics['rmse']:.4f})")
    axis.set_xticks(positions, [f"fold {fold}" for fold in folds["fold"]])
    axis.set(ylabel="RMSE de probabilidad",
             title=f"Error por fold y en test (baseline constante: RMSE {baseline:.3f})")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    save_figure(figure, path)


def plot_score_distribution(predictions: Predictions, threshold: float, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 51)
    for label, name in ((0, "legítima"), (1, "fraude")):
        axis.hist(predictions.predicted[predictions.label == label], bins=bins,
                  histtype="step", linewidth=2, label=f"{name} ({FRAUD_LABEL}={label})")
    axis.axvline(threshold, color="C2", linestyle="--", label=f"umbral elegido ({threshold:.3f})")
    # Log scale: the fraud rows pile up near 1 and would otherwise flatten the legitimate class.
    axis.set_yscale("log")
    axis.set(xlabel="probabilidad de TinyModel", ylabel="transacciones (escala log)",
             title=f"Probabilidad de TinyModel por clase — {predictions.name}")
    axis.legend()
    save_figure(figure, path)


def plot_threshold_metrics(sweep: pd.DataFrame, beta: float, threshold: float, reference: float, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(sweep["threshold"], sweep["precision"], label="precisión")
    axis.plot(sweep["threshold"], sweep["recall"], label="recall")
    axis.plot(sweep["threshold"], sweep["f1"], label="F1")
    if beta != 1:
        axis.plot(sweep["threshold"], sweep["f_beta"], label=f"F{beta:g}")
    axis.plot(sweep["threshold"], sweep["flagged_rate"], color="gray", linestyle=":", label="fracción marcada")
    axis.axvline(threshold, color="C2", linestyle="--", label=f"umbral elegido ({threshold:.3f})")
    axis.axvline(reference, color="C3", linestyle=":", label=f"corte de BigModel ({reference:g})")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="umbral sobre la probabilidad de TinyModel", ylabel="valor",
             title="Métricas de detección según el umbral (validación out-of-fold)")
    axis.legend(loc="center left", fontsize=8)
    save_figure(figure, path)


def plot_roc_and_pr(sets: list[Predictions], threshold: float, path: Path) -> None:
    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for index, predictions in enumerate(sets):
        color = f"C{index}"
        fpr, tpr = roc_curve(predictions.label, predictions.predicted)
        precision, recall = precision_recall_curve(predictions.label, predictions.predicted)
        point = binary_metrics(predictions.label, predictions.predicted, threshold)
        roc_axis.plot(fpr, tpr, color=color,
                      label=f"{predictions.name} (AUC {roc_auc(predictions.label, predictions.predicted):.3f})")
        marker = "os"[index]
        roc_axis.plot(point["false_positive_rate"], point["recall"], marker, color=color, markersize=9,
                      markerfacecolor="none", markeredgewidth=2)
        pr_axis.plot(recall, precision, color=color,
                     label=f"{predictions.name} (AP {average_precision(predictions.label, predictions.predicted):.3f})")
        pr_axis.plot(point["recall"], point["precision"], marker, color=color, markersize=9,
                     markerfacecolor="none", markeredgewidth=2)
    roc_axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="azar")
    roc_axis.set(xlabel="tasa de falsos positivos", ylabel="recall (tasa de verdaderos positivos)", title="Curva ROC")
    pr_axis.axhline(sets[0].label.mean(), color="gray", linestyle=":", label="azar (tasa de fraude)")
    pr_axis.set(xlabel="recall", ylabel="precisión", title="Curva precisión-recall")
    for axis in (roc_axis, pr_axis):
        axis.legend(loc="lower right" if axis is roc_axis else "lower left", fontsize=8)
    figure.suptitle(f"Puntos marcados: umbral elegido {threshold:.3f}")
    save_figure(figure, path)


def plot_confusion_matrices(sets: list[Predictions], threshold: float, path: Path) -> None:
    figure, axes = plt.subplots(1, len(sets), figsize=(5.5 * len(sets), 4.5), layout="constrained")
    class_names = ["legítima", "fraude"]
    for axis, predictions in zip(axes, sets):
        metrics = binary_metrics(predictions.label, predictions.predicted, threshold)
        matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
        row_share = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
        for (row, column), count in np.ndenumerate(matrix):
            axis.text(column, row, f"{count:.0f}\n({row_share[row, column]:.1%})", ha="center", va="center",
                      color="white" if row_share[row, column] > 0.5 else "black")
        axis.set_xticks([0, 1], class_names)
        axis.set_yticks([0, 1], class_names)
        axis.set(xlabel="predicción de TinyModel", ylabel=f"{FRAUD_LABEL} (BigModel)", title=predictions.name)
    figure.suptitle(f"Matrices de confusión con umbral {threshold:.3f}")
    save_figure(figure, path)


def run_analysis(config: Config) -> dict[str, Any]:
    """Choose the threshold on validation, evaluate it on test, and write figures and a summary."""
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    beta = float(config["threshold"]["beta"])
    reference = float(config["threshold"]["reference"])
    validation, test = load_predictions(config)
    threshold, sweep = choose_threshold(validation, beta, float(config["threshold"]["grid_step"]))
    sweep.to_csv(output / "threshold_sweep.csv", index=False)

    thresholds = {"recommended": threshold, "bigmodel_cutoff": reference}
    summary = {
        "recommended_threshold": threshold,
        "selection": {"criterion": f"max F{beta:g} on validation (out-of-fold)", "beta": beta},
        "validation": evaluate(validation, thresholds, beta),
        "test": evaluate(test, thresholds, beta),
    }
    with (output / "threshold_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with (Path(config["results_directory"]) / "summary.json").open(encoding="utf-8") as file:
        experiment_summary = json.load(file)
    sets = [validation, test]
    plot_predicted_vs_target(sets, threshold, reference, output / "predicted_vs_target.png")
    plot_error_by_target(sets, output / "error_by_target.png")
    plot_fold_errors(experiment_summary, output / "fold_errors.png")
    plot_score_distribution(validation, threshold, output / "score_distribution_by_class.png")
    plot_threshold_metrics(sweep, beta, threshold, reference, output / "threshold_metrics.png")
    plot_roc_and_pr(sets, threshold, output / "roc_pr_curves.png")
    plot_confusion_matrices(sets, threshold, output / "confusion_matrices.png")
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Recommended threshold: {summary['recommended_threshold']:.3f} ({summary['selection']['criterion']})")
    for set_name in ("validation", "test"):
        result = summary[set_name]
        print(f"\n{set_name}: RMSE {result['rmse']:.4f}  ROC AUC {result['roc_auc']:.4f}  AP {result['average_precision']:.4f}")
        for name, metrics in result["at_threshold"].items():
            print(f"  {name:>16} t={metrics['threshold']:.3f}  precision {metrics['precision']:.3f}  "
                  f"recall {metrics['recall']:.3f}  F1 {metrics['f1']:.3f}  flagged {metrics['flagged_rate']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation analysis and fraud threshold for TinyModel")
    parser.add_argument("config", nargs="?", help="optional JSON config; omitted keys use defaults")
    args = parser.parse_args()
    config = deep_merge(DEFAULTS, read_json(args.config)) if args.config else DEFAULTS
    summary = run_analysis(config)
    print_summary(summary)
    print(f"\nFigures written to {config['output_directory']}")


if __name__ == "__main__":
    main()
