"""Compare the generalization runs of two (or more) output activations for Exercise 1.

Reads the result directories written by `experiments.fraud.probability` and
`experiments.fraud.validation_analysis` (it never trains anything).

Usage:
    python -m experiments.fraud.activation_comparison [path/to/config.json]

Every config key is optional (see DEFAULTS).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analysis.plotting import save_figure
from experiments.config import Config, deep_merge, read_json
from experiments.fraud.data import TEACHER_TARGET, load_fraud_dataset
from experiments.fraud.probability import load_probability_model

DEFAULTS: Config = {
    "dataset": "datasets/fraud_dataset.csv",
    # Activation name -> directory with its probability and validation results.
    "runs": {
        "sigmoid": "reports/fraud_probability",
        "relu": "reports/fraud_probability_relu",
    },
    "output_directory": "reports/fraud_activation_comparison",
}


def read_json_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def comparison_table(runs: dict[str, str]) -> pd.DataFrame:
    """One row per activation with its error, validity and detection metrics."""
    rows = []
    for activation, directory in runs.items():
        results = Path(directory)
        summary = read_json_file(results / "summary.json")
        thresholds = read_json_file(results / "validation" / "threshold_summary.json")
        test_at_threshold = thresholds["test"]["at_threshold"]["recommended"]
        rows.append({
            "activation": activation,
            "train_rmse_all_samples": summary["learning_comparison"][activation]["rmse"],
            "validation_rmse": summary["cross_validation"]["rmse_mean"],
            "test_rmse": summary["final_model"]["test_metrics"]["rmse"],
            "test_outside_0_1": summary["final_model"]["test_outside_0_1"],
            "test_roc_auc": thresholds["test"]["roc_auc"],
            "recommended_threshold": thresholds["recommended_threshold"],
            "test_precision": test_at_threshold["precision"],
            "test_recall": test_at_threshold["recall"],
            "test_f1": test_at_threshold["f1"],
        })
    return pd.DataFrame(rows)


def test_pre_activations(directory: str, dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, Any]:
    """Pre-activation z = w·x + b of the final model on its test rows, the target, and the model."""
    net, scaler, feature_columns = load_probability_model(directory)
    test = pd.read_csv(Path(directory) / "test_predictions.csv")
    rows = dataset.iloc[test["row"].to_numpy()]
    dense = net.layers[0]
    z = dense.forward(scaler.transform(rows[feature_columns].to_numpy(dtype=float)))[:, 0]
    return z, rows[TEACHER_TARGET].to_numpy(), net.layers[1]


def plot_target_vs_pre_activation(runs: dict[str, str], dataset: pd.DataFrame, path: Path) -> None:
    """BigModel's probability against each model's z, with the model's output curve on top."""
    figure, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 5), layout="constrained")
    for axis, (activation, directory) in zip(np.atleast_1d(axes), runs.items()):
        z, target, activation_layer = test_pre_activations(directory, dataset)
        grid = np.linspace(z.min(), z.max(), 300).reshape(-1, 1)
        axis.scatter(z, target, s=4, alpha=0.2, label="transacción de test (BigModel)")
        axis.plot(grid[:, 0], activation_layer.forward(grid)[:, 0], color="C1", label=f"salida de TinyModel ({activation})")
        axis.axhspan(0, 1, color="gray", alpha=0.08, label="rango válido [0, 1]")
        axis.set(xlabel=r"$z = \mathbf{w}^\top \mathbf{x} + b$", ylabel="probabilidad", title=activation)
        axis.legend(loc="upper left", fontsize=8)
    figure.suptitle("Cómo cada activación transforma la misma combinación lineal (test)")
    save_figure(figure, path)


def plot_predicted_vs_target(runs: dict[str, str], path: Path) -> None:
    """Test predictions against BigModel, on a shared axis so outputs above 1 are visible."""
    figure, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5.5), sharey=True, layout="constrained")
    for axis, (activation, directory) in zip(np.atleast_1d(axes), runs.items()):
        test = pd.read_csv(Path(directory) / "test_predictions.csv")
        axis.scatter(test["target_probability"], test["predicted_probability"], s=4, alpha=0.2)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="predicción perfecta")
        axis.axhline(1, color="C3", linestyle=":", label="probabilidad máxima")
        outside = int(((test["predicted_probability"] < 0) | (test["predicted_probability"] > 1)).sum())
        axis.set(xlabel=f"{TEACHER_TARGET} (BigModel)", title=f"{activation}: {outside} de {len(test)} fuera de [0, 1]")
        axis.legend(loc="upper left", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("predicción de TinyModel")
    figure.suptitle("Predicciones en test por activación")
    save_figure(figure, path)


def run_comparison(config: Config) -> pd.DataFrame:
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    runs = config["runs"]
    table = comparison_table(runs)
    table.to_csv(output / "activation_comparison.csv", index=False)
    dataset = load_fraud_dataset(config["dataset"])
    plot_target_vs_pre_activation(runs, dataset, output / "target_vs_pre_activation.png")
    plot_predicted_vs_target(runs, output / "predicted_vs_target_by_activation.png")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare output activations for Exercise 1")
    parser.add_argument("config", nargs="?", help="optional JSON config; omitted keys use defaults")
    args = parser.parse_args()
    config = deep_merge(DEFAULTS, read_json(args.config)) if args.config else DEFAULTS
    table = run_comparison(config)
    print(table.round(4).to_string(index=False))
    print(f"\nFigures written to {config['output_directory']}")


if __name__ == "__main__":
    main()
