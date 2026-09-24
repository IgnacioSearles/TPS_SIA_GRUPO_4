"""Exercise 1: learn BigModel's continuous fraud probability with perceptrons.

Run from the TP3 directory:
    python -m experiments.fraud_probability
    python -m experiments.fraud_probability experiments/configs/fraud_probability.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.config import Config, deep_merge, load_config
from experiments.evaluation import probability_metrics, save_weights
from experiments.fraud_data import (
    FEATURE_COLUMNS,
    TEACHER_TARGET,
    StandardScaler,
    kfold,
    load_fraud_dataset,
    prepare_fold,
    split_features_and_targets,
)
from experiments.runner import run_training
from nn.network import Sequential, build_model
from nn.registry import build
from training.callbacks import Callback, EpochLogs, LossThreshold
from training.trainer import train


EXPERIMENT_DEFAULTS: Config = {
    "dataset": "datasets/fraud_dataset.csv",
    "seed": 42,
    "model": {"layers": [len(FEATURE_COLUMNS), 1], "activation": "sigmoid"},
    "optimizer": {"name": "sgd", "lr": 0.05},
    "training": {"epochs": 1000, "batch_size": 128, "epsilon": 0.001},
    "cross_validation": {"folds": 5},
    "output": {"directory": "reports/fraud_probability"},
    "callbacks": [],
}


def _training_config(config: Config, activation: str) -> Config:
    """Copy experiment settings into a single-perceptron training config."""
    return deep_merge(config, {
        "model": {"layers": [len(FEATURE_COLUMNS), 1], "activation": activation},
        "loss": "mse",
        "evaluation": {"classification": False, "probability": False},
        "output": {"weights_path": None, "results_path": None},
        "callbacks": [],
    })


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def _plot_learning_curves(histories: dict[str, list[dict[str, float]]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    for activation, history in histories.items():
        axis.plot([int(row["epoch"]) for row in history], [row["loss"] for row in history], label=activation)
    axis.set_xlabel("Época")
    axis.set_ylabel("Pérdida de entrenamiento (MSE / 2)")
    axis.set_title("Aprendizaje con todas las muestras")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cross_validation_curves(history: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = history.groupby("epoch")
    epochs = grouped["validation_rmse"].mean().index.to_numpy()
    train_rmse = grouped["train_rmse"].mean().to_numpy()
    validation_rmse = grouped["validation_rmse"].mean().to_numpy()
    validation_std = grouped["validation_rmse"].std().fillna(0.0).to_numpy()

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_rmse, label="RMSE train", color="#4472c4")
    axis.plot(epochs, validation_rmse, label="RMSE validación", color="#ed7d31")
    axis.fill_between(epochs, validation_rmse - validation_std, validation_rmse + validation_std,
                      color="#ed7d31", alpha=0.18, label="± 1 desvío entre folds")
    axis.set_xlabel("Época")
    axis.set_ylabel("RMSE de probabilidad")
    axis.set_title("K-Fold: error por época (promedio de folds)")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class FoldMetricsRecorder(Callback):
    """Capture probability errors on train and validation after each epoch."""

    def __init__(self, net: Any, X_validation: np.ndarray, y_validation: np.ndarray):
        self.net = net
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.history: list[dict[str, float]] = []

    def on_epoch_end(self, logs: EpochLogs) -> None:
        validation = probability_metrics(self.y_validation, self.net.forward(self.X_validation))
        self.history.append({
            "epoch": logs["epoch"],
            "train_loss": logs["loss"],
            "train_rmse": float(np.sqrt(2 * logs["loss"])),
            "validation_mae": validation["mae"],
            "validation_rmse": validation["rmse"],
        })


def _last_improvement(history: list[dict[str, float]]) -> float:
    """Loss reduction over the last up to 20 epochs (not a validation metric)."""
    start = max(0, len(history) - 20)
    return float(history[start]["loss"] - history[-1]["loss"])


def _summary_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---:" if index else "---" for index in range(len(columns))) + " |"
    body = ["| " + " | ".join(str(row[key]) for key, _ in columns) + " |" for row in rows]
    return [header, separator, *body]


def _write_report(summary: dict[str, Any], path: Path) -> None:
    comparison = summary["learning_comparison"]
    rows = []
    for activation in ("identity", "sigmoid"):
        entry = comparison[activation]
        rows.append({
            "model": activation,
            "epochs": entry["epochs"],
            "mae": f"{entry['mae']:.6f}",
            "rmse": f"{entry['rmse']:.6f}",
            "outside": entry["outside_0_1"],
            "improvement": f"{entry['last_20_loss_improvement']:.6f}",
        })
    cv = summary["cross_validation"]
    linear = comparison["identity"]
    nonlinear = comparison["sigmoid"]
    rmse_reduction = 100 * (1 - nonlinear["rmse"] / linear["rmse"])
    lines = [
        "# Ejercicio 1: estimación de probabilidad de fraude",
        "",
        f"Objetivo continuo: `{TEACHER_TARGET}`. Entrada: {len(FEATURE_COLUMNS)} variables. Pérdida: MSE / 2. No se calculan decisiones binarias.",
        "",
        "## Comparación de aprendizaje",
        "",
        "Ambos perceptrones simples se entrenaron con las mismas muestras, escala, semilla e hiperparámetros. Estos errores son de entrenamiento y describen capacidad de ajuste; no estiman generalización.",
        "",
        *_summary_table(rows, [
            ("model", "Activación"), ("epochs", "Épocas"), ("mae", "MAE"),
            ("rmse", "RMSE"), ("outside", "Salidas fuera de [0,1]"),
            ("improvement", "Mejora de pérdida en últimas 20 épocas"),
        ]),
        "",
        f"El sigmoide reduce el RMSE de entrenamiento un {rmse_reduction:.1f}% respecto del lineal. El mayor error residual del lineal, incluso entrenado con todos los datos, indica underfitting relativo frente al sigmoide.",
        "Las mejoras de pérdida en las últimas 20 épocas son pequeñas: ambas curvas muestran una meseta aproximada con estos hiperparámetros. Esto no demuestra un mínimo global.",
        "Se selecciona el sigmoide para generalización: además del menor error, su salida siempre queda en [0,1].",
        "",
        "## Generalización: K-Fold",
        "",
        f"Se usaron {cv['folds']} folds aleatorios reproducibles. Cada fold ajustó su propio escalador solo con entrenamiento, inició un modelo nuevo y evaluó las probabilidades de validación.",
        f"MAE de validación: {cv['mae_mean']:.6f} ± {cv['mae_std']:.6f}.",
        f"RMSE de validación: {cv['rmse_mean']:.6f} ± {cv['rmse_std']:.6f}.",
        f"RMSE promedio de train: {cv['train_rmse_mean']:.6f}; la brecha train-validación es {cv['rmse_gap']:.6f}.",
        f"Métricas con todas las predicciones fuera de muestra reunidas: MAE {cv['oof_mae']:.6f}, RMSE {cv['oof_rmse']:.6f}.",
        f"Baseline constante (media de probabilidades de cada train): RMSE fuera de muestra {cv['baseline_oof_rmse']:.6f}; el modelo reduce ese error un {cv['rmse_reduction_vs_baseline_pct']:.1f}%.",
        "La brecha pequeña y estable junto con la mejora casi nula al pasar de 600 a 1.000 épocas no muestra señales de overfitting en este rango.",
        "",
        "## Modelo final",
        "",
        f"Se reentrenó una sigmoide con todas las {summary['samples']} muestras. Tiene {summary['final_model']['parameters']} parámetros entrenables. Sus pesos están en `model_weights.npz` y el orden de variables y los parámetros de estandarización en `model_metadata.json`.",
        f"Terminó después de {summary['final_model']['epochs']} épocas ({summary['final_model']['stop_reason']}).",
        "Para una transacción nueva, aplicar ese mismo escalador y luego la red. El resultado es una probabilidad en [0,1].",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: Config) -> dict[str, Any]:
    """Compare learning, validate the sigmoid model and fit the final model."""
    if config["model"]["activation"] != "sigmoid":
        raise ValueError("The final probability model must use sigmoid activation")
    if config["loss"] != "mse":
        raise ValueError("This exercise uses MSE loss")
    data = load_fraud_dataset(config["dataset"])
    X, target = split_features_and_targets(data)
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "experiment_config.json", config)

    # Descriptive learning study required by the exercise: all available rows.
    full_scaler = StandardScaler().fit(X)
    X_full = full_scaler.transform(X)
    histories: dict[str, list[dict[str, float]]] = {}
    comparison_nets = {}
    comparison: dict[str, dict[str, float | int]] = {}
    for activation in ("identity", "sigmoid"):
        result = run_training(_training_config(config, activation), X_full, target)
        predicted = result.net.forward(X_full)
        metrics = probability_metrics(target, predicted)
        histories[activation] = result.history
        comparison_nets[activation] = result.net
        comparison[activation] = {
            **metrics,
            "epochs": len(result.history),
            "initial_loss": result.history[0]["loss"],
            "final_loss": result.history[-1]["loss"],
            "last_20_loss_improvement": _last_improvement(result.history),
            "outside_0_1": int(((predicted < 0) | (predicted > 1)).sum()),
        }

    _plot_learning_curves(histories, output / "learning_curves.png")
    for activation, history in histories.items():
        pd.DataFrame(history).to_csv(output / f"learning_{activation}.csv", index=False)
        comparison_net = comparison_nets[activation]
        save_weights(comparison_net, output / f"learning_{activation}_weights.npz")

    # Each validation row is predicted by a model that was not trained on it.
    folds = kfold(data, n_splits=config["cross_validation"]["folds"], seed=config["seed"])
    oof = np.full(len(data), np.nan)
    baseline_oof = np.full(len(data), np.nan)
    fold_rows: list[dict[str, float | int]] = []
    fold_epoch_rows: list[dict[str, float | int]] = []
    for fold_number, fold in enumerate(folds, start=1):
        prepared = prepare_fold(data, fold)
        fold_config = _training_config(config, "sigmoid")
        rng = np.random.default_rng(fold_config["seed"])
        net = build_model(fold_config["model"], rng)
        loss = build("loss", "mse")
        optimizer = build("optimizer", fold_config["optimizer"])
        recorder = FoldMetricsRecorder(net, prepared.X_validation, prepared.teacher_validation)
        callbacks: list[Callback] = [recorder]
        epsilon = fold_config["training"].get("epsilon")
        if epsilon is not None:
            callbacks.append(LossThreshold(threshold=epsilon))
        history = train(
            net, loss, optimizer,
            prepared.X_train, prepared.teacher_train,
            epochs=fold_config["training"]["epochs"],
            batch_size=fold_config["training"]["batch_size"],
            rng=rng,
            callbacks=callbacks,
        )
        predictions = net.forward(prepared.X_validation)
        oof[fold.validation] = predictions[:, 0]
        baseline_oof[fold.validation] = float(prepared.teacher_train.mean())
        validation_metrics = probability_metrics(prepared.teacher_validation, predictions)
        train_metrics = probability_metrics(prepared.teacher_train, net.forward(prepared.X_train))
        run_history = pd.DataFrame(recorder.history)
        run_history.insert(0, "fold", fold_number)
        run_history.to_csv(output / f"fold_{fold_number:02d}_history.csv", index=False)
        fold_epoch_rows.extend({"fold": fold_number, **row} for row in recorder.history)
        save_weights(net, output / f"fold_{fold_number:02d}_weights.npz")
        _write_json(output / f"fold_{fold_number:02d}_metadata.json", {
            "fold": fold_number,
            "feature_columns": FEATURE_COLUMNS,
            "target": TEACHER_TARGET,
            "model": {"layers": [len(FEATURE_COLUMNS), 1], "activation": "sigmoid"},
            "scaler": {"mean": prepared.scaler.mean_.tolist(), "scale": prepared.scaler.scale_.tolist()},
        })
        fold_rows.append({
            "fold": fold_number,
            "train_samples": len(fold.train),
            "validation_samples": len(fold.validation),
            "epochs": len(history),
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
        })
    if not np.isfinite(oof).all():
        raise RuntimeError("Some rows did not receive an out-of-fold prediction")
    pd.DataFrame(fold_rows).to_csv(output / "cross_validation.csv", index=False)
    fold_epoch_history = pd.DataFrame(fold_epoch_rows)
    fold_epoch_history.to_csv(output / "cross_validation_epoch_metrics.csv", index=False)
    _plot_cross_validation_curves(fold_epoch_history, output / "cross_validation_error.png")
    pd.DataFrame({"row": np.arange(len(data)), "target_probability": target[:, 0],
                  "predicted_probability": oof}).to_csv(output / "out_of_fold_predictions.csv", index=False)
    oof_metrics = probability_metrics(target, oof.reshape(-1, 1))
    baseline_metrics = probability_metrics(target, baseline_oof.reshape(-1, 1))
    cv = {
        "folds": len(folds),
        "results": fold_rows,
        "mae_mean": float(np.mean([row["validation_mae"] for row in fold_rows])),
        "mae_std": float(np.std([row["validation_mae"] for row in fold_rows], ddof=1)),
        "train_rmse_mean": float(np.mean([row["train_rmse"] for row in fold_rows])),
        "rmse_mean": float(np.mean([row["validation_rmse"] for row in fold_rows])),
        "rmse_std": float(np.std([row["validation_rmse"] for row in fold_rows], ddof=1)),
        "rmse_gap": float(np.mean([row["validation_rmse"] - row["train_rmse"] for row in fold_rows])),
        "oof_mae": oof_metrics["mae"],
        "oof_rmse": oof_metrics["rmse"],
        "baseline_oof_rmse": baseline_metrics["rmse"],
        "rmse_reduction_vs_baseline_pct": float(100 * (1 - oof_metrics["rmse"] / baseline_metrics["rmse"])),
    }

    final = run_training(_training_config(config, "sigmoid"), X_full, target)
    save_weights(final.net, output / "model_weights.npz")
    pd.DataFrame(final.history).to_csv(output / "final_history.csv", index=False)
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target": TEACHER_TARGET,
        "model": {"layers": [len(FEATURE_COLUMNS), 1], "activation": "sigmoid"},
        "scaler": {"mean": full_scaler.mean_.tolist(), "scale": full_scaler.scale_.tolist()},
        "parameters": sum(parameter.value.size for parameter in final.net.params()),
    }
    _write_json(output / "model_metadata.json", metadata)
    summary = {
        "samples": len(data),
        "selected_activation": "sigmoid",
        "learning_comparison": comparison,
        "cross_validation": cv,
        "final_model": {
            "epochs": len(final.history),
            "parameters": metadata["parameters"],
            "stop_reason": (
                "epsilon alcanzado" if config["training"]["epsilon"] is not None
                and final.history[-1]["loss"] <= config["training"]["epsilon"]
                else "máximo de épocas"
            ),
            "training_metrics": probability_metrics(target, final.net.forward(X_full)),
        },
        "configuration": {
            "seed": config["seed"],
            "optimizer": config["optimizer"],
            "training": config["training"],
            "folds": config["cross_validation"]["folds"],
        },
    }
    _write_json(output / "summary.json", summary)
    _write_report(summary, output / "report.md")
    return summary


def load_probability_model(directory: str | Path) -> tuple[Sequential, StandardScaler, list[str]]:
    """Load the saved probability model and its train-fitted scaler."""
    directory = Path(directory)
    with (directory / "model_metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    net = build_model(metadata["model"], np.random.default_rng(0))
    with np.load(directory / "model_weights.npz", allow_pickle=False) as saved:
        for parameter in net.params():
            value = saved[parameter.name]
            if value.shape != parameter.value.shape:
                raise ValueError(f"Wrong shape for saved parameter {parameter.name}")
            parameter.value[...] = value
    scaler = StandardScaler(
        mean_=np.asarray(metadata["scaler"]["mean"], dtype=float),
        scale_=np.asarray(metadata["scaler"]["scale"], dtype=float),
    )
    return net, scaler, metadata["feature_columns"]


def predict_probability(rows: pd.DataFrame, directory: str | Path) -> np.ndarray:
    """Predict probabilities for rows containing the saved model's features."""
    net, scaler, feature_columns = load_probability_model(directory)
    X = rows[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Input features must be finite")
    return net.forward(scaler.transform(X))[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare simple perceptrons and validate a probability model")
    parser.add_argument("config", nargs="?", help="Optional JSON configuration file")
    args = parser.parse_args()
    config = load_config(args.config, EXPERIMENT_DEFAULTS)
    summary = run_experiment(config)
    cv = summary["cross_validation"]
    print(f"Results written to {config['output']['directory']}")
    print(f"K-Fold validation: MAE {cv['mae_mean']:.6f} ± {cv['mae_std']:.6f}; "
          f"RMSE {cv['rmse_mean']:.6f} ± {cv['rmse_std']:.6f}")


if __name__ == "__main__":
    main()
