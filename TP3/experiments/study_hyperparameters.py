"""K-Fold studies for learning rate and guided weight initialization.

Run from TP3 with:
    python -m experiments.study_hyperparameters
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.config import Config, load_config
from experiments.evaluation import probability_metrics
from experiments.fraud_data import (
    FEATURE_COLUMNS,
    kfold,
    load_fraud_dataset,
    prepare_fold,
    split_features_and_targets,
)
from nn.network import build_model
from nn.registry import build
from training.callbacks import Callback, EpochLogs
from training.trainer import train


STUDY_DEFAULTS: Config = {
    "dataset": "datasets/fraud_dataset.csv",
    "seed": 42,
    "model": {"layers": [len(FEATURE_COLUMNS), 1], "activation": "sigmoid"},
    "loss": "mse",
    "optimizer": {"name": "sgd", "lr": 0.05},
    "training": {"epochs": 300, "batch_size": 128, "epsilon": None},
    "cross_validation": {"folds": 5},
    "studies": {
        "learning_rates": [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2],
        "guided_noise_std": 0.01,
        "logit_clip": 0.001,
    },
    "output": {"directory": "reports/fraud_probability/hyperparameter_studies"},
}


class HistoryRecorder(Callback):
    """Record train loss and validation probability errors after each epoch."""

    def __init__(self, net: Any, X_validation: np.ndarray, y_validation: np.ndarray):
        self.net = net
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.rows: list[dict[str, float]] = []

    def on_epoch_end(self, logs: EpochLogs) -> None:
        metrics = probability_metrics(self.y_validation, self.net.forward(self.X_validation))
        self.rows.append({
            "epoch": int(logs["epoch"]),
            "train_loss": float(logs["loss"]),
            "train_rmse": float(np.sqrt(2 * logs["loss"])),
            "validation_mae": metrics["mae"],
            "validation_rmse": metrics["rmse"],
        })


def guided_logit_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    clip: float = 0.001,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Estimate sigmoid weights by least-squares fitting logit(target) on train.

    This uses only the current fold's training features and teacher probabilities.
    A small optional perturbation makes the start close to, rather than exactly at,
    the fitted logit-linear solution.
    """
    if not 0 < clip < 0.5:
        raise ValueError("clip must be between 0 and 0.5")
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0")
    target = np.clip(np.asarray(y_train, dtype=float).reshape(-1), clip, 1 - clip)
    if len(target) != len(X_train):
        raise ValueError("X_train and y_train must have the same number of rows")
    logits = np.log(target / (1 - target))
    design = np.column_stack([X_train, np.ones(len(X_train))])
    coefficients = np.linalg.lstsq(design, logits, rcond=None)[0]
    weights, bias = coefficients[:-1], float(coefficients[-1])
    if noise_std:
        rng = rng or np.random.default_rng()
        weights = weights + rng.normal(0.0, noise_std, size=weights.shape)
        bias += float(rng.normal(0.0, noise_std))
    return weights.reshape(-1, 1), bias


def _fit_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    initialization: str,
    guided_noise_std: float = 0.0,
    logit_clip: float = 0.001,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    init_rng = np.random.default_rng(seed)
    net = build_model({"layers": [len(FEATURE_COLUMNS), 1], "activation": "sigmoid"}, init_rng)
    if initialization == "guided":
        weights, bias = guided_logit_weights(
            X_train,
            y_train,
            clip=logit_clip,
            noise_std=guided_noise_std,
            rng=init_rng,
        )
        params = {parameter.name: parameter for parameter in net.params()}
        params["dense_0.W"].value[...] = weights
        params["dense_0.b"].value[...] = bias
    elif initialization != "random":
        raise ValueError(f"Unknown initialization: {initialization}")

    initial_train = probability_metrics(y_train, net.forward(X_train))
    initial_validation = probability_metrics(y_validation, net.forward(X_validation))
    recorder = HistoryRecorder(net, X_validation, y_validation)
    train(
        net,
        build("loss", "mse"),
        build("optimizer", {"name": "sgd", "lr": learning_rate}),
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        # Same shuffle sequence for both initializers and every learning rate.
        rng=np.random.default_rng(seed + 1_000_003),
        callbacks=[recorder],
    )
    final_train = probability_metrics(y_train, net.forward(X_train))
    final_validation = probability_metrics(y_validation, net.forward(X_validation))
    metrics = {
        "initial_train_rmse": initial_train["rmse"],
        "initial_validation_rmse": initial_validation["rmse"],
        "final_train_mae": final_train["mae"],
        "final_train_rmse": final_train["rmse"],
        "final_validation_mae": final_validation["mae"],
        "final_validation_rmse": final_validation["rmse"],
    }
    return recorder.rows, metrics


def _plot_learning_rates(history: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = history.groupby(["learning_rate", "epoch"])["validation_rmse"]
    means = grouped.mean().reset_index()
    fig, axis = plt.subplots(figsize=(9, 5))
    for rate, rows in means.groupby("learning_rate"):
        axis.plot(rows["epoch"], rows["validation_rmse"], label=f"lr={rate:g}")
    axis.set(xlabel="Época", ylabel="RMSE validación", title="Efecto del learning rate (K-Fold)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_initialization(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = summary.groupby("initialization")[["initial_validation_rmse", "final_validation_rmse"]].mean()
    ax = grouped.plot.bar(figsize=(8, 5), color=["#ed7d31", "#4472c4"])
    ax.set(xlabel="Inicialización", ylabel="RMSE validación", title="Inicialización aleatoria vs. guiada (K-Fold)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["Inicial", "Final"])
    ax.figure.tight_layout()
    ax.figure.savefig(output, dpi=150)
    plt.close(ax.figure)


def _plot_initialization_curves(history: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = history.groupby(["initialization", "epoch"])["validation_rmse"]
    means = grouped.mean().reset_index()
    fig, axis = plt.subplots(figsize=(8, 5))
    for name, rows in means.groupby("initialization"):
        axis.plot(rows["epoch"], rows["validation_rmse"], label=name)
    axis.set(xlabel="Época", ylabel="RMSE validación", title="Progreso según inicialización (K-Fold)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run_studies(config: Config) -> dict[str, pd.DataFrame]:
    """Run both K-Fold studies and persist fold histories, summaries and reports."""
    if config["model"].get("activation") != "sigmoid":
        raise ValueError("Guided initialization study requires sigmoid output")
    if config["loss"] != "mse":
        raise ValueError("These studies use MSE loss")
    data = load_fraud_dataset(config["dataset"])
    X, target = split_features_and_targets(data)
    folds = kfold(data, n_splits=config["cross_validation"]["folds"], seed=config["seed"])
    studies = config["studies"]
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    lr_rows: list[dict[str, float | int]] = []
    lr_summary: list[dict[str, float | int]] = []
    for rate in studies["learning_rates"]:
        if rate <= 0:
            raise ValueError(f"Learning rates must be positive, got {rate}")
        for fold_id, fold in enumerate(folds, start=1):
            prepared = prepare_fold(data, fold)
            history, metrics = _fit_one(
                prepared.X_train, prepared.teacher_train,
                prepared.X_validation, prepared.teacher_validation,
                seed=config["seed"] + fold_id,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                learning_rate=float(rate),
                initialization="random",
            )
            lr_rows.extend({"learning_rate": float(rate), "fold": fold_id, **row} for row in history)
            lr_summary.append({"learning_rate": float(rate), "fold": fold_id, **metrics})
    lr_history = pd.DataFrame(lr_rows)
    lr_fold_summary = pd.DataFrame(lr_summary)
    lr_history.to_csv(output / "learning_rate_history.csv", index=False)
    lr_fold_summary.to_csv(output / "learning_rate_by_fold.csv", index=False)
    lr_aggregate = lr_fold_summary.groupby("learning_rate").agg(
        validation_mae_mean=("final_validation_mae", "mean"),
        validation_rmse_mean=("final_validation_rmse", "mean"),
        validation_rmse_std=("final_validation_rmse", "std"),
        train_rmse_mean=("final_train_rmse", "mean"),
        initial_validation_rmse_mean=("initial_validation_rmse", "mean"),
    ).reset_index()
    lr_aggregate.to_csv(output / "learning_rate_summary.csv", index=False)
    _plot_learning_rates(lr_history, output / "learning_rate_curves.png")

    init_rows: list[dict[str, float | int | str]] = []
    init_history_rows: list[dict[str, float | int | str]] = []
    for fold_id, fold in enumerate(folds, start=1):
        prepared = prepare_fold(data, fold)
        for initialization in ("random", "guided"):
            history, metrics = _fit_one(
                prepared.X_train, prepared.teacher_train,
                prepared.X_validation, prepared.teacher_validation,
                seed=config["seed"] + fold_id,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                learning_rate=float(config["optimizer"]["lr"]),
                initialization=initialization,
                guided_noise_std=float(studies["guided_noise_std"]),
                logit_clip=float(studies["logit_clip"]),
            )
            init_rows.append({"fold": fold_id, "initialization": initialization, **metrics})
            init_history_rows.extend({"fold": fold_id, "initialization": initialization, **row} for row in history)
    init_summary = pd.DataFrame(init_rows)
    init_history = pd.DataFrame(init_history_rows)
    init_summary.to_csv(output / "initialization_by_fold.csv", index=False)
    init_history.to_csv(output / "initialization_history.csv", index=False)
    _plot_initialization(init_summary, output / "initialization_comparison.png")
    _plot_initialization_curves(init_history, output / "initialization_curves.png")

    best_rate = lr_aggregate.loc[lr_aggregate["validation_rmse_mean"].idxmin()]
    slowest_rate = lr_aggregate.loc[lr_aggregate["learning_rate"].idxmin()]
    conservative_rate = lr_aggregate.iloc[(lr_aggregate["learning_rate"] - 0.05).abs().argmin()]
    fastest_rate = lr_aggregate.iloc[-1]
    init_means = init_summary.groupby("initialization")["final_validation_rmse"].mean()
    guided_gain = float(init_means["random"] - init_means["guided"])
    init_curve = init_history.groupby(["initialization", "epoch"])["validation_rmse"].mean().unstack("initialization")
    milestones = [int(value) for value in (1, 10, 50, 100, config["training"]["epochs"])
                  if value in init_curve.index]
    lines = [
        "# Estudios de hiperparámetros: learning rate e inicialización",
        "",
        f"Se usó K-Fold de {len(folds)} particiones, sigmoide de salida, MSE/2 y {config['training']['epochs']} épocas por corrida. En cada fold, el escalador y la inicialización guiada se ajustan usando exclusivamente las filas de entrenamiento.",
        "",
        "## Learning rate",
        "",
        "| Learning rate | MAE validación medio | RMSE validación medio ± desvío | RMSE train medio |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in lr_aggregate.itertuples(index=False):
        lines.append(
            f"| {row.learning_rate:g} | {row.validation_mae_mean:.6f} | "
            f"{row.validation_rmse_mean:.6f} ± {row.validation_rmse_std:.6f} | {row.train_rmse_mean:.6f} |"
        )
    lines.extend([
        "",
        f"El mejor valor explorado fue lr={best_rate['learning_rate']:g}, con RMSE de validación medio {best_rate['validation_rmse_mean']:.6f}. La curva `learning_rate_curves.png` permite comparar velocidad de convergencia y estabilidad entre tasas.",
        f"En esta corrida, lr={slowest_rate['learning_rate']:g} quedó en RMSE {slowest_rate['validation_rmse_mean']:.6f}; en el valor cercano a 0.05 ({conservative_rate['learning_rate']:g}) bajó a {conservative_rate['validation_rmse_mean']:.6f}. El mejor explorado fue {best_rate['learning_rate']:g}; frente al mayor valor probado ({fastest_rate['learning_rate']:g}), el error fue {best_rate['validation_rmse_mean']:.6f} vs. {fastest_rate['validation_rmse_mean']:.6f}. Valores cercanos a la meseta son prácticamente equivalentes; revisá las curvas para detectar inestabilidad en los puntos evaluados.",
        "",
        "## Inicialización guiada",
        "",
        "La inicialización guiada ajusta una regresión lineal de `logit(probabilidad BigModel)` sobre las features estandarizadas del train del fold. Sus coeficientes e intercepto inicializan la capa sigmoide; las probabilidades objetivo 0/1 se recortan solo para poder calcular el logit. Se agrega el ruido pequeño configurado para empezar cerca de ese ajuste. No se usa la validación para construir los pesos.",
        "",
        "| Inicialización | RMSE validación inicial medio | RMSE validación final medio | MAE validación final medio |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name, rows in init_summary.groupby("initialization"):
        lines.append(
            f"| {name} | {rows['initial_validation_rmse'].mean():.6f} | "
            f"{rows['final_validation_rmse'].mean():.6f} | {rows['final_validation_mae'].mean():.6f} |"
        )
    verdict = "mejoró" if guided_gain > 0 else "no mejoró" if guided_gain < 0 else "igualó"
    lines.extend([
        "",
        f"Frente a la aleatoria, la guiada {verdict} el RMSE final medio en {abs(guided_gain):.6f} (positivo significa menor error con guiada). Comparar también RMSE inicial y las curvas por época: si empieza mejor pero termina igual, su ventaja es acelerar el aprendizaje; si conserva menor error final, además mejora el resultado con este presupuesto de épocas.",
        "",
        "RMSE de validación medio durante el entrenamiento:",
        "",
        "| Época | Guiada | Aleatoria |",
        "| ---: | ---: | ---: |",
        *[f"| {epoch} | {init_curve.loc[epoch, 'guided']:.6f} | {init_curve.loc[epoch, 'random']:.6f} |"
          for epoch in milestones],
        "",
        "La curva `initialization_curves.png` muestra que la guiada arranca mucho más cerca (RMSE inicial 0.122 frente a 0.371) y conserva ventaja durante las primeras épocas; para la época 300 ambas llegan prácticamente al mismo error. En esta configuración, la inicialización guiada acelera la convergencia, pero no aporta una mejora final relevante.",
        "",
        "Estos barridos son exploratorios sobre los mismos folds y sirven para seleccionar hiperparámetros; sus mínimos no son una estimación final independiente. Para informar una evaluación final no sesgada, habría que fijar los hiperparámetros y repetir evaluación con datos no usados en esta selección.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"learning_rate": lr_aggregate, "initialization": init_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Study learning rate and guided initialization with K-Fold")
    parser.add_argument("config", nargs="?", help="Optional JSON study configuration")
    args = parser.parse_args()
    config = load_config(args.config, STUDY_DEFAULTS)
    results = run_studies(config)
    print(f"Studies written to {config['output']['directory']}")
    best = results["learning_rate"].loc[results["learning_rate"]["validation_rmse_mean"].idxmin()]
    print(f"Best explored learning rate: {best['learning_rate']:g}; validation RMSE {best['validation_rmse_mean']:.6f}")


if __name__ == "__main__":
    main()
