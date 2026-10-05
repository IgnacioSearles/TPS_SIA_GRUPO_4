"""L2 regularization (weight decay) sweep on the exercise 2 network, digits.csv only.

Only `optimizer.weight_decay` changes. Everything else is the exercise 2 protocol:
784-64-32-10, tanh, Xavier, sigmoid output, momentum 0.9, lr 0.1, batch 128,
best validation epoch. λ = 0 is trained again under the same protocol so every
row is paired by seed.

Run from TP3:
    python -m experiments.digits.l2_study [config.json]
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import config_from_cli, deep_merge
from experiments.digits.baseline import DEFAULTS, run

STUDY_DEFAULTS = deep_merge(DEFAULTS, {
    "model": {"layers": [784, 64, 32, 10]},
    "optimizer": {"name": "momentum", "lr": 0.1, "momentum": 0.9},
    "training": {"epochs": 300},
    "study": {"weight_decays": [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3], "seeds": [42, 7, 21]},
    "output": {"directory": "results/digits_l2"},
})


def squared_weight_norm(run_directory: Path) -> float:
    """Sum of squared weights (biases excluded) of the saved best model."""
    with np.load(run_directory / "best_weights.npz") as saved:
        return float(sum(np.sum(saved[name] ** 2) for name in saved.files if name.endswith(".W")))


def run_study(config: dict) -> pd.DataFrame:
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "study_config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    for weight_decay in config["study"]["weight_decays"]:
        for seed in config["study"]["seeds"]:
            current = copy.deepcopy(config)
            current.pop("study")
            current["seed"] = seed
            current["training"]["shuffle_seed"] = seed
            current["optimizer"]["weight_decay"] = weight_decay
            directory = output / f"wd_{weight_decay:g}_seed_{seed}"
            current["output"]["directory"] = str(directory)
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else run(current)
            rows.append({
                "weight_decay": weight_decay, "seed": seed, "best_epoch": summary["best_epoch"],
                "train_accuracy": summary["train"]["accuracy"],
                "validation_accuracy": summary["validation"]["accuracy"],
                "validation_loss": summary["validation"]["loss"],
                "validation_macro_f1": summary["validation"]["macro_f1"],
                "squared_weight_norm": squared_weight_norm(directory),
            })
            pd.DataFrame(rows).to_csv(output / "runs.csv", index=False)

    table = pd.DataFrame(rows)
    table["generalization_gap"] = table["train_accuracy"] - table["validation_accuracy"]
    table.to_csv(output / "runs.csv", index=False)
    aggregate = table.groupby("weight_decay").agg(
        seeds=("seed", "nunique"),
        train_accuracy_mean=("train_accuracy", "mean"),
        validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
        generalization_gap_mean=("generalization_gap", "mean"),
        squared_weight_norm_mean=("squared_weight_norm", "mean"),
        median_best_epoch=("best_epoch", "median"),
    ).reset_index()
    aggregate.to_csv(output / "aggregate.csv", index=False)
    _plot(output, aggregate, config)
    print(aggregate.to_string(index=False), flush=True)
    return aggregate


def _plot(output: Path, aggregate: pd.DataFrame, config: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{value:g}" for value in aggregate["weight_decay"]]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(positions, 100 * aggregate["train_accuracy_mean"], "o-", label="Entrenamiento")
    axes[0].errorbar(positions, 100 * aggregate["validation_accuracy_mean"],
                     yerr=100 * aggregate["validation_accuracy_std"].fillna(0), fmt="s-", capsize=4, label="Validación")
    axes[0].set(title="Accuracy (mejor época de validación)", ylabel="%")
    axes[0].legend()
    axes[1].plot(positions, 100 * aggregate["generalization_gap_mean"], "o-", color="C3")
    axes[1].set(title="Brecha entrenamiento − validación", ylabel="puntos porcentuales")
    axes[2].plot(positions, aggregate["squared_weight_norm_mean"], "o-", color="C2")
    axes[2].set(title="Norma de los pesos  Σ w²", yscale="log")
    for ax in axes:
        ax.set_xticks(positions, labels)
        ax.set_xlabel("λ (weight decay)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "l2_comparison.png", dpi=150)
    plt.close(fig)

    # Mean ± 1 standard deviation across seeds, for a readable subset of λ values.
    seeds = config["study"]["seeds"]
    shown = config["study"].get("curve_weight_decays", list(aggregate["weight_decay"]))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for index, weight_decay in enumerate(shown):
        curves = [pd.read_csv(output / f"wd_{weight_decay:g}_seed_{seed}" / "history.csv")["validation_accuracy"]
                  for seed in seeds]
        values = 100 * pd.concat(curves, axis=1)
        epochs = values.index + 1
        mean, std = values.mean(axis=1), values.std(axis=1)
        ax.plot(epochs, mean, color=f"C{index}", label=f"λ = {weight_decay:g}")
        ax.fill_between(epochs, mean - std, mean + std, color=f"C{index}", alpha=0.2)
    ax.set(xlabel="Época", ylabel="Accuracy de validación (%)",
           title=f"Curvas de validación por λ (media de {len(seeds)} semillas ± 1 desvío)", ylim=(90, 97.5))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "l2_validation_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_study(config_from_cli(STUDY_DEFAULTS))
