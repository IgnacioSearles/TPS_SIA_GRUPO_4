"""Paired five-seed comparison of hidden-layer activations, ejercicio 2 data only.

Only `model.activation` changes. Everything else is the winning protocol of the
optimizer seed study (784-64-32-10, Xavier, sigmoid output, momentum 0.9, lr 0.1,
500 epochs, best validation epoch), so the tanh row can be read from
`results/digits_seeds` instead of being trained again.

Run from TP3:
    python -m experiments.digits.activation_study [config.json]
"""

import copy
import json
from pathlib import Path

import pandas as pd

from experiments.config import config_from_cli, deep_merge
from experiments.digits.baseline import DEFAULTS, run

STUDY_DEFAULTS = deep_merge(DEFAULTS, {
    "model": {"layers": [784, 64, 32, 10]},
    "optimizer": {"name": "momentum", "lr": 0.1, "momentum": 0.9},
    "training": {"epochs": 500},
    "study": {"seeds": [42, 7, 21, 84, 123], "activations": ["sigmoid", "relu"],
              "tanh_reference": "results/digits_seeds"},
    "output": {"directory": "results/digits_activations"},
})


def run_study(config):
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "study_config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    for activation in config["study"]["activations"]:
        for seed in config["study"]["seeds"]:
            current = copy.deepcopy(config)
            current.pop("study")
            current["seed"] = seed
            current["training"]["shuffle_seed"] = seed
            current["model"]["activation"] = activation
            directory = output / f"{activation}_seed_{seed}"
            current["output"]["directory"] = str(directory)
            if (directory / "summary.json").exists():
                summary = json.loads((directory / "summary.json").read_text())
            else:
                summary = run(current)
            metric = summary["validation"]
            rows.append({"activation": activation, "seed": seed, "best_epoch": summary["best_epoch"],
                         "train_accuracy": summary["train"]["accuracy"],
                         "validation_accuracy": metric["accuracy"], "validation_loss": metric["loss"],
                         "validation_macro_f1": metric["macro_f1"],
                         "validation_recall_5": metric["per_class"][5]["recall"]})
            pd.DataFrame(rows).to_csv(output / "runs.csv", index=False)

    reference = pd.read_csv(Path(config["study"]["tanh_reference"]) / "runs.csv")
    tanh = reference[reference.candidate == "momentum_lr0.1_m0.9"]
    for row in tanh.itertuples():
        rows.append({"activation": "tanh", "seed": row.seed, "best_epoch": row.best_epoch,
                     "train_accuracy": row.train_accuracy, "validation_accuracy": row.validation_accuracy,
                     "validation_loss": row.validation_loss, "validation_macro_f1": row.validation_macro_f1,
                     "validation_recall_5": row.validation_recall_5})
    table = pd.DataFrame(rows)
    table.to_csv(output / "runs.csv", index=False)
    grouped = table.groupby("activation").agg(
        seeds=("seed", "nunique"), median_best_epoch=("best_epoch", "median"),
        train_accuracy_mean=("train_accuracy", "mean"),
        validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
        validation_loss_mean=("validation_loss", "mean"),
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
    ).reset_index().sort_values("validation_accuracy_mean", ascending=False)
    grouped.to_csv(output / "aggregate.csv", index=False)
    _plot(output, config)
    print(grouped.to_string(index=False), flush=True)
    return grouped


def _plot(output, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reference = Path(config["study"]["tanh_reference"])
    seed = config["study"]["seeds"][0]
    curves = {activation: output / f"{activation}_seed_{seed}" / "history.csv"
              for activation in config["study"]["activations"]}
    runs = pd.read_csv(reference / "runs.csv")
    tanh_dir = runs[(runs.candidate == "momentum_lr0.1_m0.9") & (runs.seed == seed)].run_directory.iloc[0]
    curves["tanh"] = reference / tanh_dir / "history.csv"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for activation, path in curves.items():
        history = pd.read_csv(path)
        ax.plot(history.epoch, history.validation_accuracy, label=activation)
    ax.set(xlabel="Época", ylabel="Accuracy de validación",
           title=f"Activación de las capas ocultas (semilla {seed})", ylim=(0.85, 0.98))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "activation_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_study(config_from_cli(STUDY_DEFAULTS))
