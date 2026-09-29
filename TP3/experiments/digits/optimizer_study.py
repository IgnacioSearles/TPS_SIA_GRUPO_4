"""Controlled SGD/momentum comparison, using only the exercise 2 development data."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import config_from_cli, deep_merge
from experiments.digits.baseline import DEFAULTS, run

STUDY_DEFAULTS = deep_merge(DEFAULTS, {
    "model": {"layers": [784, 64, 32, 10]},
    "study": {"learning_rates": [0.01, 0.1], "momentum_factors": [0.5, 0.9]},
    "training": {"epochs": 500, "shuffle_seed": 42},
    "output": {"directory": "results/digits_optimizers"},
})


def configurations(config):
    """Same rate grid for SGD and classical momentum; all runs start fresh."""
    rates = config["study"]["learning_rates"]
    factors = config["study"]["momentum_factors"]
    if not rates or any(not np.isfinite(v) or v <= 0 for v in rates) or len(set(rates)) != len(rates):
        raise ValueError("learning_rates must be unique, finite and positive")
    if not factors or any(not np.isfinite(v) or not 0 < v < 1 for v in factors) or len(set(factors)) != len(factors):
        raise ValueError("momentum_factors must be unique and in (0, 1)")
    if config["training"].get("shuffle_seed") is None:
        raise ValueError("An independent shuffle_seed is required")
    runs = []
    for factor in [0.0, *factors]:
        for rate in rates:
            current = copy.deepcopy(config)
            current.pop("study")
            current["optimizer"] = {"name": "sgd" if factor == 0 else "momentum", "lr": rate}
            if factor:
                current["optimizer"]["momentum"] = factor
            current["output"]["directory"] = str(Path(config["output"]["directory"]) / f"run_{len(runs):02d}")
            runs.append(current)
    return runs


def run_study(config):
    runs = configurations(config)
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "study_config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    rows, histories = [], []
    reference = None
    for current in runs:
        summary = run(current)
        directory = Path(current["output"]["directory"])
        with np.load(directory / "split_indices.npz") as split:
            signature = (summary["dataset_sha256"], split["train"].copy(), split["validation"].copy(),
                         summary["initial_validation"])
        if reference is None:
            reference = signature
        else:
            assert signature[0] == reference[0], "Dataset changed between runs"
            np.testing.assert_array_equal(signature[1], reference[1])
            np.testing.assert_array_equal(signature[2], reference[2])
            assert signature[3] == reference[3], "Initial model evaluation differs"
        validation = summary["validation"]
        layers = current["model"]["layers"]
        rows.append({"optimizer": current["optimizer"]["name"],
                     "momentum": current["optimizer"].get("momentum", 0.0),
                     "architecture": "-".join(map(str, layers)),
                     "parameters": sum(a * b + b for a, b in zip(layers[:-1], layers[1:])),
                     "learning_rate": current["optimizer"]["lr"], "seed": current["seed"],
                     "best_epoch": summary["best_epoch"], "train_accuracy": summary["train"]["accuracy"],
                     "validation_accuracy": validation["accuracy"], "validation_loss": validation["loss"],
                     "validation_macro_f1": validation["macro_f1"],
                     "validation_recall_5": validation["per_class"][5]["recall"],
                     "run_directory": directory.name})
        histories.append(pd.read_csv(directory / "history.csv"))
        pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    winner = max(rows, key=lambda row: (row["validation_accuracy"], -row["validation_loss"]))
    result = {"selection": "highest validation accuracy, tie: lowest validation loss",
              "winner": winner, "runs": rows, "same_split_and_initial_metrics_verified": True, "shuffle_seed": config["training"]["shuffle_seed"],
              "dataset_sha256": reference[0], "external_test_evaluated": False,
              "epochs_budget": config["training"]["epochs"],
              "scope": "Single training seed and fixed epoch budget; not a global optimum"}
    (output / "study_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for row, history in zip(rows, histories):
        label = f"{row['optimizer']} η={row['learning_rate']:g}, memoria={row['momentum']:g}"
        axes[0].plot(history.epoch, history.validation_loss, label=label)
        axes[1].plot(history.epoch, history.validation_accuracy, label=label)
    for ax, title in zip(axes, ["Error cuadrático en validación", "Accuracy de validación"]):
        ax.set(xlabel="Época", title=title)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "optimizer_comparison.png", dpi=150)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return result


if __name__ == "__main__":
    run_study(config_from_cli(STUDY_DEFAULTS))
