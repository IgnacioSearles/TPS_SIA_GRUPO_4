"""Controlled architecture comparison, using only the exercise 2 development data."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import config_from_cli, deep_merge
from experiments.digits.baseline import DEFAULTS, run

STUDY_DEFAULTS = deep_merge(DEFAULTS, {
    "study": {"architectures": [[784, 32, 10], [784, 64, 10], [784, 64, 32, 10]]},
    "training": {"epochs": 500, "shuffle_seed": 42},
    "output": {"directory": "results/digits_architectures"},
})


def configurations(config):
    """Vary layer widths only; use an independent, common batch-order seed."""
    architectures = config["study"]["architectures"]
    if not architectures:
        raise ValueError("architectures must not be empty")
    for layers in architectures:
        if (len(layers) < 3 or layers[0] != 784 or layers[-1] != 10
                or any(type(n) is not int or n <= 0 for n in layers)):
            raise ValueError("Expected positive integer widths, 784 inputs, hidden layers and 10 outputs")
    if len({tuple(layers) for layers in architectures}) != len(architectures):
        raise ValueError("architectures must be unique")
    if config["training"].get("shuffle_seed") is None:
        raise ValueError("An independent shuffle_seed is required for architecture comparisons")
    seeds = config["study"].get("seeds", [config["seed"]])
    if not seeds or any(type(seed) is not int for seed in seeds):
        raise ValueError("study.seeds must be a non-empty list of integer seeds")
    runs = []
    for index, (layers, seed) in enumerate(
            ((layers, seed) for layers in architectures for seed in seeds)):
        current = copy.deepcopy(config)
        current.pop("study")
        current["model"]["layers"] = layers.copy()
        current["seed"] = seed
        current["output"]["directory"] = str(
            Path(config["output"]["directory"]) / f"run_{index:02d}_seed_{seed}")
        runs.append(current)
    return runs


def run_study(config):
    runs = configurations(config)
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    study_config_path = output / "study_config.json"
    if study_config_path.exists() and json.loads(study_config_path.read_text()) != config:
        raise ValueError(f"Study config differs from existing {study_config_path}; use a new output directory")
    study_config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    comparison_path = output / "comparison.csv"
    rows = pd.read_csv(comparison_path).to_dict("records") if comparison_path.exists() else []
    completed = {row["run_directory"] for row in rows}
    histories = []
    reference = None
    for current in runs:
        directory = Path(current["output"]["directory"])
        saved_config, saved_summary = directory / "config.json", directory / "summary.json"
        if directory.name in completed and saved_config.exists() and saved_summary.exists():
            if json.loads(saved_config.read_text()) != current:
                raise ValueError(f"Completed run config mismatch in {directory}")
            summary = json.loads(saved_summary.read_text())
        else:
            summary = run(current)
        with np.load(directory / "split_indices.npz") as split:
            signature = (summary["dataset_sha256"], split["train"].copy(), split["validation"].copy(),
                         summary["initial_validation"])
        if reference is None:
            reference = signature
        else:
            assert signature[0] == reference[0], "Dataset changed between runs"
            np.testing.assert_array_equal(signature[1], reference[1])
            np.testing.assert_array_equal(signature[2], reference[2])
            # Different architectures have different initial outputs and parameter shapes.
        validation = summary["validation"]
        layers = current["model"]["layers"]
        if directory.name not in completed:
            rows.append({"architecture": "-".join(map(str, layers)),
                         "parameters": sum(a * b + b for a, b in zip(layers[:-1], layers[1:])),
                         "learning_rate": current["optimizer"]["lr"], "seed": current["seed"],
                         "epochs_ran": summary["epochs_ran"], "best_epoch": summary["best_epoch"],
                         "train_accuracy": summary["train"]["accuracy"],
                         "validation_accuracy": validation["accuracy"], "validation_loss": validation["loss"],
                         "validation_macro_f1": validation["macro_f1"],
                         "validation_recall_5": validation["per_class"][5]["recall"],
                         "run_directory": directory.name})
            completed.add(directory.name)
        pd.DataFrame(rows).to_csv(comparison_path, index=False)
        histories.append(pd.read_csv(directory / "history.csv"))
    grouped = pd.DataFrame(rows).groupby("architecture").agg(
        parameters=("parameters", "first"), seeds=("seed", "nunique"),
        validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
        validation_loss_mean=("validation_loss", "mean"),
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
    ).reset_index()
    grouped.to_csv(output / "architecture_summary.csv", index=False)
    winner = grouped.sort_values(
        ["validation_accuracy_mean", "validation_loss_mean"],
        ascending=[False, True],
    ).iloc[0].to_dict()
    result = {"selection": "highest mean validation accuracy across seeds, tie: lowest mean validation loss",
              "winner": winner, "runs": rows, "same_split_verified": True, "shuffle_seed": config["training"]["shuffle_seed"],
              "dataset_sha256": reference[0], "external_test_evaluated": False,
              "epochs_budget": config["training"]["epochs"],
              "scope": "Repeated training seeds with fixed split and early stopping; not a global optimum"}
    (output / "study_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for row, history in zip(rows, histories):
        label = f"{row['architecture']} seed {row['seed']}"
        axes[0].plot(history.epoch, history.validation_loss, label=label)
        axes[1].plot(history.epoch, history.validation_accuracy, label=label)
    for ax, title in zip(axes, ["Error cuadrático en validación", "Accuracy de validación"]):
        ax.set(xlabel="Época", title=title)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "architecture_comparison.png", dpi=150)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return result


if __name__ == "__main__":
    run_study(config_from_cli(STUDY_DEFAULTS))
