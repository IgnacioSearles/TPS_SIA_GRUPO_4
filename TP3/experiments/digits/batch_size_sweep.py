"""Compare mini-batch sizes on the selected exercise 3 digit model."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import deep_merge, read_json
from experiments.digits.ablation import Variant, train_variant
from experiments.digits.data import balanced_indices, merge_digit_files, split_train_validation

DEFAULTS = {
    "files": ["datasets/digits.csv", "datasets/more_digits.csv"],
    "validation_ratio": 0.2,
    "split_seed": 2026,
    "seeds": [42, 7, 21],
    "model": {"layers": [784, 128, 128, 10], "activation": "tanh",
              "output_activation": "sigmoid", "initializer": "xavier"},
    "optimizer": {"name": "momentum", "lr": 0.1, "momentum": 0.9},
    "training": {"epochs": 400, "batch_size": 128,
                  "early_stopping": {"patience": 50, "min_delta": 0.0005}},
    "augmentation": {"max_rotation": 10.0, "max_scale": 0.1,
                     "max_shift": 2.0, "sigma": 0.0},
    "study": {"batch_sizes": [32, 64, 128, 256], "balance": True, "augment": True},
    "output": {"directory": "reports/digits_e3_batch_sizes"},
}


def run_study(config):
    config = deep_merge(DEFAULTS, config)
    batch_sizes = config["study"]["batch_sizes"]
    if not batch_sizes or any(type(size) is not int or size < 1 for size in batch_sizes):
        raise ValueError("study.batch_sizes must contain positive integers")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("study.batch_sizes must be unique")
    if not config["seeds"]:
        raise ValueError("At least one seed is required")

    data = merge_digit_files(config["files"])
    train_rows, validation_rows = split_train_validation(
        data.y, config["validation_ratio"], config["split_seed"])
    X_train, y_train = data.X[train_rows], data.y[train_rows]
    if config["study"]["balance"]:
        indices = balanced_indices(y_train, np.random.default_rng(config["split_seed"]))
        X_train, y_train = X_train[indices], y_train[indices]
    variant = Variant(
        "balance+augment" if config["study"]["balance"] and config["study"]["augment"]
        else "batch-size study",
        list(config["files"]), config["study"]["balance"], config["study"]["augment"],
        X_train, y_train, validation_rows, data.X[validation_rows], data.y[validation_rows],
    )

    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    study_path = output / "study_config.json"
    if study_path.exists() and json.loads(study_path.read_text()) != config:
        raise ValueError(f"Config differs from {study_path}; choose a new output directory")
    study_path.write_text(json.dumps(config, indent=2) + "\n")

    runs_path = output / "runs.csv"
    rows = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    steps_by_epoch = {
        size: math.ceil(len(variant.y_train) / size) for size in batch_sizes
    }

    for batch_size in batch_sizes:
        for seed in config["seeds"]:
            key = (batch_size, seed)
            run_dir = output / f"batch_{batch_size:03d}" / f"seed_{seed}"
            relative = str(run_dir.relative_to(output))
            protocol = {"config": config, "batch_size": batch_size, "seed": seed,
                        "training_samples_per_epoch": len(variant.y_train),
                        "steps_per_epoch": steps_by_epoch[batch_size]}
            protocol_path = run_dir / "protocol.json"
            summary_path = run_dir / "run_summary.json"
            history_path = run_dir / "history.csv"

            if all(path.exists() for path in (protocol_path, summary_path, history_path)):
                if json.loads(protocol_path.read_text()) != protocol:
                    raise ValueError(f"Saved run protocol differs in {run_dir}")
                result = json.loads(summary_path.read_text())
            else:
                run_config = deep_merge(config, {"training": {"batch_size": batch_size}})
                start = time.perf_counter()
                result = train_variant(variant, run_config, seed)
                elapsed = time.perf_counter() - start
                run_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(result["history"]).to_csv(history_path, index=False)
                protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
                summary = {"best": result["best"], "epochs_ran": result["epochs_ran"],
                           "train_accuracy": result["train_accuracy"],
                           "training_seconds": elapsed}
                summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
                result = summary

            best = result["best"]
            rows = [row for row in rows
                    if (int(row["batch_size"]), int(row["seed"])) != key]
            rows.append({"batch_size": batch_size, "seed": seed,
                         "training_samples_per_epoch": len(variant.y_train),
                         "steps_per_epoch": steps_by_epoch[batch_size],
                         "optimizer_updates": result["epochs_ran"] * steps_by_epoch[batch_size],
                         "epochs_ran": result["epochs_ran"],
                         "best_epoch": best["epoch"],
                         "training_seconds": result["training_seconds"],
                         "train_accuracy": result["train_accuracy"],
                         "validation_accuracy": best["accuracy"],
                         "validation_macro_f1": best["macro_f1"],
                         "validation_loss": best["loss"],
                         "validation_recall_5": best["per_class"][5]["recall"],
                         "validation_recall_8": best["per_class"][8]["recall"],
                         "run_directory": relative})
            pd.DataFrame(rows).to_csv(runs_path, index=False)
            print(f"lote={batch_size} semilla={seed}: accuracy={best['accuracy']:.4%}, "
                  f"F1 macro={best['macro_f1']:.4f}, época={result['epochs_ran']}, "
                  f"actualizaciones={result['epochs_ran'] * steps_by_epoch[batch_size]}",
                  flush=True)

    table = pd.DataFrame(rows)
    grouped = table.groupby("batch_size").agg(
        seeds=("seed", "nunique"), steps_per_epoch=("steps_per_epoch", "first"),
        validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
        validation_macro_f1_std=("validation_macro_f1", "std"),
        validation_loss_mean=("validation_loss", "mean"),
        epochs_ran_mean=("epochs_ran", "mean"),
        optimizer_updates_mean=("optimizer_updates", "mean"),
        training_seconds_mean=("training_seconds", "mean"),
        training_seconds_std=("training_seconds", "std"),
    ).reset_index()
    grouped.to_csv(output / "batch_size_summary.csv", index=False)
    _plot(grouped, output)
    report = {
        "batch_sizes": batch_sizes,
        "seeds": config["seeds"],
        "split_seed": config["split_seed"],
        "validation_samples": int(len(validation_rows)),
        "training_samples_per_epoch": int(len(variant.y_train)),
        "architecture": config["model"]["layers"],
        "optimizer": config["optimizer"],
        "validation_only": True,
    }
    (output / "study_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(grouped.to_string(index=False), flush=True)
    return report


def _plot(grouped, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].errorbar(grouped.batch_size, grouped.validation_accuracy_mean,
                     yerr=grouped.validation_accuracy_std, marker="o", capsize=4)
    axes[0].set(title="Accuracy de validación", xlabel="Tamaño de lote", ylabel="Accuracy")
    axes[1].errorbar(grouped.batch_size, grouped.validation_macro_f1_mean,
                     yerr=grouped.validation_macro_f1_std, marker="o", capsize=4,
                     color="darkorange")
    axes[1].set(title="F1 macro de validación", xlabel="Tamaño de lote", ylabel="F1 macro")
    axes[2].errorbar(grouped.batch_size, grouped.training_seconds_mean,
                     yerr=grouped.training_seconds_std.fillna(0), marker="o", capsize=4,
                     color="seagreen")
    axes[2].set(title="Tiempo medio por entrenamiento", xlabel="Tamaño de lote", ylabel="Segundos")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "batch_size_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?",
                        default="experiments/digits/configs/e3_batch_sizes.json")
    args = parser.parse_args()
    run_study(read_json(args.config))
