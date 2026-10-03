"""Architecture sweep on the expanded, deduplicated exercise 3 dataset."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import config_from_cli, deep_merge
from experiments.digits.ablation import Variant, train_variant
from experiments.digits.data import balanced_indices, merge_digit_files, split_train_validation
from nn.network import build_model, load_weights, save_weights

DEFAULTS = {
    "files": ["datasets/digits.csv", "datasets/more_digits.csv"],
    "validation_ratio": 0.2,
    "split_seed": 2026,
    "seeds": [42, 7, 21],
    "target_accuracy": 0.98,
    "model": {"layers": [784, 64, 32, 10], "activation": "tanh",
              "output_activation": "sigmoid", "initializer": "xavier"},
    "optimizer": {"name": "momentum", "lr": 0.1, "momentum": 0.9},
    "training": {"epochs": 400, "batch_size": 128,
                  "early_stopping": {"patience": 50, "min_delta": 0.0005}},
    "augmentation": {"max_rotation": 10.0, "max_scale": 0.1,
                     "max_shift": 2.0, "sigma": 0.0},
    "study": {"architectures": [[784, 64, 10], [784, 64, 32, 10],
                                  [784, 128, 128, 10], [784, 64, 64, 64, 10],
                                  [784, 64, 64, 64, 64, 10],
                                  [784, 64, 64, 64, 64, 64, 64, 10],
                                  [784, 64, 64, 64, 64, 64, 64, 64, 64, 10]]},
    "output": {"directory": "reports/digits_e3_architectures"},
}


def _run_config(config, architecture, seed, run_dir):
    current = copy.deepcopy(config)
    current["model"]["layers"] = architecture
    current["output"] = {"directory": str(run_dir)}
    current["seed"] = seed
    return current


def run_study(config):
    config = deep_merge(DEFAULTS, config)
    architectures = config["study"]["architectures"]
    if not architectures or not config["seeds"]:
        raise ValueError("At least one architecture and seed are required")
    for layers in architectures:
        if len(layers) < 3 or layers[0] != 784 or layers[-1] != 10 or any(n <= 0 for n in layers):
            raise ValueError(f"Invalid digit architecture: {layers}")

    data = merge_digit_files(config["files"])
    train_rows, validation_rows = split_train_validation(
        data.y, config["validation_ratio"], config["split_seed"])
    X_train, y_train = data.X[train_rows], data.y[train_rows]
    balance_rng = np.random.default_rng(config["split_seed"])
    balanced = balanced_indices(y_train, balance_rng)
    variant = Variant("balance+augment", list(config["files"]), True, True,
                      X_train[balanced], y_train[balanced], validation_rows,
                      data.X[validation_rows], data.y[validation_rows])

    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    study_path = output / "study_config.json"
    if study_path.exists() and json.loads(study_path.read_text()) != config:
        raise ValueError(f"Config differs from existing {study_path}; choose a new output directory")
    study_path.write_text(json.dumps(config, indent=2) + "\n")
    rows_path = output / "runs.csv"
    rows = pd.read_csv(rows_path).to_dict("records") if rows_path.exists() else []
    completed = {row["run_directory"] for row in rows}

    for arch_index, architecture in enumerate(architectures):
        for seed in config["seeds"]:
            run_dir = output / f"architecture_{arch_index:02d}" / f"seed_{seed}"
            relative = str(run_dir.relative_to(output))
            current = _run_config(config, architecture, seed, run_dir)
            protocol = {"architecture": architecture, "seed": seed, "split_seed": config["split_seed"],
                        "training_rows": len(variant.y_train), "config": config}
            protocol_path, summary_path = run_dir / "protocol.json", run_dir / "summary.json"
            history_path, weights_path = run_dir / "history.csv", run_dir / "best_weights.npz"
            if relative in completed and all(p.exists() for p in (protocol_path, summary_path, history_path, weights_path)):
                if json.loads(protocol_path.read_text()) != protocol:
                    raise ValueError(f"Saved run protocol differs in {run_dir}")
                result = json.loads(summary_path.read_text())
            else:
                result = train_variant(variant, current, seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                save_weights(result["net"], weights_path)
                pd.DataFrame(result["history"]).to_csv(history_path, index=False)
                protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
                summary = {"best": result["best"], "epochs_ran": result["epochs_ran"],
                           "train_accuracy": result["train_accuracy"]}
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                result = summary
            best = result["best"]
            if relative not in completed:
                rows.append({"architecture": "-".join(map(str, architecture)),
                             "hidden_layers": len(architecture) - 2,
                             "parameters": sum(a * b + b for a, b in zip(architecture[:-1], architecture[1:])),
                             "seed": seed, "epochs_ran": result["epochs_ran"],
                             "validation_accuracy": best["accuracy"],
                             "validation_macro_f1": best["macro_f1"],
                             "validation_loss": best["loss"],
                             "validation_recall_5": best["per_class"][5]["recall"],
                             "validation_recall_8": best["per_class"][8]["recall"],
                             "run_directory": relative})
                completed.add(relative)
                pd.DataFrame(rows).to_csv(rows_path, index=False)
            print(f"{relative}: accuracy={best['accuracy']:.4%}, F1 macro={best['macro_f1']:.4f}, "
                  f"época={result['epochs_ran']}", flush=True)

    table = pd.DataFrame(rows)
    grouped = table.groupby("architecture").agg(
        hidden_layers=("hidden_layers", "first"), parameters=("parameters", "first"),
        seeds=("seed", "nunique"), validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
        validation_macro_f1_std=("validation_macro_f1", "std"),
        validation_loss_mean=("validation_loss", "mean"),
    ).reset_index()
    target = config["target_accuracy"]
    eligible = grouped[grouped.validation_accuracy_mean >= target]
    if len(eligible):
        winner = eligible.sort_values(["validation_macro_f1_mean", "validation_accuracy_mean"],
                                      ascending=[False, False]).iloc[0]
        selection = f"among mean validation accuracy >= {target:.4f}, highest mean macro F1"
    else:
        winner = grouped.sort_values(["validation_accuracy_mean", "validation_macro_f1_mean"],
                                     ascending=[False, False]).iloc[0]
        selection = "no architecture met target; highest mean validation accuracy, macro F1 tie-break"
    grouped.to_csv(output / "architecture_summary.csv", index=False)

    winner_rows = table[table.architecture == winner.architecture]
    candidate = winner_rows[winner_rows.validation_accuracy >= target]
    if candidate.empty:
        candidate = winner_rows
        selected = candidate.sort_values(["validation_accuracy", "validation_macro_f1"],
                                         ascending=[False, False]).iloc[0]
    else:
        selected = candidate.sort_values(["validation_macro_f1", "validation_accuracy"],
                                         ascending=[False, False]).iloc[0]
    selected_dir = output / selected.run_directory
    selected_architecture = next(a for a in architectures if "-".join(map(str, a)) == winner.architecture)
    net = build_model({**config["model"], "layers": selected_architecture}, np.random.default_rng(0))
    load_weights(net, selected_dir / "best_weights.npz")
    save_weights(net, output / "winner_weights.npz")
    winner_dict = {"architecture": str(winner.architecture),
                   "hidden_layers": int(winner.hidden_layers), "parameters": int(winner.parameters),
                   "seeds": int(winner.seeds),
                   "validation_accuracy_mean": float(winner.validation_accuracy_mean),
                   "validation_accuracy_std": float(winner.validation_accuracy_std),
                   "validation_macro_f1_mean": float(winner.validation_macro_f1_mean),
                   "validation_macro_f1_std": float(winner.validation_macro_f1_std),
                   "validation_loss_mean": float(winner.validation_loss_mean)}
    summary = {"selection": selection, "target_accuracy": target,
               "winner": winner_dict, "selected_seed": int(selected.seed),
               "selected_run": selected.run_directory,
               "seeds": config["seeds"], "split_seed": config["split_seed"],
               "validation_samples": int(len(validation_rows)), "external_test_evaluated": False}
    (output / "study_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot(grouped, output)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _plot(grouped, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    positions = np.arange(len(grouped))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(positions, grouped.validation_accuracy_mean,
                     yerr=grouped.validation_accuracy_std, fmt="o", capsize=4)
    axes[0].axhline(0.98, color="red", linestyle="--", label="objetivo 98 %")
    axes[0].set(title="Accuracy media de validación", ylabel="Accuracy", xticks=positions,
                xticklabels=grouped.architecture, xlabel="Arquitectura")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()
    axes[1].errorbar(positions, grouped.validation_macro_f1_mean,
                     yerr=grouped.validation_macro_f1_std, fmt="o", capsize=4, color="darkorange")
    axes[1].set(title="F1 macro medio de validación", ylabel="F1 macro", xticks=positions,
                xticklabels=grouped.architecture, xlabel="Arquitectura")
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "architecture_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run_study(config_from_cli(DEFAULTS))
