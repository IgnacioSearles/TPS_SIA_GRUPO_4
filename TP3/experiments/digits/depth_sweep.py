"""Sequential hidden-depth study for exercise 2 (development split only).

All hidden layers have a fixed width so the sweep isolates depth. Each depth is
repeated across seeds; stop after `patience` consecutive depths fail to improve
the mean validation accuracy by `min_delta`.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import config_from_cli, deep_merge
from experiments.digits.baseline import DEFAULTS, run

DEFAULT_STUDY = {
    "study": {"max_hidden_layers": 8, "width": 64, "seeds": [42, 7, 21],
              "patience": 2, "min_delta": 0.002, "stop_on_plateau": True},
    "training": {"epochs": 350, "shuffle_seed": 42,
                  "early_stopping": {"patience": 40, "min_delta": 0.0005}},
    "output": {"directory": "results/digits_depth_sweep"},
}
STUDY_DEFAULTS = deep_merge(DEFAULTS, DEFAULT_STUDY)


def run_study(config):
    study = config["study"]
    max_depth, width = study["max_hidden_layers"], study["width"]
    seeds, patience, min_delta = study["seeds"], study["patience"], study["min_delta"]
    if max_depth < 1 or width < 1 or not seeds or patience < 1 or min_delta < 0:
        raise ValueError("Invalid depth sweep limits, width, seeds, patience, or min_delta")
    if config["training"].get("shuffle_seed") is None:
        raise ValueError("A common shuffle_seed is required for architecture comparisons")

    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "study_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        config_without_stop = copy.deepcopy(config)
        previous_study = dict(previous.get("study", {}))
        current_study = dict(config["study"])
        for key in ("max_hidden_layers", "patience", "min_delta", "stop_on_plateau"):
            previous_study.pop(key, None)
            current_study.pop(key, None)
        previous["study"], config_without_stop["study"] = previous_study, current_study
        if previous != config_without_stop:
            raise ValueError(f"Training protocol differs from existing {config_path}; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    runs_path = output / "runs.csv"
    rows = pd.read_csv(runs_path).to_dict("records") if runs_path.exists() else []
    completed = {(int(row["hidden_layers"]), int(row["seed"])) for row in rows}
    depth_means = []
    stop_reason = "reached_max_hidden_layers"

    for depth in range(1, max_depth + 1):
        scores = []
        for seed in seeds:
            current = copy.deepcopy(config)
            current.pop("study")
            current["seed"] = seed
            current["model"]["layers"] = [784, *([width] * depth), 10]
            current["output"]["directory"] = str(output / f"depth_{depth:02d}" / f"seed_{seed}")
            run_dir = Path(current["output"]["directory"])
            saved_config = run_dir / "config.json"
            saved_summary = run_dir / "summary.json"
            if (depth, seed) in completed and saved_config.exists() and saved_summary.exists():
                if json.loads(saved_config.read_text()) != current:
                    raise ValueError(f"Completed run config mismatch in {run_dir}")
                summary = json.loads(saved_summary.read_text())
            else:
                summary = run(current)
            val = summary["validation"]
            row = {"hidden_layers": depth, "width": width, "seed": seed,
                   "architecture": "-".join(map(str, current["model"]["layers"])),
                   "parameters": sum(a * b + b for a, b in zip(current["model"]["layers"][:-1], current["model"]["layers"][1:])),
                   "epochs_ran": summary["epochs_ran"], "best_epoch": summary["best_epoch"],
                   "validation_accuracy": val["accuracy"],
                   "validation_loss": val["loss"], "validation_macro_f1": val["macro_f1"],
                   "validation_recall_5": val["per_class"][5]["recall"],
                   "run_directory": str(Path(current["output"]["directory"]).relative_to(output))}
            if (depth, seed) not in completed:
                rows.append(row)
            completed.add((depth, seed))
            scores.append(val["accuracy"])
            pd.DataFrame(rows).to_csv(runs_path, index=False)
        mean = float(np.mean(scores))
        depth_means.append(mean)
        # Require two consecutive non-improvements by default. A one-depth wobble
        # does not stop the sweep; the minimum accuracy difference is configurable.
        if study.get("stop_on_plateau", True) and len(depth_means) > patience:
            recent_best = max(depth_means[:-patience])
            if all(value <= recent_best + min_delta for value in depth_means[-patience:]):
                stop_reason = f"{patience}_consecutive_depths_without_improvement_over_{recent_best:.6f}"
                break

    grouped = pd.DataFrame(rows).groupby("hidden_layers").agg(
        validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
        validation_loss_mean=("validation_loss", "mean"),
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
        validation_macro_f1_std=("validation_macro_f1", "std"),
        validation_recall_5_mean=("validation_recall_5", "mean"),
        parameters=("parameters", "first"),
    ).reset_index()
    grouped.to_csv(output / "depth_summary.csv", index=False)
    best_depth = int(grouped.loc[grouped.validation_accuracy_mean.idxmax(), "hidden_layers"])
    result = {"selection": "highest mean validation accuracy across seeds",
              "best_depth": best_depth, "best_mean_validation_accuracy": float(grouped.validation_accuracy_mean.max()),
              "stop_reason": stop_reason, "depths_completed": grouped.hidden_layers.astype(int).tolist(),
              "seeds": seeds, "width": width, "patience": patience, "min_delta": min_delta,
              "stop_on_plateau": study.get("stop_on_plateau", True),
              "external_test_evaluated": False}
    (output / "study_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_plots(output, grouped, rows)
    print(grouped.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _write_plots(output: Path, grouped: pd.DataFrame, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = grouped["hidden_layers"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(depths, grouped["validation_accuracy_mean"],
                     yerr=grouped["validation_accuracy_std"].fillna(0), marker="o", capsize=4)
    axes[0].set(title="Accuracy de validación por profundidad", xlabel="Capas ocultas",
                ylabel="Accuracy", xticks=depths)
    axes[1].errorbar(depths, grouped["validation_macro_f1_mean"],
                     yerr=grouped["validation_macro_f1_std"].fillna(0), marker="o", capsize=4,
                     color="darkorange")
    axes[1].set(title="F1 macro de validación por profundidad", xlabel="Capas ocultas",
                ylabel="F1 macro", xticks=depths)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "depth_comparison.png", dpi=160)
    plt.close(fig)

    histories = {}
    for row in rows:
        path = output / row["run_directory"] / "history.csv"
        history = pd.read_csv(path)
        histories.setdefault(int(row["hidden_layers"]), []).append(history)
    fig, ax = plt.subplots(figsize=(9, 5))
    for depth, runs in sorted(histories.items()):
        max_epoch = max(len(history) for history in runs)
        values = np.full((len(runs), max_epoch), np.nan)
        for index, history in enumerate(runs):
            values[index, :len(history)] = history["validation_accuracy"].to_numpy()
        epochs = np.arange(1, max_epoch + 1)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        ax.plot(epochs, mean, label=f"{depth} capa{'s' if depth != 1 else ''}")
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.12)
    ax.set(title="Curvas de validación por profundidad (media ± DE)",
           xlabel="Época", ylabel="Accuracy de validación")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output / "depth_learning_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run_study(config_from_cli(STUDY_DEFAULTS))
