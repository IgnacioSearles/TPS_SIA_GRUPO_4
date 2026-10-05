"""Choose the exercise 3 weight decay on validation only.

Reads the runs written by `experiments.digits.architecture_e3` for each λ (same
protocol as the final exercise 3 model, only `optimizer.weight_decay` changes)
and the existing λ = 0 runs of the final model. The selected λ is the one with
the highest mean validation accuracy; it is adopted only if it beats λ = 0.
The test set is not read here: scoring the selected runs is a separate, single
step with `experiments.digits.final_test_e3`.

Run from TP3:
    python -m experiments.digits.l2_e3_selection [config.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import Config, deep_merge, read_json

DEFAULTS: Config = {
    "runs": {
        "0": "reports/digits_e3_final/entrenamiento",
        "0.0001": "reports/digits_e3_l2/wd_0.0001",
        "0.0003": "reports/digits_e3_l2/wd_0.0003",
        "0.001": "reports/digits_e3_l2/wd_0.001",
    },
    # One λ = 0 run retrained now, to check it reproduces the saved final-model run.
    "reproduction": "reports/digits_e3_l2/reproduccion_wd_0",
    "output_directory": "reports/digits_e3_l2",
}


def load_runs(runs: dict[str, str]) -> pd.DataFrame:
    tables = []
    for weight_decay, directory in runs.items():
        table = pd.read_csv(Path(directory) / "runs.csv")
        table.insert(0, "weight_decay", float(weight_decay))
        table["directory"] = directory
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    baseline = table[table.weight_decay == 0].set_index("seed").validation_accuracy
    table = table.assign(gain_vs_l2_0=lambda rows: rows.validation_accuracy - rows.seed.map(baseline))
    return table.groupby("weight_decay").agg(
        seeds=("seed", "nunique"),
        validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
        validation_macro_f1_mean=("validation_macro_f1", "mean"),
        validation_recall_5_mean=("validation_recall_5", "mean"),
        validation_recall_8_mean=("validation_recall_8", "mean"),
        gain_vs_l2_0_mean=("gain_vs_l2_0", "mean"),
        seeds_improved=("gain_vs_l2_0", lambda gains: int((gains > 0).sum())),
        median_epochs=("epochs_ran", "median"),
        directory=("directory", "first"),
    ).reset_index()


def plot(table: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{value:g}" for value in summary.weight_decay]
    positions = {value: index for index, value in enumerate(summary.weight_decay)}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed, rows in table.groupby("seed"):
        ax.plot([positions[v] for v in rows.weight_decay], 100 * rows.validation_accuracy,
                "o", alpha=0.5, label=f"semilla {seed}")
    ax.errorbar(range(len(summary)), 100 * summary.validation_accuracy_mean,
                yerr=100 * summary.validation_accuracy_std, fmt="s-", color="black", capsize=4, label="media")
    ax.axhline(98, color="C3", linestyle=":", label="objetivo 98 %")
    ax.set_xticks(range(len(summary)), labels)
    ax.set(xlabel="λ (weight decay)", ylabel="Accuracy de validación (%)",
           title="Ejercicio 3: L2 sobre 784-128-128-10 (validación)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose the exercise 3 weight decay on validation")
    parser.add_argument("config", nargs="?", help="optional JSON config; omitted keys use defaults")
    args = parser.parse_args()
    config = deep_merge(DEFAULTS, read_json(args.config)) if args.config else DEFAULTS
    output = Path(config["output_directory"])

    table = load_runs(config["runs"])
    summary = summarize(table)
    table.to_csv(output / "runs_by_weight_decay.csv", index=False)
    summary.to_csv(output / "summary_by_weight_decay.csv", index=False)
    plot(table, summary, output / "l2_e3_validation.png")

    best = summary.sort_values(["validation_accuracy_mean", "validation_macro_f1_mean"], ascending=False).iloc[0]
    baseline = summary[summary.weight_decay == 0].iloc[0]
    adopted = bool(best.weight_decay > 0 and best.validation_accuracy_mean > baseline.validation_accuracy_mean)
    reproduction = None
    reproduction_runs = Path(config["reproduction"]) / "runs.csv"
    if reproduction_runs.exists():
        retrained = pd.read_csv(reproduction_runs).iloc[0]
        saved = table[(table.weight_decay == 0) & (table.seed == retrained.seed)].iloc[0]
        reproduction = {"seed": int(retrained.seed), "saved_validation_accuracy": float(saved.validation_accuracy),
                        "retrained_validation_accuracy": float(retrained.validation_accuracy),
                        "identical": bool(np.isclose(saved.validation_accuracy, retrained.validation_accuracy))}
    selection = {
        "rule": "highest mean validation accuracy (macro F1 tie-break); adopted only if it beats λ = 0",
        "selected_weight_decay": float(best.weight_decay), "adopted": adopted,
        "selected_directory": best.directory,
        "validation_accuracy_mean": float(best.validation_accuracy_mean),
        "baseline_validation_accuracy_mean": float(baseline.validation_accuracy_mean),
        "reproduction_check": reproduction, "external_test_evaluated": False,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(summary.drop(columns="directory").to_string(index=False))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
