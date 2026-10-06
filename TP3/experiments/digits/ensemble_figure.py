"""Figure for the ensemble: every network alone against their average, on validation and test.

Reads what `experiments.digits.ensemble --evaluate-test` saved; evaluates nothing.

Run from TP3:
    python -m experiments.digits.ensemble_figure [config.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.config import Config, deep_merge, read_json

DEFAULTS: Config = {
    "ensemble_directory": "reports/digits_e3_ensemble",
    "output": "reports/digits_e3_ensemble/ensemble_vs_single.png",
}


def load_accuracies(directory: Path) -> dict[str, tuple[np.ndarray, float]]:
    """Per split: the accuracy of each network alone and of the ensemble, in percent."""
    test = json.loads((directory / "test.json").read_text(encoding="utf-8"))
    members = [network["run_directory"] for network in test["networks"]]
    models = pd.read_csv(directory / "models.csv").set_index("run_directory")
    ensembles = pd.read_csv(directory / "ensembles.csv").set_index("ensemble")
    validation_single = 100 * models.loc[members, "validation_accuracy"].to_numpy()
    test_single = 100 * np.array([network["test_accuracy"] for network in test["networks"]])
    return {"Validación": (validation_single, 100 * ensembles.loc[test["ensemble"], "validation_accuracy"]),
            "Test": (test_single, 100 * test["test_accuracy"])}


def plot(accuracies: dict[str, tuple[np.ndarray, float]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6))
    jitter = np.random.default_rng(0)
    for position, (split, (single, ensemble)) in enumerate(accuracies.items()):
        offsets = jitter.uniform(-0.12, 0.12, len(single))
        ax.scatter(position - 0.15 + offsets, single, color="C0", alpha=0.7,
                   label="cada red sola" if position == 0 else None)
        ax.hlines(single.mean(), position - 0.3, position, color="C0", linestyle="--",
                  label="promedio de las redes" if position == 0 else None)
        ax.scatter(position + 0.25, ensemble, color="C3", marker="*", s=260, zorder=3,
                   label="ensamble de las 12" if position == 0 else None)
        ax.annotate(f"{ensemble:.2f} %".replace(".", ","), (position + 0.25, ensemble), textcoords="offset points",
                    xytext=(0, 12), ha="center")
    ax.set_xticks(range(len(accuracies)), list(accuracies))
    ax.set_xlim(-0.6, len(accuracies) - 0.4)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.12)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble against single networks, validation and test")
    parser.add_argument("config", nargs="?", help="optional JSON config; omitted keys use defaults")
    args = parser.parse_args()
    config = deep_merge(DEFAULTS, read_json(args.config)) if args.config else DEFAULTS
    output = Path(config["output"])
    plot(load_accuracies(Path(config["ensemble_directory"])), output)
    print(f"Figure written to {output}")


if __name__ == "__main__":
    main()
