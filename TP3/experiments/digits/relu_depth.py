"""ReLU in deeper networks: gradient profile at initialization and depth-sweep comparison.

1. Gradient profile: for a network with `depth` hidden layers of `width` units,
   the mean |dL/dW| of every Dense layer on one batch of training data, before
   any update. Vanishing gradients show up as layers near the input receiving
   much smaller gradients than layers near the output.
2. Depth comparison: reads two depth sweeps produced by
   `experiments.digits.depth_sweep` (same protocol, different activation) and
   plots validation accuracy against depth.

Run from TP3, after both depth sweeps exist:
    python -m experiments.digits.relu_depth [config.json]
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.splits import stratified_holdout
from datasets.digit_dataset_loader import load_digit_arrays
from experiments.config import Config, deep_merge, read_json
from experiments.digits.baseline import one_hot
from nn.layers import Dense
from nn.network import build_model
from nn.registry import build

DEFAULTS: Config = {
    "data": {"path": "datasets/digits.csv", "validation_ratio": 0.2, "split_seed": 42},
    "gradient_profile": {
        "depth": 8,
        "width": 64,
        "batch_size": 1000,
        "seeds": [42, 7, 21],
        # Each activation with the initializer designed for it.
        "configurations": [
            {"label": "sigmoide + Xavier", "activation": "sigmoid", "initializer": "xavier"},
            {"label": "tanh + Xavier", "activation": "tanh", "initializer": "xavier"},
            {"label": "ReLU + He", "activation": "relu", "initializer": "he"},
        ],
    },
    # Same protocol except activation, initializer and learning rate. ReLU's η = 0,01 was chosen on
    # validation at depth 4 (results/digits_relu_lr_check); η = 0,1 is tanh's protocol, kept to show the collapse.
    "depth_sweeps": {
        "tanh + Xavier, η = 0,1": "results/digits_depth_sweep",
        "ReLU + He, η = 0,01": "results/digits_depth_sweep_relu",
        "ReLU + He, η = 0,1": "results/digits_depth_sweep_relu_lr0.1",
    },
    "output_directory": "results/digits_relu_depth",
}


def gradient_profile(config: Config) -> pd.DataFrame:
    """Mean |dL/dW| per Dense layer at initialization, for every configuration and seed."""
    settings = config["gradient_profile"]
    X, labels = load_digit_arrays(Path(config["data"]["path"]))
    train_rows, _ = stratified_holdout(labels, config["data"]["validation_ratio"], config["data"]["split_seed"])
    batch = train_rows[: settings["batch_size"]]
    X_batch, targets = X[batch], one_hot(labels[batch])
    layers = [784, *([settings["width"]] * settings["depth"]), 10]
    rows = []
    for configuration in settings["configurations"]:
        for seed in settings["seeds"]:
            model = {"layers": layers, "activation": configuration["activation"],
                     "output_activation": "sigmoid", "initializer": configuration["initializer"]}
            net = build_model(model, np.random.default_rng(seed))
            loss = build("loss", "mse")
            loss.forward(net.forward(X_batch), targets)
            net.backward(loss.backward())
            dense_layers = [layer for layer in net.layers if isinstance(layer, Dense)]
            for index, layer in enumerate(dense_layers, start=1):
                rows.append({"configuration": configuration["label"], "seed": seed, "layer": index,
                             "mean_abs_gradient": float(np.mean(np.abs(layer.W.grad)))})
    return pd.DataFrame(rows)


def plot_gradient_profile(profile: pd.DataFrame, depth: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, rows in profile.groupby("configuration", sort=False):
        mean = rows.groupby("layer")["mean_abs_gradient"].mean()
        ax.plot(mean.index, mean.values, "o-", label=label)
    ax.set(yscale="log", xlabel="Capa (1 = la más cercana a la entrada)", ylabel="media de |∂L/∂W|",
           title=f"Gradiente por capa al inicializar ({depth} capas ocultas de 64)")
    ax.set_xticks(sorted(profile["layer"].unique()))
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def depth_comparison(sweeps: dict[str, str]) -> pd.DataFrame:
    """Join the depth summaries of several sweeps into one table."""
    tables = []
    for label, directory in sweeps.items():
        summary = pd.read_csv(Path(directory) / "depth_summary.csv")
        summary.insert(0, "configuration", label)
        tables.append(summary)
    return pd.concat(tables, ignore_index=True)


def plot_depth_comparison(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, rows in table.groupby("configuration", sort=False):
        ax.errorbar(rows["hidden_layers"], 100 * rows["validation_accuracy_mean"],
                    yerr=100 * rows["validation_accuracy_std"].fillna(0), fmt="o-", capsize=4, label=label)
    ax.set(xlabel="Capas ocultas (64 neuronas cada una)", ylabel="Accuracy de validación (%), media de 3 semillas",
           title="Accuracy según profundidad")
    ax.set_xticks(sorted(table["hidden_layers"].unique()))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gradient profile and depth comparison for ReLU")
    parser.add_argument("config", nargs="?", help="optional JSON config; omitted keys use defaults")
    args = parser.parse_args()
    config = deep_merge(DEFAULTS, read_json(args.config)) if args.config else DEFAULTS
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)

    profile = gradient_profile(config)
    profile.to_csv(output / "gradient_profile.csv", index=False)
    plot_gradient_profile(profile, config["gradient_profile"]["depth"], output / "gradient_profile.png")
    summary = profile.groupby(["configuration", "layer"])["mean_abs_gradient"].mean().unstack("layer")
    print("Mean |dL/dW| per layer at initialization:\n", summary.to_string(float_format="{:.2e}".format))
    print("\nFirst layer / last layer ratio:\n", (summary.iloc[:, 0] / summary.iloc[:, -1]).to_string())

    if all((Path(directory) / "depth_summary.csv").exists() for directory in config["depth_sweeps"].values()):
        table = depth_comparison(config["depth_sweeps"])
        table.to_csv(output / "depth_comparison.csv", index=False)
        plot_depth_comparison(table, output / "depth_comparison.png")
        print("\n", table[["configuration", "hidden_layers", "validation_accuracy_mean",
                            "validation_accuracy_std"]].to_string(index=False))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
