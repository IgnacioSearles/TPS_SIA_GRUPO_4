"""Evaluate the selected exercise 3 network under increasing Gaussian image noise."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nn  # noqa: F401 - register model components
from datasets.digit_dataset_loader import load_digit_arrays
from experiments.digits.baseline import digit_metrics
from data.augmentation import add_gaussian_noise
from nn.network import build_model, load_weights


def run_study(config):
    model_dir = Path(config["architecture_study"])
    study = json.loads((model_dir / "study_config.json").read_text())
    summary = json.loads((model_dir / "study_summary.json").read_text())
    architecture = [int(value) for value in summary["winner"]["architecture"].split("-")]
    model_config = dict(study["model"])
    model_config["layers"] = architecture
    net = build_model(model_config, np.random.default_rng(0))
    load_weights(net, model_dir / "winner_weights.npz")
    X_test, y_test = load_digit_arrays(config["test_file"])
    rows = []
    rng = np.random.default_rng(config["seed"])
    for sigma in config["sigmas"]:
        repeats = 1 if sigma == 0 else config["replicates"]
        for replicate in range(repeats):
            noisy = add_gaussian_noise(X_test, rng, sigma=sigma)
            metrics = digit_metrics(y_test, net.forward(noisy))
            row = {"sigma": sigma, "replicate": replicate,
                   "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
            row.update({f"recall_{digit}": item["recall"]
                        for digit, item in enumerate(metrics["per_class"])})
            rows.append(row)
            print(f"sigma={sigma:.2f} repetición={replicate + 1}/{repeats}: "
                  f"accuracy={metrics['accuracy']:.4%}, F1 macro={metrics['macro_f1']:.4f}", flush=True)

    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "noise_runs.csv", index=False)
    metrics_cols = ["accuracy", "macro_f1", *[f"recall_{digit}" for digit in range(10)]]
    summary_table = raw.groupby("sigma")[metrics_cols].agg(["mean", "std"])
    summary_table.columns = [f"{metric}_{stat}" for metric, stat in summary_table.columns]
    summary_table = summary_table.reset_index()
    summary_table.to_csv(output / "noise_summary.csv", index=False)
    result = {"architecture": architecture, "selected_by": summary["selection"],
              "samples": len(y_test), "sigmas": config["sigmas"],
              "replicates": config["replicates"], "uses_digits_test": True,
              "digits_test_was_evaluated_in_prior_reports": True,
              "interpretation": "descriptive robustness analysis; do not use to retune the model"}
    (output / "study_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    _plot(summary_table, output)
    print(summary_table.to_string(index=False), flush=True)
    return result


def _plot(summary_table, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    x = summary_table["sigma"]
    ax.errorbar(x, summary_table["accuracy_mean"], yerr=summary_table["accuracy_std"].fillna(0),
                marker="o", capsize=4, label="Accuracy")
    ax.errorbar(x, summary_table["macro_f1_mean"], yerr=summary_table["macro_f1_std"].fillna(0),
                marker="s", capsize=4, label="F1 macro")
    ax.set(title="Efecto del ruido gaussiano en dígitos", xlabel="Desvío del ruido σ",
           ylabel="Métrica", ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "noise_robustness.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="experiments/digits/configs/e3_noise.json")
    args = parser.parse_args()
    run_study(json.loads(Path(args.config).read_text()))
