"""Validation: a step perceptron learns the logical AND.

Usage:
    python -m experiments.and_perceptron [path/to/config.json]

Every config key is optional; see experiments/config.py for how defaults are layered.
"""

import numpy as np

from experiments.config import Config, config_from_cli
from experiments.runner import run_training
from nn.network import Sequential

# Inputs and targets use {-1, 1} (not {0, 1}) to match the step activation's outputs.
AND_INPUTS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
AND_TARGETS = np.array([[-1], [-1], [-1], [1]], dtype=float)

EXPERIMENT_DEFAULTS: Config = {
    "model": {"layers": [2, 1], "activation": "step"},
    "optimizer": {"lr": 0.1},
    "training": {"epochs": 20, "batch_size": 1},
    "evaluation": {"classification": True},
    "callbacks": [
        {"name": "progress", "every": 1},
        {"name": "loss_threshold", "threshold": 0.0},
    ],
}


def print_report(net: Sequential, metrics: dict | None = None) -> None:
    dense = net.layers[0]
    w1, w2 = dense.W.value[:, 0]
    bias = dense.b.value[0]
    predictions = net.forward(AND_INPUTS)

    print(f"\nDecision boundary: {w1:+.3f}*x1 {w2:+.3f}*x2 {bias:+.3f} = 0")
    print("\n x1  x2 | target  predicted")
    for (x1, x2), target, predicted in zip(AND_INPUTS, AND_TARGETS[:, 0], predictions[:, 0]):
        mark = "ok" if target == predicted else "WRONG"
        print(f"{x1:+.0f}  {x2:+.0f} |  {target:+.0f}      {predicted:+.0f}    {mark}")

    accuracy = float(np.mean(predictions == AND_TARGETS))
    print(f"\nAccuracy: {accuracy:.0%}")
    if metrics is not None:
        print("Confusion matrix (rows=true, columns=predicted):")
        print(np.array(metrics["confusion_matrix"]))
        print(f"Macro F1: {metrics['macro_avg']['f1']:.3f}")


def main() -> None:
    config = config_from_cli(EXPERIMENT_DEFAULTS)
    result = run_training(config, AND_INPUTS, AND_TARGETS)
    print_report(result.net, result.metrics)


if __name__ == "__main__":
    main()
