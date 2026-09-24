"""Builds every component from a config and trains a model on the given data."""

from dataclasses import dataclass

import numpy as np

import nn  # noqa: F401  (registers activations, losses, optimizers, initializers)
import training  # noqa: F401  (registers callbacks)
from experiments.config import Config
from experiments.evaluation import classification_metrics, probability_metrics, save_results, save_weights
from nn.network import Sequential, build_model
from nn.registry import build
from training.callbacks import EpochLogs, LossThreshold
from training.trainer import train


@dataclass
class TrainingResult:
    net: Sequential
    history: list[EpochLogs]
    metrics: dict | None = None


def run_training(config: Config, X: np.ndarray, Y: np.ndarray) -> TrainingResult:
    """Train a fresh model described by `config` on (X, Y)."""
    epsilon = config["training"].get("epsilon")
    if epsilon is not None and (not np.isfinite(epsilon) or epsilon < 0):
        raise ValueError(f"training.epsilon must be a finite number >= 0 or null, got {epsilon}")
    rng = np.random.default_rng(config["seed"])
    net = build_model(config["model"], rng)
    callbacks = [build("callback", spec) for spec in config["callbacks"]]
    if epsilon is not None:
        callbacks.append(LossThreshold(threshold=epsilon))
    history = train(
        net,
        build("loss", config["loss"]),
        build("optimizer", config["optimizer"]),
        X,
        Y,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        rng=rng,
        callbacks=callbacks,
    )
    evaluation = config.get("evaluation", {})
    if evaluation.get("classification", False) and evaluation.get("probability", False):
        raise ValueError("Choose one evaluation mode: classification or probability")
    predictions = net.forward(X)
    if evaluation.get("probability", False):
        metrics = probability_metrics(Y, predictions)
    elif evaluation.get("classification", False):
        metrics = classification_metrics(Y, predictions)
    else:
        metrics = None

    output = config.get("output", {})
    if weights_path := output.get("weights_path"):
        save_weights(net, weights_path)
    if results_path := output.get("results_path"):
        save_results(history, metrics, results_path)

    return TrainingResult(net, history, metrics)
