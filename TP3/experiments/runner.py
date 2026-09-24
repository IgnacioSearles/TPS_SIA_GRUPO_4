"""Builds every component from a config and trains a model on the given data."""

from dataclasses import dataclass

import numpy as np

import nn  # noqa: F401  (registers activations, losses, optimizers, initializers)
import training  # noqa: F401  (registers callbacks)
from experiments.config import Config
from nn.network import Sequential, build_model
from nn.registry import build
from training.callbacks import EpochLogs
from training.trainer import train


@dataclass
class TrainingResult:
    net: Sequential
    history: list[EpochLogs]


def run_training(config: Config, X: np.ndarray, Y: np.ndarray) -> TrainingResult:
    """Train a fresh model described by `config` on (X, Y)."""
    rng = np.random.default_rng(config["seed"])
    net = build_model(config["model"], rng)
    history = train(
        net,
        build("loss", config["loss"]),
        build("optimizer", config["optimizer"]),
        X,
        Y,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        rng=rng,
        callbacks=[build("callback", spec) for spec in config["callbacks"]],
    )
    return TrainingResult(net, history)
