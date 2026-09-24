import numpy as np
import pytest

from experiments.config import load_config
from experiments.runner import run_training
from nn import build, build_model
from training import Callback, EpochLogs, LossThreshold, ProgressPrinter, train


class Recorder(Callback):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_begin(self, epochs: int) -> None:
        self.events.append(f"begin {epochs}")

    def on_epoch_end(self, logs: EpochLogs) -> None:
        self.events.append(f"epoch {logs['epoch']}")

    def on_train_end(self, history: list[EpochLogs]) -> None:
        self.events.append(f"end {len(history)}")


def train_tiny(epochs: int, callbacks: list[Callback]) -> list[EpochLogs]:
    rng = np.random.default_rng(0)
    X = np.array([[-1.0], [1.0]])
    net = build_model({"layers": [1, 1], "activation": "identity"}, rng)
    return train(net, build("loss", "mse"), build("optimizer", "sgd"), X, X,
                 epochs=epochs, batch_size=None, rng=rng, callbacks=callbacks)


def test_trainer_calls_hooks_in_order():
    recorder = Recorder()

    train_tiny(epochs=2, callbacks=[recorder])

    assert recorder.events == ["begin 2", "epoch 1", "epoch 2", "end 2"]


def test_loss_threshold_stops_training_early():
    history = train_tiny(epochs=100, callbacks=[LossThreshold(threshold=1.0)])

    assert len(history) == 1


def test_progress_printer_respects_every(capsys):
    train_tiny(epochs=4, callbacks=[ProgressPrinter(every=2)])

    lines = capsys.readouterr().out.splitlines()
    printed_epochs = [line.split()[1] for line in lines if line.startswith("epoch")]
    assert printed_epochs == ["1/4", "2/4", "4/4"]
    assert lines[-1].startswith("Training finished after 4/4 epochs")


def test_epsilon_stops_on_post_update_full_training_loss():
    X = np.array([[1.0], [1.0]])
    Y = np.array([[1.0], [1.0]])
    config = load_config(experiment_defaults={
        "model": {"layers": [1, 1], "activation": "identity"},
        "optimizer": {"lr": 0.5},
        "training": {"epochs": 20, "batch_size": 1, "epsilon": 0.01},
        "callbacks": [],
    })

    result = run_training(config, X, Y)

    assert len(result.history) == 1
    assert result.history[-1]["loss"] <= 0.01
    expected = 0.5 * np.mean((result.net.forward(X) - Y) ** 2)
    assert result.history[-1]["loss"] == pytest.approx(expected)


def test_invalid_epsilon_is_rejected():
    X = np.array([[0.0], [1.0]])
    for epsilon in (-0.1, float("nan"), float("inf")):
        config = load_config(experiment_defaults={
            "model": {"layers": [1, 1]},
            "training": {"epsilon": epsilon},
            "callbacks": [],
        })
        with pytest.raises(ValueError, match="training.epsilon"):
            run_training(config, X, X)
