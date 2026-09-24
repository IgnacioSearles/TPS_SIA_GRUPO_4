"""Training loop. Only talks to the network / loss / optimizer / callback interfaces."""

import time
from typing import Any, Iterator, Sequence

import numpy as np

from nn.network import Sequential
from training.callbacks import Callback, EpochLogs


def batches(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int | None,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield shuffled (x, y) batches.

    batch_size=1 is online learning, None is full batch, anything else is mini-batch.
    """
    n_samples = len(X)
    if len(Y) != n_samples:
        raise ValueError(f"X has {n_samples} samples but Y has {len(Y)}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be >= 1 or None, got {batch_size}")

    size = n_samples if batch_size is None else batch_size
    order = rng.permutation(n_samples)
    for start in range(0, n_samples, size):
        indices = order[start:start + size]
        yield X[indices], Y[indices]


def train_epoch(
    net: Sequential,
    loss: Any,
    optimizer: Any,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int | None,
    rng: np.random.Generator,
) -> float:
    """One pass over the data. Returns the mean of the batch losses (each measured before its update)."""
    batch_losses = []
    for xb, yb in batches(X, Y, batch_size, rng):
        batch_losses.append(loss.forward(net.forward(xb), yb))
        net.backward(loss.backward())
        optimizer.step(net.params())
    return float(np.mean(batch_losses))


def train(
    net: Sequential,
    loss: Any,
    optimizer: Any,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int,
    batch_size: int | None,
    rng: np.random.Generator,
    callbacks: Sequence[Callback] = (),
) -> list[EpochLogs]:
    """Train `net` in place and return one log record per epoch run.

    Stops early if any callback sets `stop_requested`.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    for callback in callbacks:
        callback.on_train_begin(epochs)

    history: list[EpochLogs] = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        epoch_loss = train_epoch(net, loss, optimizer, X, Y, batch_size, rng)
        logs: EpochLogs = {"epoch": epoch, "loss": epoch_loss, "epoch_time": time.perf_counter() - start}
        history.append(logs)

        for callback in callbacks:
            callback.on_epoch_end(logs)
        if any(callback.stop_requested for callback in callbacks):
            break

    for callback in callbacks:
        callback.on_train_end(history)
    return history
