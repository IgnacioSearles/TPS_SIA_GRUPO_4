"""Callbacks hook into the training loop without the trainer knowing what they do.

Built from configs like any other component:
    "callbacks": ["progress", {"name": "loss_threshold", "threshold": 0.0}]
"""

from nn.registry import register

# Per-epoch record produced by the trainer, e.g. {"epoch": 3, "loss": 0.12, "epoch_time": 0.004}.
EpochLogs = dict[str, float]


class Callback:
    """No-op base. Subclasses override only the hooks they need.

    Setting `stop_requested = True` asks the trainer to stop after the current epoch.
    """

    stop_requested: bool = False

    def on_train_begin(self, epochs: int) -> None:
        pass

    def on_epoch_end(self, logs: EpochLogs) -> None:
        pass

    def on_train_end(self, history: list[EpochLogs]) -> None:
        pass


@register("callback", "progress")
class ProgressPrinter(Callback):
    """Prints the epoch number and every logged value every `every` epochs."""

    def __init__(self, every: int = 1):
        if every < 1:
            raise ValueError(f"'every' must be >= 1, got {every}")
        self.every = every
        self._epochs = 0

    def on_train_begin(self, epochs: int) -> None:
        self._epochs = epochs

    def on_epoch_end(self, logs: EpochLogs) -> None:
        epoch = int(logs["epoch"])
        if epoch == 1 or epoch % self.every == 0:
            print(self._format(logs))

    def on_train_end(self, history: list[EpochLogs]) -> None:
        if not history:
            return
        print(f"Training finished after {len(history)}/{self._epochs} epochs: {self._format(history[-1])}")

    def _format(self, logs: EpochLogs) -> str:
        width = len(str(self._epochs))
        values = "  ".join(f"{key} {value:.6f}" for key, value in logs.items() if key != "epoch")
        return f"epoch {int(logs['epoch']):>{width}}/{self._epochs}  {values}"


@register("callback", "loss_threshold")
class LossThreshold(Callback):
    """Stops training once `monitor` drops to `threshold` or below.

    With threshold 0 this is the classic perceptron stopping rule: stop when every sample is right.
    """

    def __init__(self, threshold: float = 0.0, monitor: str = "loss"):
        self.threshold = threshold
        self.monitor = monitor

    def on_epoch_end(self, logs: EpochLogs) -> None:
        if self.monitor not in logs:
            raise KeyError(f"LossThreshold monitors '{self.monitor}', but logs only have {sorted(logs)}")
        if logs[self.monitor] <= self.threshold:
            self.stop_requested = True
