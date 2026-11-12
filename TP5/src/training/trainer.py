"""Entrenamiento por minilotes, común a todos los modelos.

Si hay corrupción configurada, la red recibe la entrada corrompida y debe
reconstruir la limpia (Denoising Autoencoder); el ruido se sortea de nuevo en
cada lote.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.data.noise import Noise, build_noise
from src.models.autoencoder import Autoencoder
from src.nn.optimizers import Optimizer, build_optimizer
from src.options import ConfigurationError, Options

Array = np.ndarray
HistoryRow = dict[str, float]
Monitor = Callable[[Autoencoder], dict[str, float]]


@dataclass(frozen=True)
class TrainingPlan:
    epochs: int
    batch_size: int
    log_every: int
    optimizer: Optimizer
    corruption: Noise | None

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "log_every"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"training.{name} debe ser un entero positivo: {value}")


def build_training_plan(options: Options) -> TrainingPlan:
    """Sección `training`. La subsección `noise` activa el modo denoising."""
    noise_options = options.optional_section("noise")
    plan = TrainingPlan(
        epochs=options.take("epochs", 5000),
        batch_size=options.take("batch_size", 32),
        log_every=options.take("log_every", 100),
        optimizer=build_optimizer(options.section("optimizer")),
        corruption=None if noise_options is None else build_noise(noise_options),
    )
    options.ensure_consumed()
    return plan


class TrainingDiverged(FloatingPointError):
    """La pérdida dejó de ser finita; `history` tiene las filas registradas hasta entonces."""

    def __init__(self, epoch: int, history: list[HistoryRow]) -> None:
        super().__init__(f"La pérdida divergió en la época {epoch}; probar un learning_rate menor")
        self.epoch = epoch
        self.history = history


class Trainer:
    def __init__(
        self,
        model: Autoencoder,
        plan: TrainingPlan,
        monitor: Monitor,
        shuffle_rng: np.random.Generator,
        corruption_rng: np.random.Generator,
    ) -> None:
        self._model = model
        self._plan = plan
        self._monitor = monitor
        self._shuffle_rng = shuffle_rng
        self._corruption_rng = corruption_rng

    def train(self, samples: Array) -> list[HistoryRow]:
        """Entrena y devuelve una fila de historial cada `log_every` épocas.

        Cada fila tiene los términos de la pérdida promediados en la época y
        las métricas de `monitor` medidas sobre los datos limpios.
        """
        samples = samples.astype(self._model.precision, copy=False)
        history: list[HistoryRow] = []
        for epoch in range(1, self._plan.epochs + 1):
            loss_terms = self._run_epoch(samples, epoch)
            if loss_terms is None:
                raise TrainingDiverged(epoch, history)
            if epoch == 1 or epoch % self._plan.log_every == 0 or epoch == self._plan.epochs:
                row = {"epoch": epoch, **loss_terms, **self._monitor(self._model)}
                history.append(row)
                print(_format_progress(row), flush=True)
        return history

    def _run_epoch(self, samples: Array, epoch: int) -> dict[str, float] | None:
        """Términos de la pérdida promediados en la época; None si la pérdida divergió."""
        order = self._shuffle_rng.permutation(len(samples))
        totals: dict[str, float] = {}
        for start in range(0, len(samples), self._plan.batch_size):
            targets = samples[order[start : start + self._plan.batch_size]]
            inputs = self._corrupt(targets)
            loss_terms, gradients = self._model.loss_and_gradients(inputs, targets, epoch)
            if not np.isfinite(loss_terms["loss"]):
                return None
            self._plan.optimizer.step(self._model.parameters(), gradients)
            for name, value in loss_terms.items():
                totals[name] = totals.get(name, 0.0) + value * len(targets)
        return {name: total / len(samples) for name, total in totals.items()}

    def _corrupt(self, targets: Array) -> Array:
        if self._plan.corruption is None:
            return targets
        corrupted = self._plan.corruption.apply(targets, self._corruption_rng)
        return corrupted.astype(targets.dtype, copy=False)


def _format_progress(row: HistoryRow) -> str:
    details = ", ".join(f"{name}={value:.5g}" for name, value in row.items() if name != "epoch")
    return f"  época {int(row['epoch'])}: {details}"
