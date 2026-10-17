"""Tipos de historiales y evaluaciones; diccionarios solo al serializar."""
from dataclasses import asdict, dataclass
from typing import Literal


StopReason = Literal['max_epochs', 'early_stopping']


@dataclass(frozen=True)
class KohonenMetrics:
    quantization_error: float
    topographic_error: float


@dataclass(frozen=True)
class OjaMetrics:
    pc1_alignment: float
    pc1_score_correlation: float
    captured_variance: float
    explained_variance_ratio: float
    pc1_variance: float


@dataclass(frozen=True)
class KohonenEpoch:
    epoch: int
    updates: int
    quantization_error: float
    topographic_error: float
    weight_change: float
    learning_rate: float
    radius: float


@dataclass(frozen=True)
class OjaEpoch:
    epoch: int
    updates: int
    direction_change: float
    norm_before_normalization: float
    learning_rate: float
    captured_variance: float
    pc1_alignment: float = 0.0
    explained_variance_ratio: float = 0.0


@dataclass(frozen=True)
class RunMetrics:
    evaluation: KohonenMetrics | OjaMetrics
    epochs: int
    updates: int
    stop_reason: StopReason
    features: int
    selected_epoch: int | None = None
    best_epoch: int | None = None

    def to_dict(self) -> dict:
        """Mantiene el formato plano de metrics.json y runs.csv."""
        values = {
            **asdict(self.evaluation),
            'epochs': self.epochs,
            'updates': self.updates,
            'stop_reason': self.stop_reason,
            'features': self.features,
        }
        if self.selected_epoch is not None:
            values['selected_epoch'] = self.selected_epoch
        if self.best_epoch is not None:
            values['best_epoch'] = self.best_epoch
        return values
