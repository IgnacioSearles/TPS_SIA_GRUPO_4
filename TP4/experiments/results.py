"""Resultados tipados de una ejecución y su registro en el experimento."""
from dataclasses import dataclass
from typing import Literal

from src.tp4.results import KohonenMetrics, OjaMetrics, RunMetrics


@dataclass(frozen=True)
class ExecutionResult:
    config: dict
    metrics: RunMetrics


@dataclass(frozen=True)
class RunFailure:
    type: str
    message: str
    traceback: str


@dataclass(frozen=True)
class RunRecord:
    run: str
    group: str
    seed: int
    metrics: RunMetrics | None = None
    failure: RunFailure | None = None

    def __post_init__(self):
        if (self.metrics is None) == (self.failure is None):
            raise ValueError('Una corrida debe tener métricas o un fallo, exclusivamente')

    @property
    def status(self) -> Literal['ok', 'failed']:
        return 'ok' if self.metrics is not None else 'failed'

    def to_dict(self) -> dict:
        values = {'run': self.run, 'group': self.group, 'seed': self.seed, 'status': self.status}
        if self.metrics is not None:
            values.update(self.metrics.to_dict())
        else:
            values['error'] = self.failure.message
        return values


def restore_record(state: dict) -> RunRecord:
    """Restaura el formato anidado de record.json, sin inferencias desde CSV."""
    metrics = state['metrics']
    if metrics is not None:
        metrics = dict(metrics)
        evaluation = metrics['evaluation']
        evaluation_type = KohonenMetrics if 'quantization_error' in evaluation else OjaMetrics
        metrics['evaluation'] = evaluation_type(**evaluation)
        metrics = RunMetrics(**metrics)
    failure = RunFailure(**state['failure']) if state['failure'] is not None else None
    return RunRecord(state['run'], state['group'], state['seed'], metrics, failure)
