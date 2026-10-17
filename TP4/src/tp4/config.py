"""Configuraciones explícitas y validación de hiperparámetros."""
from dataclasses import dataclass, field
import math

from .topology import topology_names

@dataclass
class EarlyStopping:
    enabled: bool = False
    min_delta: float = 1e-6
    patience: int = 20
    min_epochs: int = 30

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled debe ser booleano')
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError('min_delta debe ser finito y no negativo')
        for name in ('patience', 'min_epochs'):
            positive_int(name, getattr(self, name))

def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} debe ser entero positivo')

def positive(name, value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} debe ser positivo y finito')

@dataclass
class TrainingConfig:
    seed: int = 0
    epochs: int = 500
    shuffle: bool = True
    learning_rate_start: float = .3
    learning_rate_end: float = .01
    decay: str = 'exponential'
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)

    def __post_init__(self):
        if isinstance(self.early_stopping, dict):
            self.early_stopping = EarlyStopping(**self.early_stopping)
        if not isinstance(self.early_stopping, EarlyStopping):
            raise ValueError('early_stopping debe ser una configuración válida')
        if not isinstance(self.shuffle, bool):
            raise ValueError('shuffle debe ser booleano')
        positive_int('epochs', self.epochs)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed debe ser entero no negativo')
        for name in ('learning_rate_start', 'learning_rate_end'):
            positive(name, getattr(self, name))
        if self.decay not in ('constant', 'linear', 'exponential'):
            raise ValueError('Decaimiento desconocido')

@dataclass
class KohonenConfig(TrainingConfig):
    rows: int = 5
    cols: int = 5
    topology: str = 'rectangular'
    neighborhood: str = 'gaussian'
    initialization: str = 'samples'
    radius_start: float | None = None
    radius_end: float = .5
    radius_decay: str = 'exponential'

    def __post_init__(self):
        super().__post_init__()
        positive_int('rows', self.rows)
        positive_int('cols', self.cols)
        if self.rows * self.cols < 2:
            raise ValueError('Se requieren al menos dos neuronas para error topográfico')
        if self.topology not in topology_names():
            raise ValueError('Topología desconocida')
        if self.neighborhood not in ('gaussian', 'uniform'):
            raise ValueError('Vecindad desconocida')
        if self.initialization not in ('samples', 'uniform'):
            raise ValueError('Inicialización desconocida')
        if self.radius_start is None:
            self.radius_start = max(self.rows, self.cols) / 2
        positive('radius_start', self.radius_start)
        positive('radius_end', self.radius_end)
        if self.radius_decay not in ('constant', 'linear', 'exponential'):
            raise ValueError('Decaimiento del radio desconocido')

@dataclass
class OjaConfig(TrainingConfig):
    epochs: int = 1000
    learning_rate_start: float = .001
    learning_rate_end: float = .00001
