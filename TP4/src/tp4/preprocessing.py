"""Selección, logaritmos, estandarización y PCA reutilizables."""
from dataclasses import dataclass, field
import numpy as np

from .pca import compute_pca

DEFAULT_DROP = ['price_ars', 'organic_posts_pct', 'total_engagement_per_1000', 'estimated_argentina_reach']
DEFAULT_LOG = ['followers', 'average_reach', 'average_impressions', 'average_video_views', 'price_usd']

@dataclass
class PreprocessingConfig:
    include: list[str] | None = None
    exclude: list[str] = field(default_factory=lambda: DEFAULT_DROP.copy())
    log_columns: list[str] = field(default_factory=list)
    pca_components: int | None = None

    def __post_init__(self):
        for name in ('include', 'exclude', 'log_columns'):
            value = getattr(self, name)
            if name == 'include' and value is None:
                continue
            if (not isinstance(value, list) or not all(isinstance(c, str) for c in value)
                    or len(set(value)) != len(value)):
                raise ValueError(f'{name} debe ser una lista de nombres únicos')
        k = self.pca_components
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            raise ValueError('pca_components debe ser un entero positivo o null')

class Preprocessor:
    def __init__(self, config=None):
        self.config = config or PreprocessingConfig()

    def _values(self, frame):
        missing = set(self.columns_) - set(frame.columns)
        if missing:
            raise ValueError(f'Faltan columnas: {sorted(missing)}')
        values = frame[self.columns_].to_numpy(dtype=float, copy=True)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError('Los datos deben ser no vacíos y finitos')
        for column in self.config.log_columns:
            i = self.columns_.index(column)
            if (values[:, i] <= -1).any():
                raise ValueError(f'log1p fuera de dominio: {column}')
            values[:, i] = np.log1p(values[:, i])
        return values

    def fit(self, frame):
        if not frame.columns.is_unique:
            raise ValueError('Columnas repetidas')
        cfg = self.config
        candidates = cfg.include if cfg.include is not None else list(frame.select_dtypes(include='number').columns)
        self.columns_ = [c for c in candidates if c not in cfg.exclude]
        if not self.columns_ or len(set(self.columns_)) != len(self.columns_):
            raise ValueError('Seleccionar columnas únicas y no vacías')
        if not set(cfg.log_columns).issubset(self.columns_):
            raise ValueError('Las columnas log deben estar seleccionadas')
        values = self._values(frame)
        if len(values) < 2:
            raise ValueError('Se requieren al menos dos filas')
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0, ddof=1)
        if (self.scale_ == 0).any():
            raise ValueError('Variables constantes: revisar selección')
        z = (values-self.mean_)/self.scale_
        k = cfg.pca_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        if k is not None:
            self._fit_pca(z, k)
        return self

    def _fit_pca(self, standardized, component_count):
        result = compute_pca(standardized, center=False)
        if component_count > result.rank:
            raise ValueError('pca_components debe estar entre 1 y el rango de los datos')
        self.components_ = result.components[:component_count]
        self.explained_variance_ratio_ = result.explained_variance_ratio[:component_count]

    def transform(self, frame):
        if not hasattr(self, 'scale_'):
            raise ValueError('Primero ejecutar fit')
        z = (self._values(frame)-self.mean_)/self.scale_
        return z if self.components_ is None else z@self.components_.T

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)

    def state(self):
        if not hasattr(self, 'scale_'):
            raise ValueError('Primero ejecutar fit')
        return {'columns':self.columns_, 'log_columns':self.config.log_columns,
                'mean':self.mean_.tolist(), 'scale':self.scale_.tolist(),
                'components':None if self.components_ is None else self.components_.tolist(),
                'explained_variance_ratio':None if self.components_ is None else self.explained_variance_ratio_.tolist()}

    @classmethod
    def from_state(cls, state):
        obj = cls(PreprocessingConfig(include=state['columns'], exclude=[], log_columns=state['log_columns']))
        obj.columns_ = state['columns']
        obj.mean_, obj.scale_ = np.array(state['mean']), np.array(state['scale'])
        obj.components_ = None if state['components'] is None else np.array(state['components'])
        obj.explained_variance_ratio_ = (None if state['explained_variance_ratio'] is None
                                         else np.asarray(state['explained_variance_ratio']))
        return obj
