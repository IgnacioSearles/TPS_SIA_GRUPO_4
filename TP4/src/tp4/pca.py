"""Única implementación de PCA por SVD, compartida por análisis y modelos."""
from dataclasses import dataclass

import numpy as np

from .numerics import matrix


@dataclass(frozen=True)
class PCAResult:
    mean: np.ndarray
    components: np.ndarray  # Filas = componentes; columnas = variables.
    scores: np.ndarray
    singular_values: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    rank: int

    @property
    def cumulative_variance(self) -> np.ndarray:
        return self.explained_variance_ratio.cumsum()

    @property
    def correlation_loadings(self) -> np.ndarray:
        """Correlaciones variable-componente cuando la entrada tiene desvío uno."""
        return self.components.T * np.sqrt(self.eigenvalues)

    def components_for_variance(self, fraction: float) -> int:
        if not 0 < fraction <= 1:
            raise ValueError('La fracción de varianza debe estar entre 0 y 1')
        cumulative = self.cumulative_variance
        return min(int(np.searchsorted(cumulative, fraction)) + 1, len(cumulative))


def compute_pca(samples, *, center: bool = True) -> PCAResult:
    """PCA sin escalar ni whitening; center=False exige datos ya centrados."""
    samples = matrix(samples)
    if len(samples) < 2:
        raise ValueError('PCA requiere al menos dos observaciones')
    mean = samples.mean(axis=0) if center else np.zeros(samples.shape[1])
    centered = samples - mean if center else samples
    left, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])
    components *= signs[:, None]
    scores = left * singular_values * signs
    eigenvalues = singular_values**2 / (len(samples) - 1)
    if eigenvalues.sum() <= 0:
        raise ValueError('PCA requiere variación no nula')
    tolerance = singular_values[0] * max(samples.shape) * np.finfo(float).eps
    rank = int((singular_values > tolerance).sum())
    return PCAResult(
        mean, components, scores, singular_values, eigenvalues,
        eigenvalues / eigenvalues.sum(), rank,
    )
