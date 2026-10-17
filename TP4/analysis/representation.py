"""Representaciones del catálogo y sus resultados de PCA tipados."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.tp4.pca import PCAResult, compute_pca
from src.tp4.preprocessing import DEFAULT_LOG, Preprocessor


@dataclass(frozen=True)
class PCAAnalysis:
    name: str
    standardized: pd.DataFrame
    means: pd.Series
    scales: pd.Series
    pca: PCAResult

    @property
    def columns(self):
        return self.standardized.columns

    @property
    def component_names(self) -> list[str]:
        return [f'PC{index + 1}' for index in range(len(self.pca.eigenvalues))]

    def summary(self) -> dict:
        ratio = self.pca.explained_variance_ratio
        return {
            'variante': self.name, 'variables': len(self.columns),
            'PC1': ratio[0], 'PC2': ratio[1], 'PC1_PC2': self.pca.cumulative_variance[1],
            **{f'k_{threshold}': self.pca.components_for_variance(threshold / 100) for threshold in (80, 90, 95)},
            'rango': self.pca.rank,
        }


def analyze_pca(frame: pd.DataFrame, name: str) -> PCAAnalysis:
    means, scales = frame.mean(), frame.std(ddof=1)
    standardized = (frame - means) / scales
    result = compute_pca(standardized.to_numpy(), center=False)
    return PCAAnalysis(name, standardized, means, scales, result)


def analyze_variants(frame: pd.DataFrame, numeric: pd.DataFrame) -> list[PCAAnalysis]:
    preprocessor = Preprocessor().fit(frame)
    selected = numeric[preprocessor.columns_]
    logged = selected.copy()
    logged[DEFAULT_LOG] = np.log1p(logged[DEFAULT_LOG])
    return [
        analyze_pca(numeric, 'todas_z'),
        analyze_pca(selected, 'seleccion_z'),
        analyze_pca(logged, 'seleccion_log_z'),
    ]
