"""Validaciones y tablas del EDA; no dibuja ni escribe archivos."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityReport:
    filas: int
    variables_numericas: int
    faltantes: int
    filas_duplicadas: int
    ids_duplicados: int
    nombres_duplicados: int
    error_max_precio_ars: float
    error_max_engagement: float
    error_max_alcance_argentino: float
    error_suma_100_publicidad: float
    error_suma_100_temas: float
    error_suma_100_contenido: float
    porcentajes_fuera_rango: int


@dataclass(frozen=True)
class ExploratoryTables:
    descriptive: pd.DataFrame
    outliers: pd.Series
    correlations: pd.DataFrame
    high_correlations: pd.DataFrame


def validate_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include='number')
    if frame.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Revisar faltantes o valores no finitos antes del PCA')
    if len(frame) < 2 or (numeric.std() == 0).any():
        raise ValueError('Revisar cantidad de filas y variables constantes')
    return numeric


def quality_report(frame: pd.DataFrame, numeric: pd.DataFrame) -> QualityReport:
    engagement = frame[['likes_per_1000', 'comments_per_1000', 'saves_per_1000', 'shares_per_1000']]
    percentages = numeric.filter(regex='_pct$')
    return QualityReport(
        filas=len(frame),
        variables_numericas=len(numeric.columns),
        faltantes=int(frame.isna().sum().sum()),
        filas_duplicadas=int(frame.duplicated().sum()),
        ids_duplicados=int(frame.creator_id.duplicated().sum()),
        nombres_duplicados=int(frame.name.duplicated().sum()),
        error_max_precio_ars=float(abs(frame.price_ars - 1500 * frame.price_usd).max()),
        error_max_engagement=float(abs(frame.total_engagement_per_1000 - engagement.sum(axis=1)).max()),
        error_max_alcance_argentino=float(abs(frame.estimated_argentina_reach - frame.average_reach * frame.argentina_audience_pct / 100).max()),
        error_suma_100_publicidad=_percentage_sum_error(frame[['sponsored_posts_pct', 'organic_posts_pct']]),
        error_suma_100_temas=_percentage_sum_error(frame.filter(regex='^topic_')),
        error_suma_100_contenido=_percentage_sum_error(frame.filter(regex='^content_')),
        porcentajes_fuera_rango=int(((percentages < 0) | (percentages > 100)).sum().sum()),
    )


def _percentage_sum_error(percentages: pd.DataFrame) -> float:
    return float(abs(percentages.sum(axis=1) - 100).max())


def exploratory_tables(numeric: pd.DataFrame, correlation_threshold: float = .85) -> ExploratoryTables:
    correlations = numeric.corr()
    return ExploratoryTables(
        descriptive=numeric.describe().T.assign(skew=numeric.skew()),
        outliers=outlier_counts(numeric),
        correlations=correlations,
        high_correlations=high_correlations(correlations, correlation_threshold),
    )


def outlier_counts(numeric: pd.DataFrame, multiplier: float = 1.5) -> pd.Series:
    lower, upper = numeric.quantile(.25), numeric.quantile(.75)
    spread = upper - lower
    flags = (numeric < lower - multiplier * spread) | (numeric > upper + multiplier * spread)
    return flags.sum().rename('cantidad_fuera_1_5_IQR')


def high_correlations(correlations: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows, columns = np.triu_indices(len(correlations), k=1)
    values = correlations.to_numpy()[rows, columns]
    selected = np.abs(values) >= threshold
    names = correlations.columns.to_numpy()
    return pd.DataFrame({
        'variable_1': names[rows[selected]],
        'variable_2': names[columns[selected]],
        'correlacion': values[selected],
    })
