"""Validaciones y tablas del EDA; no dibuja ni escribe archivos."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

TOPICS = ['football', 'makeup', 'dogs', 'art']
# Variables que los clientes necesitan comparar entre temas; cpm_arg se calcula aparte.
TOPIC_PROFILE_COLUMNS = [
    'followers', 'price_usd', 'cpm_arg', 'argentina_audience_pct', 'non_follower_reach_pct',
    'content_tutorial_pct', 'saves_per_1000', 'profile_visits_per_1000', 'average_watch_time_seconds',
]
# Variables que usa la sección de campañas para discutir la selección por seguidores.
FOLLOWER_RELATION_TARGETS = [
    'price_usd', 'cpm_arg', 'argentina_audience_pct',
    'non_follower_reach_pct', 'content_tutorial_pct', 'saves_per_1000',
]


@dataclass(frozen=True)
class TopicTables:
    counts: pd.Series
    specialization: pd.DataFrame
    medians: pd.DataFrame


@dataclass(frozen=True)
class VariableBlocks:
    ordered_correlations: pd.DataFrame
    membership: pd.Series

    def summary(self) -> pd.DataFrame:
        rows = []
        for block, members in self.membership.groupby(self.membership, sort=False):
            names = list(members.index)
            inner = self.ordered_correlations.loc[names, names].to_numpy()
            pairs = inner[np.triu_indices(len(names), k=1)]
            rows.append({'bloque': block, 'variables': len(names),
                         'correlacion_media': pairs.mean() if len(pairs) else np.nan,
                         'miembros': ', '.join(names)})
        return pd.DataFrame(rows)


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


def main_topic(frame: pd.DataFrame) -> pd.Series:
    """Tema con mayor porcentaje de publicaciones de cada creador."""
    shares = frame[[f'topic_{topic}_pct' for topic in TOPICS]]
    return shares.idxmax(axis=1).str.removeprefix('topic_').str.removesuffix('_pct').rename('main_topic')


def cost_per_thousand_argentine_reach(frame: pd.DataFrame) -> pd.Series:
    """USD de una publicación por cada 1.000 cuentas alcanzadas en Argentina."""
    return (frame.price_usd / frame.estimated_argentina_reach * 1000).rename('cpm_arg')


def topic_tables(frame: pd.DataFrame) -> TopicTables:
    topics = main_topic(frame)
    shares = frame[[f'topic_{topic}_pct' for topic in TOPICS]].to_numpy()
    ranked = np.sort(shares, axis=1)
    specialization = pd.DataFrame({
        'creator_id': frame.creator_id, 'main_topic': topics,
        'main_share': ranked[:, -1], 'second_share': ranked[:, -2],
    })
    profiled = frame.assign(cpm_arg=cost_per_thousand_argentine_reach(frame), main_topic=topics)
    counts = topics.value_counts().reindex(TOPICS, fill_value=0).rename('creadores')
    medians = profiled.groupby('main_topic')[TOPIC_PROFILE_COLUMNS].median().reindex(TOPICS)
    return TopicTables(counts=counts, specialization=specialization, medians=medians.assign(creadores=counts))


def variable_blocks(numeric: pd.DataFrame, min_mean_correlation: float = .3) -> VariableBlocks:
    """Agrupa variables por correlación con signo (distancia 1 − r, enlace promedio).

    Se usa el signo para que dos variables opuestas queden en bloques distintos:
    con 1 − |r| alcance y profundidad caerían juntos por ser contrarios.
    """
    correlations = numeric.corr()
    distances = squareform(1 - correlations.to_numpy(), checks=False).clip(min=0)
    tree = linkage(distances, method='average', optimal_ordering=True)
    labels = fcluster(tree, t=1 - min_mean_correlation, criterion='distance')
    order = correlations.columns[leaves_list(tree)]
    membership = pd.Series(labels, index=correlations.columns).loc[order]
    renumbered = membership.map({label: index + 1 for index, label in enumerate(membership.unique())})
    return VariableBlocks(correlations.loc[order, order], renumbered.rename('bloque'))


def follower_relations(frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Pearson y Spearman de seguidores contra cada variable; Spearman resiste la asimetría."""
    profiled = frame.assign(cpm_arg=cost_per_thousand_argentine_reach(frame))
    return pd.DataFrame({
        'pearson': [profiled.followers.corr(profiled[target]) for target in targets],
        'spearman': [profiled.followers.corr(profiled[target], method='spearman') for target in targets],
    }, index=pd.Index(targets, name='variable'))


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
