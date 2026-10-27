"""Interpretación de Kohonen para las campañas: regiones del mapa, estabilidad, planos y selección.

Uso desde TP4: python3 -m analysis.som_campanas
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.tp4.config import KohonenConfig
from src.tp4.kohonen import Kohonen
from src.tp4.pca import compute_pca
from src.tp4.preprocessing import Preprocessor
from .eda import TOPICS, cost_per_thousand_argentine_reach, main_topic
from .representation_summary import eta_squared, topic_purity
from .som_regions import contiguous_ward, mean_pairwise_ari, membership_firmness

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ejercicio1' / 'som'
CAMPAIGN_OUT = ROOT / 'reports' / 'ejercicio1' / 'campanas'

# Mismo protocolo que los barridos de configs/kohonen_suite_1000: 1.000 épocas, decaimiento en las primeras 500.
MODEL = KohonenConfig(rows=4, cols=4, epochs=1000, schedule_epochs=500, seed=0)
SEEDS = range(5)
REGION_COUNT = 5
REGION_COUNT_CANDIDATES = range(2, 9)
TOP = 3

PROFILE_COLUMNS = ['followers', 'price_usd', 'cpm_arg', 'argentina_audience_pct', 'non_follower_reach_pct',
                   'content_tutorial_pct', 'saves_per_1000', 'profile_visits_per_1000', 'sponsored_posts_pct']
COMPARISON_COLUMNS = ['followers', 'price_usd', 'cpm_arg', 'argentina_audience_pct', 'non_follower_reach_pct',
                      'content_tutorial_pct', 'main_topic_share']
# Variables de entrada cuyo prototipo se muestra como plano de componentes, en unidades originales.
COMPONENT_PLANES = ['followers', 'price_usd', 'argentina_audience_pct', 'non_follower_reach_pct',
                    'content_tutorial_pct', 'saves_per_1000', 'sponsored_posts_pct', 'average_caption_words',
                    'topic_football_pct', 'topic_makeup_pct', 'topic_dogs_pct', 'topic_art_pct']
# Criterios de un solo indicador: (columna, mayor es mejor). Profundidad educativa = % de tutoriales.
CRITERIA = {
    'followers': ('followers', True),
    'costo_alcance_ar': ('cpm_arg', False),
    'descubrimiento': ('non_follower_reach_pct', True),
    'profundidad': ('content_tutorial_pct', True),
}
# Región elegida para cada cliente según el objetivo (ver reports/ejercicio1/som/README.md).
# Las etiquetas R1–R5 ordenan las regiones por mediana de seguidores; revisar si cambia el modelo.
TARGET_REGIONS = {'football': ['R4', 'R5'], 'makeup': ['R4', 'R5'], 'dogs': ['R1'], 'art': ['R3']}


@dataclass(frozen=True)
class CatalogMap:
    """Modelo de referencia y regiones de cada creador, con su estabilidad entre semillas."""
    som: Kohonen
    winners: np.ndarray
    neuron_regions: np.ndarray
    region_names: list[str]
    creator_regions: np.ndarray
    firmness: np.ndarray


def prepare(frame):
    preprocessor = Preprocessor()
    x = preprocessor.fit_transform(frame)
    pca = compute_pca(x)
    profiled = frame.assign(
        cpm_arg=cost_per_thousand_argentine_reach(frame), main_topic=main_topic(frame),
        pc1=pca.scores[:, 0], pc2=pca.scores[:, 1], pc3=pca.scores[:, 2])
    profiled['main_topic_share'] = [row[f'topic_{row.main_topic}_pct'] for _, row in profiled.iterrows()]
    return x, preprocessor, profiled


def train_maps(x):
    return [Kohonen(replace(MODEL, seed=seed)).fit(x) for seed in SEEDS]


def creator_regions(som, x, region_count):
    return contiguous_ward(som.weights_, som.grid.adjacency, region_count)[som.predict(x)]


def region_count_table(soms, x, topics):
    """Para cada cantidad de regiones: estabilidad entre semillas, pureza de tema y tamaños."""
    rows = []
    for count in REGION_COUNT_CANDIDATES:
        assignments = [creator_regions(som, x, count) for som in soms]
        sizes = [np.bincount(regions, minlength=count) for regions in assignments]
        rows.append({
            'regions': count, 'ari_between_seeds': mean_pairwise_ari(assignments),
            'topic_purity': float(np.mean([topic_purity(regions, topics) for regions in assignments])),
            'smallest_region': float(np.mean([size.min() for size in sizes])),
            'largest_region': float(np.mean([size.max() for size in sizes])),
        })
    return pd.DataFrame(rows)


def random_purity_by_count(soms, x, topics, permutations=200, seed=0):
    """Pureza de tema que darían regiones del mismo tamaño armadas al azar, para cada cantidad de regiones."""
    rng = np.random.default_rng(seed)
    baseline = {}
    for count in REGION_COUNT_CANDIDATES:
        assignments = [creator_regions(som, x, count) for som in soms]
        baseline[count] = float(np.mean([topic_purity(rng.permutation(regions), topics)
                                         for regions in assignments for _ in range(permutations)]))
    return pd.Series(baseline, name='random_purity')


def catalog_map(soms, x, followers):
    reference = soms[0]
    labels = contiguous_ward(reference.weights_, reference.grid.adjacency, REGION_COUNT)
    winners = reference.predict(x)
    labels = _order_by_median(labels, winners, followers)
    creators = labels[winners]
    firmness = membership_firmness(creators, [creator_regions(som, x, REGION_COUNT) for som in soms])
    names = [f'R{index + 1}' for index in range(REGION_COUNT)]
    return CatalogMap(reference, winners, labels, names, creators, firmness)


def _order_by_median(neuron_labels, winners, followers):
    """Renumera las regiones de menor a mayor mediana de seguidores, para que R1–R5 no dependan de la semilla."""
    medians = [np.median(followers[neuron_labels[winners] == label]) for label in range(neuron_labels.max() + 1)]
    rank = np.argsort(np.argsort(medians))
    return rank[neuron_labels]


def with_regions(frame, catalog: CatalogMap):
    som = catalog.som
    return frame.assign(
        region=[catalog.region_names[label] for label in catalog.creator_regions],
        neuron=['{},{}'.format(*som.grid.indices[winner]) for winner in catalog.winners],
        firmness=catalog.firmness)


def region_profiles(frame):
    grouped = frame.groupby('region')
    topics = grouped.main_topic.value_counts().unstack(fill_value=0).reindex(columns=TOPICS, fill_value=0)
    return pd.concat([
        grouped.size().rename('creators'), topics, grouped[PROFILE_COLUMNS].median(),
        grouped[['pc1', 'pc2', 'firmness']].mean(),
    ], axis=1)


def region_separation(frame):
    """η² de cada variable del caso de uso por región, y el valor esperado si las regiones fueran al azar."""
    values = {
        'log(seguidores)': np.log(frame.followers), 'Alcance a no seguidores': frame.non_follower_reach_pct,
        'Tutoriales': frame.content_tutorial_pct, 'Publicidad': frame.sponsored_posts_pct,
        'Audiencia argentina': frame.argentina_audience_pct, 'log(USD / 1.000 AR)': np.log(frame.cpm_arg),
    }
    separation = pd.Series({name: eta_squared(frame.region, column) for name, column in values.items()})
    # Con k grupos al azar entre n creadores, E[η²] = (k − 1) / (n − 1).
    random_baseline = (frame.region.nunique() - 1) / (len(frame) - 1)
    return separation, random_baseline


@dataclass(frozen=True)
class TopicAssociation:
    """Cruce región × tema, con pureza y V de Cramér frente a regiones del mismo tamaño asignadas al azar."""
    counts: pd.DataFrame
    purity: float
    cramers_v: float
    random_purity: float
    random_cramers_v: float


def region_topic_association(frame, permutations=1000, seed=0):
    regions, topics = frame.region.to_numpy(), frame.main_topic.to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(regions) for _ in range(permutations)]
    return TopicAssociation(
        counts=pd.crosstab(frame.region, frame.main_topic).reindex(columns=TOPICS, fill_value=0),
        purity=topic_purity(regions, topics), cramers_v=cramers_v(regions, topics),
        random_purity=float(np.mean([topic_purity(sample, topics) for sample in shuffled])),
        random_cramers_v=float(np.mean([cramers_v(sample, topics) for sample in shuffled])))


def cramers_v(first, second):
    """Asociación entre dos variables categóricas: 0 si son independientes, 1 si una determina a la otra."""
    table = pd.crosstab(np.asarray(first), np.asarray(second)).to_numpy()
    if min(table.shape) < 2:
        raise ValueError('V de Cramér requiere al menos dos categorías en cada variable')
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    chi_squared = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi_squared / (table.sum() * (min(table.shape) - 1))))


def topic_region_options(frame):
    """Qué tipos de creador hay para cada cliente: candidatos de su tema, separados por región."""
    grouped = frame.groupby(['main_topic', 'region'])
    return pd.concat([grouped.size().rename('candidates'), grouped[COMPARISON_COLUMNS].median()], axis=1) \
        .reindex(TOPICS, level=0)


def criterion_selections(frame):
    """Top 3 por tema según cada criterio de un solo indicador, con la región de cada creador."""
    rows = []
    for topic in TOPICS:
        candidates = frame[frame.main_topic == topic]
        for criterion, (column, higher) in CRITERIA.items():
            chosen = candidates.sort_values(column, ascending=not higher).head(TOP)
            rows.append(chosen.assign(topic=topic, criterion=criterion))
    return pd.concat(rows)[['topic', 'criterion', 'name', 'region', *PROFILE_COLUMNS]]


def strategy_selections(frame):
    """Top 3 por tema con cada estrategia. SOM = región del objetivo y, dentro, menor costo por cuenta argentina."""
    rows = []
    for topic in TOPICS:
        candidates = frame[frame.main_topic == topic]
        target = candidates[candidates.region.isin(TARGET_REGIONS[topic])]
        strategies = {
            'seguidores': candidates.nlargest(TOP, 'followers'),
            'oja': candidates.nlargest(TOP, 'pc1'),
            'costo_alcance_ar': candidates.nsmallest(TOP, 'cpm_arg'),
            'som': target.nsmallest(TOP, 'cpm_arg'),
        }
        for strategy, chosen in strategies.items():
            rows.append(chosen.assign(topic=topic, strategy=strategy))
    return pd.concat(rows)[['topic', 'strategy', 'name', 'region', 'neuron', *COMPARISON_COLUMNS]]


def strategy_summary(selections):
    return selections.groupby(['topic', 'strategy'], sort=False)[COMPARISON_COLUMNS].mean() \
        .assign(regions=selections.groupby(['topic', 'strategy'], sort=False).region.agg(lambda r: ', '.join(sorted(set(r)))))


def substitutes(frame, x, catalog: CatalogMap, chosen, count=2):
    """Reemplazos del mismo tema para cada elegido, fuera de la selección: los más cercanos en la grilla y,
    a igual distancia, en el espacio de entrada."""
    grid_distances = catalog.som.grid.distances
    rows = []
    for _, creator in chosen.iterrows():
        position = frame.index[frame.name == creator['name']][0]
        same_topic = frame.index[(frame.main_topic == creator.topic) & ~frame.name.isin(chosen.name)]
        order = sorted(same_topic, key=lambda other: (grid_distances[catalog.winners[position], catalog.winners[other]],
                                                      np.linalg.norm(x[position] - x[other])))
        for other in order[:count]:
            rows.append({'chosen': creator['name'], 'substitute': frame.at[other, 'name'],
                         'grid_distance': grid_distances[catalog.winners[position], catalog.winners[other]],
                         **frame.loc[other, ['region', 'cpm_arg', 'argentina_audience_pct', 'followers']].to_dict()})
    return pd.DataFrame(rows)


def neuron_planes(catalog: CatalogMap, preprocessor):
    """Prototipos en unidades originales: invierte el z-score de las variables de entrada."""
    columns = preprocessor.columns_
    original = catalog.som.weights_ * preprocessor.scale_ + preprocessor.mean_
    return {column: original[:, columns.index(column)] for column in COMPONENT_PLANES}


def frontiers(columns, som, top=5):
    """Pares de neuronas vecinas más distantes y las variables que más los separan."""
    first, second = np.where(np.triu(som.grid.adjacency, 1))
    gaps = np.linalg.norm(som.weights_[first] - som.weights_[second], axis=1)
    rows = []
    for pair in np.argsort(-gaps)[:top]:
        difference = pd.Series(som.weights_[second[pair]] - som.weights_[first[pair]], index=columns)
        difference = difference.reindex(difference.abs().sort_values(ascending=False).index)
        rows.append({
            'from': '{},{}'.format(*som.grid.indices[first[pair]]),
            'to': '{},{}'.format(*som.grid.indices[second[pair]]),
            'distance': gaps[pair],
            'main_differences': ', '.join(f'{column} {value:+.1f}' for column, value in difference.head(4).items()),
        })
    return pd.DataFrame(rows)


def run(output=OUT, campaign_output=CAMPAIGN_OUT, data=ROOT / 'data' / 'creadores.csv'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, preprocessor, frame = prepare(pd.read_csv(data))
    soms = train_maps(x)
    counts = region_count_table(soms, x, frame.main_topic.to_numpy())
    catalog = catalog_map(soms, x, frame.followers.to_numpy())
    frame = with_regions(frame, catalog)
    metrics = catalog.som.evaluate(x)

    profiles = region_profiles(frame)
    options = topic_region_options(frame)
    selections = strategy_selections(frame)
    summary = strategy_summary(selections)
    replacements = substitutes(frame, x, catalog, selections[selections.strategy == 'som'])
    edges = frontiers(preprocessor.columns_, catalog.som)

    counts.to_csv(output / 'cantidad_regiones.csv', index=False, float_format='%.3f')
    frame[['name', 'main_topic', 'region', 'neuron', 'firmness', *PROFILE_COLUMNS, 'pc1', 'pc2']] \
        .sort_values(['region', 'neuron']).to_csv(output / 'creadores.csv', index=False, float_format='%.4g')
    profiles.to_csv(output / 'regiones.csv', float_format='%.4g')
    options.to_csv(output / 'opciones_por_tema.csv', float_format='%.4g')
    selections.to_csv(output / 'selecciones.csv', index=False, float_format='%.4g')
    criteria = criterion_selections(frame)
    criteria.to_csv(output / 'criterios.csv', index=False, float_format='%.4g')
    summary.to_csv(output / 'estrategias.csv', float_format='%.4g')
    replacements.to_csv(output / 'reemplazos.csv', index=False, float_format='%.4g')
    edges.to_csv(output / 'fronteras.csv', index=False, float_format='%.3f')

    from .som_plots import plot_component_planes, plot_map, plot_regions_3d, plot_regions_on_pca
    plot_map(catalog, frame.main_topic.to_numpy(), output)
    plot_component_planes(catalog.som.grid, neuron_planes(catalog, preprocessor), output)
    plot_regions_on_pca(frame, output)
    plot_regions_3d(frame, output)

    _plot_campaigns(frame, x, preprocessor, catalog, criteria, selections, Path(campaign_output))
    _print(metrics, counts, profiles, options, summary)
    return frame


def _plot_campaigns(frame, x, preprocessor, catalog, criteria, selections, output):
    from . import campaign_plots as plots
    output.mkdir(parents=True, exist_ok=True)
    chosen = selections[selections.strategy == 'som']
    plots.plot_pca_axes(frame, compute_pca(x), preprocessor.columns_, chosen, output)
    plots.plot_oja_axis(frame, criteria, chosen, output)
    ar_plane = neuron_planes(catalog, preprocessor)['argentina_audience_pct']
    plots.plot_map_targets(catalog, ar_plane, chosen, TARGET_REGIONS, output)
    plots.plot_region_topics(catalog, frame.main_topic.to_numpy(), region_topic_association(frame), output)
    plots.plot_region_separation(*region_separation(frame), output)
    plots.plot_candidates(frame, selections, TARGET_REGIONS, output)
    plots.plot_against_followers(selections, output)


def _print(metrics, counts, profiles, options, summary):
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(f'Kohonen {MODEL.rows}x{MODEL.cols} {MODEL.topology}: '
              f'QE={metrics.quantization_error:.3f} TE={metrics.topographic_error:.2f}')
        print(counts.round(2).to_string(index=False))
        print(profiles.round(1).to_string())
        print(options.round(1).to_string())
        print(summary.round(1).to_string())


if __name__ == '__main__':
    run()
