"""Compara representaciones de Kohonen por lo que separa cada mapa, sin datos de creadores ni rutas locales.

Uso desde TP4: python3 -m analysis.representation_summary
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.tp4.numerics import distances
from src.tp4.preprocessing import DEFAULT_DROP, Preprocessor
from .eda import main_topic
from .kohonen_experiments import compare

# Permutaciones de las asignaciones de referencia para estimar el valor esperado al azar.
BASELINE_PERMUTATIONS = 200
REFERENCE_REPRESENTATION = '27 variables'
REPRESENTATION_ORDER = [
    '27 variables', '31 variables', '24 variables',
    'PCA 2', 'PCA 6', 'PCA 10', 'PCA 13', 'PCA 25',
    'log', 'log + PCA 6', 'log + PCA 10', 'log + PCA 13',
]
VISIBILITY_DROPPED = 'followers'


@dataclass(frozen=True)
class UseCaseTargets:
    """Variables del caso de uso contra las que se mide qué separa cada mapa."""
    topic: np.ndarray
    log_followers: np.ndarray
    argentina_audience: np.ndarray
    tutorials: np.ndarray

    @classmethod
    def from_frame(cls, frame):
        return cls(main_topic(frame).to_numpy(), np.log(frame.followers.to_numpy(dtype=float)),
                   frame.argentina_audience_pct.to_numpy(dtype=float),
                   frame.content_tutorial_pct.to_numpy(dtype=float))


def eta_squared(assignments, values):
    """Fracción de la varianza de values explicada por la neurona ganadora (entre grupos / total)."""
    assignments, values = np.asarray(assignments), np.asarray(values, dtype=float)
    if assignments.shape != values.shape or len(values) < 2:
        raise ValueError('eta² requiere asignaciones y valores de igual longitud, con al menos dos')
    total = ((values - values.mean()) ** 2).sum()
    if total == 0:
        raise ValueError('eta² requiere valores con variación')
    _, groups = np.unique(assignments, return_inverse=True)
    counts = np.bincount(groups)
    means = np.bincount(groups, weights=values) / counts
    return float((counts * (means - values.mean()) ** 2).sum() / total)


def topic_purity(assignments, topics):
    """Fracción de creadores que comparten neurona con la mayoría de su mismo tema principal."""
    table = pd.crosstab(np.asarray(assignments), np.asarray(topics))
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def use_case_metrics(assignments, targets: UseCaseTargets):
    return {
        'topic_purity': topic_purity(assignments, targets.topic),
        'eta2_size': eta_squared(assignments, targets.log_followers),
        'eta2_argentina': eta_squared(assignments, targets.argentina_audience),
        'eta2_tutorials': eta_squared(assignments, targets.tutorials),
    }


def representation_label(preprocessing):
    excluded, logged, components = preprocessing['exclude'], preprocessing['log_columns'], preprocessing['pca_components']
    if not excluded:
        return '31 variables'
    if VISIBILITY_DROPPED in excluded:
        return REPRESENTATION_ORDER[2]
    if sorted(excluded) != sorted(DEFAULT_DROP):
        raise ValueError(f'Representación no prevista: {preprocessing}')
    prefix = 'log' if logged else ''
    if components is None:
        return prefix or REFERENCE_REPRESENTATION
    return f'{prefix} + PCA {components}' if prefix else f'PCA {components}'


def collect_use_case(root, frame):
    targets = UseCaseTargets.from_frame(frame)
    rows = []
    for plan_path in sorted(Path(root).rglob('plan.json')):
        plan = json.loads(plan_path.read_text())
        for index, config in enumerate(plan['configs']):
            directory = plan_path.parent / f'run_{index:04d}'
            assignments = _winners(directory, frame)
            rows.append({
                'representation': representation_label(config['preprocessing']),
                'rows': config['model']['rows'], 'cols': config['model']['cols'],
                'topology': config['model']['topology'], 'seed': config['model']['seed'],
                **json.loads((directory / 'metrics.json').read_text()),
                **use_case_metrics(assignments, targets),
                'assignments': assignments,
            })
    if not rows:
        raise ValueError(f'No hay corridas en {root}')
    return pd.DataFrame(rows), targets


def _winners(directory, frame):
    preprocessing = Preprocessor.from_state(json.loads((directory / 'preprocessing.json').read_text()))
    with np.load(directory / 'model.npz', allow_pickle=False) as arrays:
        weights = arrays['weights']
    return distances(preprocessing.transform(frame), weights).argmin(axis=1)


def random_baseline(runs, targets, rng):
    """Mismas ocupaciones que la representación de referencia, con creadores asignados al azar."""
    reference = runs[runs.representation == REFERENCE_REPRESENTATION]
    rows = []
    for size, group in reference.groupby('rows'):
        samples = [use_case_metrics(rng.permutation(assignments), targets)
                   for assignments in group.assignments for _ in range(BASELINE_PERMUTATIONS)]
        rows.append({'rows': size, **pd.DataFrame(samples).mean().to_dict()})
    return pd.DataFrame(rows)


def summarize(runs, comparison):
    """Promedia semillas y topologías por representación y tamaño; agrega ARI y ocupación del comparador."""
    metrics = ['topic_purity', 'eta2_size', 'eta2_argentina', 'eta2_tutorials', 'topographic_error']
    grouped = runs.groupby(['rows', 'representation'])[metrics].agg(['mean', 'std'])
    grouped.columns = ['_'.join(column) for column in grouped.columns]
    stability = _comparison_by_representation(comparison)
    summary = grouped.join(stability).reset_index()
    summary['representation'] = pd.Categorical(summary.representation, REPRESENTATION_ORDER, ordered=True)
    return summary.sort_values(['rows', 'representation']).reset_index(drop=True)


def _comparison_by_representation(comparison):
    comparison = comparison.assign(representation=[
        representation_label({'exclude': json.loads(row['preprocessing.exclude']),
                               'log_columns': json.loads(row['preprocessing.log_columns']),
                               'pca_components': None if pd.isna(row['preprocessing.pca_components'])
                               else int(row['preprocessing.pca_components'])})
        for _, row in comparison.iterrows()])
    columns = {'ari_mean': 'ARI_mean', 'empty_fraction_mean': 'empty_fraction_mean',
               'common_centroid_distance_mean': 'common_cohesion_mean'}
    return comparison.groupby(['model.rows', 'representation'])[list(columns)].mean() \
        .rename(columns=columns).rename_axis(['rows', 'representation'])


def plot_summary(summary, baseline, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('topic_purity', 'Pureza de tema principal'), ('eta2_argentina', 'η² audiencia argentina'),
              ('eta2_size', 'η² log(seguidores)'), ('topographic_error', 'Error topográfico')]
    labels = REPRESENTATION_ORDER
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    try:
        for offset, (size, group) in zip((-.2, 0, .2), summary.groupby('rows')):
            group = group.set_index('representation').reindex(labels)
            positions = np.arange(len(labels)) + offset
            for ax, (metric, title) in zip(axes.flat, panels):
                ax.errorbar(positions, group[f'{metric}_mean'], yerr=group[f'{metric}_std'],
                            fmt='o', capsize=2, label=f'{size}×{size}')
                if metric in baseline:
                    level = baseline.loc[baseline.rows == size, metric].iloc[0]
                    ax.axhline(level, ls='--', lw=.8, color=ax.lines[-1].get_color())
                ax.set_title(title)
        for ax in axes[1]:
            ax.set_xticks(np.arange(len(labels)), labels, rotation=60, ha='right', fontsize=8)
        axes[0, 0].legend(title='Grilla (punteada: al azar)', fontsize=8)
        fig.suptitle('Representaciones de Kohonen: media y desvío de 10 corridas (2 topologías × 5 semillas)')
        fig.tight_layout()
        fig.savefig(output / 'comparacion.png', dpi=160)
    finally:
        plt.close(fig)


def build(root, output, work, data):
    output, work = Path(output), Path(work)
    comparison = pd.read_csv(compare(root, work) / 'summary.csv')
    frame = pd.read_csv(data)
    runs, targets = collect_use_case(root, frame)
    baseline = random_baseline(runs, targets, np.random.default_rng(0))
    summary = summarize(runs, comparison)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / 'resumen.csv', index=False, float_format='%.4g')
    baseline.to_csv(output / 'azar.csv', index=False, float_format='%.4g')
    plot_summary(summary, baseline, output)
    print(f'{len(runs)} corridas; {summary.representation.nunique()} representaciones. Resultados: {output}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/kohonen_suite_1000/04_representation')
    parser.add_argument('--output', default='reports/ejercicio1/representacion_kohonen')
    parser.add_argument('--work', default='results/representation_analysis')
    parser.add_argument('--data', default='data/creadores.csv')
    args = parser.parse_args()
    build(args.root, args.output, args.work, args.data)


if __name__ == '__main__':
    main()
