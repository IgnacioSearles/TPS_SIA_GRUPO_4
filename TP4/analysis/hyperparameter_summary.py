"""Barridos de tasa de aprendizaje y radio en 4×4: tablas agregadas y figura, sin datos de creadores.

Uso desde TP4: python3 -m analysis.hyperparameter_summary
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .kohonen_experiments import compare

SWEEPS = {'02_learning': 'aprendizaje', '03_neighborhood': 'vecindad'}
COLUMNS = {
    'model.topology': 'topology', 'model.learning_rate_start': 'learning_rate_start',
    'model.learning_rate_end': 'learning_rate_end', 'model.decay': 'decay', 'model.neighborhood': 'neighborhood',
    'model.radius_start': 'radius_start', 'model.radius_end': 'radius_end', 'model.radius_decay': 'radius_decay',
    'successful_seeds': 'seeds', 'quantization_error_mean': 'QE_mean', 'quantization_error_std': 'QE_std',
    'topographic_error_mean': 'TE_mean', 'topographic_error_std': 'TE_std',
    'empty_fraction_mean': 'empty_fraction_mean', 'ari_mean': 'ARI_mean',
}
# Configuración usada en el modelo de interpretación (analysis/som_campanas.py).
CHOSEN = {'topology': 'rectangular', 'learning_rate_start': .3, 'learning_rate_end': .01, 'decay': 'exponential',
          'neighborhood': 'gaussian', 'radius_start': 2.0, 'radius_end': .5, 'radius_decay': 'exponential'}


def public_summary(work):
    summary = pd.read_csv(work / 'summary.csv')
    failed = int(summary.failed_runs.sum())
    if failed:
        raise ValueError(f'{failed} corridas fallidas en {work}: revisar antes de resumir')
    public = summary[list(COLUMNS)].rename(columns=COLUMNS)
    public['chosen'] = np.logical_and.reduce([public[key] == value for key, value in CHOSEN.items()])
    return public.sort_values(['topology', 'QE_mean']).reset_index(drop=True)


def radius_family(row):
    if row.radius_decay == 'constant':
        return f'radio constante {row.radius_start:g}'
    return f'radio final {row.radius_end:g}'


def plot_sweeps(tables, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    learning, neighborhood = tables['aprendizaje'], tables['vecindad']
    for marker, (topology, group) in zip('os', learning.groupby('topology')):
        constant = group.decay == 'constant'
        axes[0].scatter(group.QE_mean[~constant], group.TE_mean[~constant], marker=marker, color='C0',
                        label=f'{topology}, con decaimiento')
        axes[0].scatter(group.QE_mean[constant], group.TE_mean[constant], marker=marker, color='C1',
                        label=f'{topology}, tasa constante')
    axes[0].set(title='Tasa de aprendizaje (radio 2 → 0,5)', xlabel='QE medio', ylabel='TE medio')

    families = neighborhood.assign(family=neighborhood.apply(radius_family, axis=1))
    for color, (family, group) in zip((f'C{index}' for index in range(10)), families.groupby('family')):
        for marker, (kernel, subset) in zip('o^', group.groupby('neighborhood')):
            axes[1].scatter(subset.QE_mean, subset.TE_mean, color=color, marker=marker, s=25,
                            label=f'{family}, {kernel}')
    axes[1].set(title='Radio y vecindad (tasa 0,3 → 0,01)', xlabel='QE medio', ylabel='TE medio')

    for ax, table in zip(axes, (learning, neighborhood)):
        chosen = table[table.chosen]
        ax.scatter(chosen.QE_mean, chosen.TE_mean, s=220, facecolors='none', edgecolors='black', lw=1.5,
                   label='elegida')
        ax.legend(fontsize=7)
    fig.suptitle('Kohonen 4×4: cada punto es una configuración (media de 5 semillas)')
    fig.tight_layout()
    fig.savefig(output / 'comparacion.png', dpi=160)
    plt.close(fig)


def build(root, output, work):
    output, work = Path(output), Path(work)
    output.mkdir(parents=True, exist_ok=True)
    tables = {}
    for sweep, name in SWEEPS.items():
        table = public_summary(compare(Path(root) / sweep, work / sweep))
        table.to_csv(output / f'{name}.csv', index=False, float_format='%.4g')
        tables[name] = table
    plot_sweeps(tables, output)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/kohonen_suite_1000')
    parser.add_argument('--output', default='reports/ejercicio1/aprendizaje_vecindad_kohonen')
    parser.add_argument('--work', default='results/sweep_analysis')
    args = parser.parse_args()
    build(args.root, args.output, args.work)


if __name__ == '__main__':
    main()
