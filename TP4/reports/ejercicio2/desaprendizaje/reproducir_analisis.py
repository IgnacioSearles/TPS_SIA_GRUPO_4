"""Regenera figura e intervalos de las pruebas completas del 22/10/2026.

Desde TP4: OPENBLAS_NUM_THREADS=1 python3 reports/ejercicio2/desaprendizaje/reproducir_analisis.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
RUNS = ROOT / 'results/hopfield_unlearning'


def main(published=False):
    source = REPORT / 'evidencia' if published else RUNS
    confirmation = source / 'confirmacion_sin_compartidos_20261022'
    damage_directory = source / 'dano_espacial_confirmado_20261022'
    for directory in (confirmation, damage_directory):
        if not (directory / 'COMPLETED').exists():
            raise ValueError(f'Faltan resultados completos: {directory}')
    def read_results(directory):
        path = directory / 'resultados.csv'
        return pd.read_csv(path if path.exists() else directory / 'resultados.csv.gz')

    noise = read_results(confirmation)
    damage = read_results(damage_directory)
    means = noise[noise.noise > 0].groupby(['query_seed', 'method']).correct.mean().unstack('method')
    rng = np.random.default_rng(20261022)
    indices = rng.integers(0, len(means), size=(20000, len(means)))

    def interval(values):
        return [float(v) for v in np.quantile(values[indices].mean(axis=1), [.025, .975])]

    stats = {'bootstrap_repetitions': 20000, 'bootstrap_seed': 20261022,
             'unit': 'query_seed block across all fixed QR, noise levels and training seeds; QR set fixed',
             'accuracy': float(means.unlearning.mean()),
             'accuracy_95_interval': interval(means.unlearning.to_numpy()),
             'delta_hebb': float((means.unlearning - means.hebb).mean()),
             'delta_hebb_95_interval': interval((means.unlearning - means.hebb).to_numpy()),
             'delta_nearest': float((means.unlearning - means.nearest).mean()),
             'delta_nearest_95_interval': interval((means.unlearning - means.nearest).to_numpy()),
             'training_seed_scores': noise[(noise.method == 'unlearning') & (noise.noise > 0)]
                 .groupby('training_seed').correct.mean().to_dict(),
             'counts': noise.groupby('method').size().to_dict()}
    if not published:
        (confirmation / 'estadisticas_confirmacion.json').write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
        noise.groupby(['method', 'noise']).agg(
            correct=('correct', 'sum'), count=('correct', 'size'), accuracy=('correct', 'mean'),
            hamming=('hamming', 'mean'), cycles=('cycle', 'sum'), caps=('capped', 'sum')
        ).to_csv(confirmation / 'comparacion_por_ruido.csv')
        noise[noise.method == 'unlearning'].groupby(['qr', 'noise']).correct.mean().to_csv(
            confirmation / 'comparacion_por_qr.csv')
        damage.groupby(['kind', 'block_side', 'method']).agg(
            correct=('correct', 'mean'), hamming=('hamming', 'mean'),
            cycles=('cycle', 'sum'), caps=('capped', 'sum')).to_csv(damage_directory / 'comparacion.csv')

    styles = {'hebb': ('Hebb sin desaprendizaje', '#2a78d6', 's'),
              'nearest': ('Vecino por Hamming', '#707070', '^'),
              'unlearning': ('Desaprendizaje aleatorio', '#eb6834', 'o')}
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), sharey=True)
    panels = [(noise, 'noise', 'Ruido aleatorio: confirmación', 'Probabilidad de inversión'),
              (damage[damage.kind == 'corner_white'], 'block_side', 'Esquina en blanco', 'Lado del bloque (módulos)'),
              (damage[damage.kind == 'spot_black'], 'block_side', 'Mancha negra', 'Lado del bloque (módulos)')]
    for ax, (table, x, title, xlabel) in zip(axes, panels):
        for method, (label, color, marker) in styles.items():
            series = table[table.method == method].groupby(x).correct.mean()
            ax.plot(series.index, series.values, label=label, color=color, marker=marker,
                    linewidth=1.8, markersize=4)
        ax.set(title=title, xlabel=xlabel, ylim=(-.03, 1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', color='#dddddd', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
        if x == 'noise':
            ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Recuperación exacta')
    fig.suptitle('QR sin módulos compartidos: candidato congelado, tasa 0,1 y 50 intentos')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(REPORT / 'CONFIRMACION.png', dpi=170)
    plt.close(fig)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(REPORT / 'CONFIRMACION.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published', action='store_true', help='Usar sólo la evidencia liviana del repositorio')
    main(parser.parse_args().published)
