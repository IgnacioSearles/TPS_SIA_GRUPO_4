"""Exporta tablas agregadas y figuras de geometría, sin datos de creadores ni rutas locales."""
import argparse
from pathlib import Path

import pandas as pd

from .kohonen_convergence import collect, summarize
from .kohonen_experiments import compare

COLUMNS = {
    'model.rows': 'rows', 'model.cols': 'cols', 'model.topology': 'topology',
    'model.initialization': 'initialization', 'successful_seeds': 'seeds',
    'model.epochs': 'epochs', 'model.schedule_epochs': 'schedule_epochs',
    'model.learning_rate_start': 'learning_rate_start', 'model.learning_rate_end': 'learning_rate_end',
    'model.radius_start': 'radius_start', 'model.radius_end': 'radius_end',
    'model.neighborhood': 'neighborhood', 'model.decay': 'decay', 'model.radius_decay': 'radius_decay',
    'model.shuffle': 'shuffle', 'quantization_error_mean': 'QE_mean', 'quantization_error_std': 'QE_std',
    'topographic_error_mean': 'TE_mean', 'topographic_error_std': 'TE_std',
    'occupied_neurons_mean': 'occupied_mean', 'empty_fraction_mean': 'empty_fraction_mean',
    'singleton_creator_fraction_mean': 'singleton_fraction_mean',
    'ari_mean': 'ARI_mean', 'ari_std': 'ARI_std', 'ari_min': 'ARI_min',
}


def export_summary(work, output, convergence):
    summary = pd.read_csv(work / 'summary.csv')
    combined = summary.merge(convergence, left_on='config_group', right_on='group',
                             suffixes=('', '_convergence'), validate='one_to_one')
    public = combined[list(COLUMNS)].rename(columns=COLUMNS)
    public['near_plateau_seeds'] = combined.near_plateau
    public['gain_500_1000_pct'] = combined.gain_second_half_pct
    public['tail_smoothed_gain_pct'] = combined.smoothed_gain_pct
    public = public.sort_values(['rows', 'cols', 'topology', 'initialization']).reset_index(drop=True)
    public.to_csv(output / 'resumen.csv', index=False, float_format='%.9g')
    return public


def topology_comparison(summary):
    keys = ['rows', 'cols', 'initialization']
    rectangular = summary[summary.topology == 'rectangular'].set_index(keys).sort_index()
    hexagonal = summary[summary.topology == 'hexagonal'].set_index(keys).sort_index()
    result = pd.DataFrame(index=rectangular.index)
    result['QE_rectangular'] = rectangular.QE_mean
    result['QE_hexagonal'] = hexagonal.QE_mean
    result['QE_hex_change_pct'] = 100 * (hexagonal.QE_mean - rectangular.QE_mean) / rectangular.QE_mean
    result['TE_rectangular'] = rectangular.TE_mean
    result['TE_hexagonal'] = hexagonal.TE_mean
    result['ARI_rectangular'] = rectangular.ARI_mean
    result['ARI_hexagonal'] = hexagonal.ARI_mean
    return result.reset_index()


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [('QE_mean', 'QE_std', 'Error de cuantización'),
               ('TE_mean', 'TE_std', 'Error topográfico'),
               ('empty_fraction_mean', None, 'Fracción de neuronas vacías'),
               ('ARI_mean', None, 'ARI entre semillas (estabilidad de asignaciones)')]
    try:
        for (topology, initialization), group in summary.groupby(['topology', 'initialization']):
            color = 'tab:blue' if topology == 'rectangular' else 'tab:orange'
            marker = 'o' if initialization == 'samples' else 'x'
            counts = group.rows * group.cols
            for ax, (field, error, title) in zip(axes.ravel(), metrics):
                ax.errorbar(counts, group[field], yerr=group[error] if error else None,
                            fmt=marker, color=color, alpha=.7, capsize=2,
                            label=f'{topology}, {initialization}')
                ax.set(title=title, xlabel='Número de neuronas')
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / 'comparacion.png', dpi=170)
    finally:
        plt.close(fig)


def build(root, output, work):
    output, work = Path(output), Path(work)
    compare(root, work)
    rows, _ = collect(root, 1000, 50, .5)
    convergence = summarize(rows)
    rows.to_csv(work / 'convergence_runs.csv', index=False)
    convergence.to_csv(work / 'convergence_summary.csv', index=False)
    output.mkdir(parents=True, exist_ok=True)
    public = export_summary(work, output, convergence)
    topology_comparison(public).to_csv(output / 'topologias.csv', index=False, float_format='%.9g')
    plot_summary(public, output)
    print(f'{len(public)} configuraciones agregadas; {len(rows)} corridas; {rows.near_plateau.sum()} con meseta')
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/kohonen_geometry_1000')
    parser.add_argument('--output', default='reports/ejercicio1/geometria_kohonen')
    parser.add_argument('--work', default='results/kohonen_geometry_analysis')
    args = parser.parse_args()
    build(args.root, args.output, args.work)


if __name__ == '__main__':
    main()
