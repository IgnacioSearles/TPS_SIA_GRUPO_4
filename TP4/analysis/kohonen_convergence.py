"""Diagnóstico de meseta sobre corridas terminadas; no cambia el entrenamiento."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


def relative_improvement(before, after):
    return 100 * (before - after) / max(abs(before), np.finfo(float).eps)


def history_diagnostics(history, window=50, improvement_threshold=.5):
    if len(history) < 2 * window:
        raise ValueError('Se requieren al menos dos ventanas completas')
    qe = history.quantization_error
    previous, last = history.iloc[-2 * window:-window], history.iloc[-window:]
    gain = relative_improvement(previous.quantization_error.mean(), last.quantization_error.mean())
    span = 100 * (last.quantization_error.max() - last.quantization_error.min()) / last.quantization_error.mean()
    te_change = last.topographic_error.mean() - previous.topographic_error.mean()
    return {
        'epochs': len(history), 'qe_final': float(qe.iloc[-1]),
        'qe_half': float(qe.iloc[len(history) // 2 - 1]),
        'gain_second_half_pct': float(relative_improvement(qe.iloc[len(history) // 2 - 1], qe.iloc[-1])),
        'gain_last_window_pct': float(relative_improvement(qe.iloc[-window - 1], qe.iloc[-1])),
        'gain_last_two_windows_pct': float(relative_improvement(qe.iloc[-2 * window - 1], qe.iloc[-1])) if len(history) > 2 * window else None,
        'smoothed_gain_pct': float(gain), 'tail_span_pct': float(span),
        'te_final': float(history.topographic_error.iloc[-1]),
        'te_window_change': float(te_change),
        'final_weight_change': float(history.weight_change.iloc[-1]),
        'best_epoch': int(history.epoch.iloc[qe.to_numpy().argmin()]),
        'near_plateau': bool(abs(gain) < improvement_threshold and span < 1 and abs(te_change) <= .02),
    }


def collect(root, epochs, window, threshold):
    rows, histories, seen = [], {}, set()
    for marker in sorted(Path(root).rglob('record.json')):
        record = json.loads(marker.read_text())
        if record['failure'] is not None:
            continue
        config = json.loads((marker.parent / 'config.json').read_text())
        if config['algorithm'] != 'kohonen' or config['model']['epochs'] != epochs:
            continue
        key = (record['group'], record['seed'])
        if key in seen:
            continue
        history = pd.read_csv(marker.parent / 'history.csv')
        if len(history) != epochs:
            continue
        model = config['model']
        row = {'run_directory': str(marker.parent), 'group': record['group'], 'seed': record['seed'],
               'rows': model['rows'], 'cols': model['cols'], 'topology': model['topology'],
               'initialization': model['initialization'], **history_diagnostics(history, window, threshold)}
        rows.append(row)
        histories.setdefault(record['group'], []).append(history)
        seen.add(key)
    if not rows:
        raise ValueError(f'No hay corridas completas de {epochs} épocas en {root}')
    return pd.DataFrame(rows), histories


def summarize(rows):
    return rows.groupby(['group', 'rows', 'cols', 'topology', 'initialization']).agg(
        runs=('seed', 'size'), qe_mean=('qe_final', 'mean'), qe_std=('qe_final', 'std'),
        gain_second_half_pct=('gain_second_half_pct', 'mean'),
        gain_last_window_pct=('gain_last_window_pct', 'mean'),
        gain_last_two_windows_pct=('gain_last_two_windows_pct', 'mean'),
        smoothed_gain_pct=('smoothed_gain_pct', 'mean'), te_mean=('te_final', 'mean'),
        best_at_end=('best_epoch', lambda values: (values == rows.epochs.iloc[0]).sum()),
        near_plateau=('near_plateau', 'sum'),
    ).reset_index()


def plot_histories(histories, rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    try:
        for group, items in histories.items():
            meta = rows[rows.group == group].iloc[0]
            label = f'{meta.rows}×{meta.cols} {meta.topology} {meta.initialization} (n={len(items)})'
            for ax, field in zip(axes, ['quantization_error', 'topographic_error']):
                values = np.array([item[field] for item in items])
                mean, spread = values.mean(axis=0), values.std(axis=0)
                ax.plot(items[0].epoch, mean, label=label)
                ax.fill_between(items[0].epoch, mean - spread, mean + spread, alpha=.15)
                ax.set(xlabel='Época', title=field)
                ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / 'curves.png', dpi=150)
    finally:
        plt.close(fig)


def measure(root, output, epochs=500, window=50, threshold=.5):
    rows, histories = collect(root, epochs, window, threshold)
    summary = summarize(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / 'runs.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    snapshot = {'measured_at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
                'epochs': epochs, 'window': window, 'smoothed_improvement_threshold_pct': threshold,
                'complete_unique_runs': len(rows), 'best_at_last_epoch': int((rows.best_epoch == epochs).sum()),
                'near_plateau': int(rows.near_plateau.sum()),
                'criterion': 'abs(smoothed gain) < threshold; tail span < 1%; abs(TE window change) <= 0.02',
                'limitation': 'Diagnóstico de la cola bajo un decaimiento activo; no demuestra que entrenar más no ayude.'}
    (output / 'snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    plot_histories(histories, rows, output)
    print(json.dumps(snapshot, indent=2))
    print(summary.drop(columns='group').to_string(index=False))
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/kohonen_suite')
    parser.add_argument('--output', default='results/kohonen_convergence')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    if args.window < 1 or args.epochs < 2 * args.window or args.threshold <= 0:
        parser.error('Se requiere window >= 1, epochs >= 2*window y threshold > 0')
    measure(args.root, args.output, args.epochs, args.window, args.threshold)


if __name__ == '__main__':
    main()
