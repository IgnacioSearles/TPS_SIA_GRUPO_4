"""Consolida augmentation y congela una selección mediante criterios predeclarados."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .kohonen_experiments import adjusted_rand_index


def summarize(output):
    protocol = json.loads((output/'protocol.json').read_text())
    if protocol['smoke']:
        raise ValueError('La prueba inicial no constituye un experimento completo')
    runs = [json.loads(s) for s in (output/'runs.jsonl').read_text().splitlines()]
    canonical = lambda task: json.dumps(task, sort_keys=True)
    if sorted(canonical(r['task']) for r in runs) != sorted(canonical(t) for t in protocol['tasks']):
        raise ValueError('Las tareas no coinciden exactamente con el protocolo')
    if any(r['status'] != 'ok' for r in runs):
        raise ValueError('Hay tareas fallidas')
    strengths = protocol['strengths']
    if any([r['strength'] for r in run['conditions']] != strengths for run in runs):
        raise ValueError('Condiciones incompletas o fuera de orden')
    records = []
    for run in runs:
        baseline = run['conditions'][0]
        for condition in run['conditions']:
            row = dict(sample=run['task']['sample'], seed=run['task']['seed'], strength=condition['strength'],
                train_qe=condition['train_metrics']['quantization_error'],
                excluded_qe=condition['excluded_metrics']['quantization_error'],
                excluded_te=condition['excluded_metrics']['topographic_error'],
                train_te=condition['train_metrics']['topographic_error'],
                ari_reference=condition['ari_reference'], ari_paired_baseline=condition['ari_paired_baseline'],
                mean_displacement_z=condition['mean_displacement_z'], occupied_neurons=condition['occupied_neurons'],
                qe_difference=condition['excluded_metrics']['quantization_error']-baseline['excluded_metrics']['quantization_error'],
                te_difference=condition['excluded_metrics']['topographic_error']-baseline['excluded_metrics']['topographic_error'])
            for topic in ('football', 'makeup', 'dogs', 'art'):
                row[topic+'_overlap'] = condition['selection_overlap'][topic]
                row[topic+'_overlap_difference'] = condition['selection_overlap'][topic]-baseline['selection_overlap'][topic]
                row[topic+'_shortlist'] = condition['selection_sizes'][topic] < 3
            row['epochs'], row['updates'] = condition['epochs'], condition['updates']
            if row['epochs'] != 1000 or row['updates'] != 40000:
                raise ValueError('No se respetó el presupuesto fijado')
            records.append(row)
    frame = pd.DataFrame(records)
    if len(frame) != protocol['fits']:
        raise ValueError('Cantidad de entrenamientos distinta del protocolo')
    frame.to_csv(output/'runs.csv', index=False)
    numeric = frame.select_dtypes(include='number').columns.drop(['sample', 'seed', 'strength'])
    splits = frame.groupby(['sample', 'strength'])[numeric].mean().reset_index()
    splits.to_csv(output/'splits.csv', index=False)
    baseline_qe = splits.loc[splits.strength == 0, 'excluded_qe'].mean()
    gates = protocol['gates']
    rows, campaign_rows = [], []
    for strength in strengths:
        group = splits[splits.strength == strength]
        part = frame[frame.strength == strength]
        pair_aris = []
        for sample in sorted(group['sample'].unique()):
            labels = [c['labels'] for run in runs if run['task']['sample'] == sample
                      for c in run['conditions'] if c['strength'] == strength]
            pair_aris.extend(adjusted_rand_index(a,b) for a,b in combinations(labels,2))
        row = dict(strength=strength, fits=len(part), split_count=len(group),
            train_qe=group.train_qe.mean(), excluded_qe=group.excluded_qe.mean(),
            excluded_te=group.excluded_te.mean(), train_te=group.train_te.mean(),
            ari_reference=group.ari_reference.mean(), ari_between_seeds=np.mean(pair_aris),
            ari_paired_baseline=group.ari_paired_baseline.mean(),
            mean_displacement_z=group.mean_displacement_z.mean(),
            mean_qe_difference=group.qe_difference.mean(),
            qe_difference_p05=group.qe_difference.quantile(.05), qe_difference_p95=group.qe_difference.quantile(.95),
            relative_qe_improvement=1-group.excluded_qe.mean()/baseline_qe,
            split_win_fraction=(group.qe_difference < 0).mean(),
            mean_te_difference=group.te_difference.mean(),
            mean_art_overlap=group.art_overlap.mean(),
            mean_art_overlap_difference=group.art_overlap_difference.mean())
        row['passes_qe'] = bool(row['relative_qe_improvement'] >= gates['min_relative_qe_improvement'])
        row['passes_wins'] = bool(row['split_win_fraction'] >= gates['min_split_win_fraction'])
        row['passes_te'] = bool(row['mean_te_difference'] <= gates['max_te_increase'])
        row['passes_art'] = bool(row['mean_art_overlap_difference'] >= -gates['max_art_overlap_decrease'])
        row['eligible'] = strength > 0 and all(row[name] for name in ('passes_qe','passes_wins','passes_te','passes_art'))
        rows.append(row)
        for topic in ('football', 'makeup', 'dogs', 'art'):
            campaign_rows.append(dict(strength=strength, topic=topic, mean_overlap=part[topic+'_overlap'].mean(),
                                      all_three_fraction=(part[topic+'_overlap'] == 1).mean(),
                                      fewer_than_three_fraction=part[topic+'_shortlist'].mean()))
    table = pd.DataFrame(rows)
    table.to_csv(output/'summary.csv', index=False)
    pd.DataFrame(campaign_rows).to_csv(output/'campaigns.csv', index=False)
    eligible = [r for r in rows if r['eligible']]
    selected = min(eligible, key=lambda r: (r['excluded_qe'], r['strength'])) if eligible else None
    supported = 'supported_on_new_splits' if protocol['stage'] == 'confirmation' else 'candidate_for_confirmation'
    selection = dict(stage=protocol['stage'], selected_strength=selected['strength'] if selected else None,
                     decision=supported if selected else 'keep_baseline',
                     criteria=gates, data_sha256=protocol['data_sha256'],
                     note='Los cuantiles describen 20 particiones superpuestas de este catálogo; no una población nueva.')
    (output/'selection.json').write_text(json.dumps(selection, indent=2, ensure_ascii=False)+'\n')
    (output/'summary.json').write_text(json.dumps(dict(fits=len(frame), conditions=rows, campaigns=campaign_rows,
        selection=selection), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    nonzero = [s for s in strengths if s > 0]
    axes[0].boxplot([splits.loc[splits.strength == s,'qe_difference'] for s in nonzero],
                    tick_labels=[f'{s:.0%}' for s in nonzero])
    axes[0].axhline(0, color='#bc4539', linestyle='--', linewidth=1)
    axes[0].set(xlabel='Desplazamiento máximo hacia un vecino', ylabel='QE excluidos − QE sin augmentation',
                title='Diferencias pareadas por partición')
    art = [r for r in campaign_rows if r['topic'] == 'art']
    axes[1].bar([f'{r["strength"]:.0%}' for r in art], [r['mean_overlap']*100 for r in art], color='#347a91')
    axes[1].set(xlabel='Desplazamiento máximo hacia un vecino', ylabel='% compartido de los tres elegidos',
                ylim=(0,100), title='Arte: estabilidad descriptiva')
    fig.suptitle('Kohonen con interpolaciones locales — presupuesto idéntico', fontsize=12)
    fig.tight_layout()
    fig.savefig(output/'augmentation.png', dpi=150)
    plt.close(fig)
    print(table[['strength','excluded_qe','relative_qe_improvement','split_win_fraction','mean_te_difference',
                 'mean_art_overlap','eligible']].to_string(index=False))
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    summarize(parser.parse_args().output)
