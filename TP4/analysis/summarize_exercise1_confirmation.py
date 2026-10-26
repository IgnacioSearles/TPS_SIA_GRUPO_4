"""Tablas y figuras descriptivas para confirm_exercise1, sin reentrenar."""
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
from .som_regions import co_assignment, membership_firmness


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95)))


def summarize(output):
    protocol = json.loads((output/'protocol.json').read_text())
    if protocol['smoke']:
        raise ValueError('La prueba de funcionamiento no constituye una confirmación completa')
    reference = json.loads((output/'reference.json').read_text())
    runs = [json.loads(line) for line in (output/'runs.jsonl').read_text().splitlines()]
    if len(runs) != len(protocol['tasks']):
        raise ValueError(f'Faltan corridas: {len(runs)}/{len(protocol["tasks"])}')
    keys = lambda task: json.dumps(task, sort_keys=True)
    if sorted(keys(r['task']) for r in runs) != sorted(keys(t) for t in protocol['tasks']):
        raise ValueError('Las corridas no coinciden con el protocolo')
    ok = [r for r in runs if r['status'] == 'ok']
    report = dict(planned=len(runs), successful=len(ok), failed=len(runs)-len(ok))
    finalist_rows, paired_rows, stable_rows, oja_rows, campaign_rows = [], [], [], [], []
    for candidate in ('rect4x4', 'hex4x4', 'hex2x8'):
        group = [r for r in ok if r['task'].get('candidate') == candidate]
        if not group:
            continue
        row = dict(candidate=candidate, runs=len(group))
        for metric in ('quantization_error', 'topographic_error'):
            row.update({f'{metric}_{stat}': value for stat, value in
                        quantiles([r['train_metrics'][metric] for r in group]).items()})
        row['occupied_neurons_mean'] = np.mean([r['occupied_neurons'] for r in group])
        row['ari_reference_mean'] = np.mean([r['ari_reference'] for r in group])
        pairs = [adjusted_rand_index(a['labels'], b['labels']) for a, b in combinations(group, 2)]
        row['ari_pairwise_mean'] = np.mean(pairs)
        finalist_rows.append(row)
        for other in ('rect4x4',):
            if candidate == other:
                continue
            base = {r['task']['seed']: r for r in ok if r['task'].get('candidate') == other}
            differences = [r['train_metrics']['quantization_error'] - base[r['task']['seed']]['train_metrics']['quantization_error']
                           for r in group if r['task']['seed'] in base]
            paired_rows.append(dict(candidate=candidate, baseline=other, pairs=len(differences),
                                    qe_difference_mean=np.mean(differences),
                                    qe_win_fraction=np.mean(np.asarray(differences) < 0)))
    som_groups = {name: [r for r in ok if r['task'].get('candidate') == name]
                  for name in ('rect4x4', 'hex4x4', 'hex2x8')}
    som_groups['subsamples'] = [r for r in ok if r['task']['stage'] == 'subsamples']
    for name, group in som_groups.items():
        if not group:
            continue
        firm = membership_firmness(reference['labels'], [r['labels'] for r in group])
        stable_rows.extend(dict(group=name, row_index=i, reference_region=reference['labels'][i]+1,
                                membership_firmness=value) for i, value in enumerate(firm))
        for topic in reference['selections']:
            scores = [r['selection_overlap'][topic] for r in group]
            campaign_rows.append(dict(group=name, topic=topic, mean_overlap=np.mean(scores),
                                      all_three_fraction=np.mean(np.asarray(scores) == 1.),
                                      fewer_than_three_fraction=np.mean([len(r['selections'][topic]) < 3 for r in group])))
        np.savetxt(output/f'coassignment_{name}.csv', co_assignment([r['labels'] for r in group]),
                   delimiter=',', fmt='%.4f')
    sample_rows = []
    for sample in range(100):
        group = [r for r in som_groups['subsamples'] if r['task']['sample'] == sample]
        if not group:
            continue
        sample_rows.append(dict(sample=sample, runs=len(group),
            ari_reference=np.mean([r['ari_reference'] for r in group]),
            ari_between_seeds=np.mean([adjusted_rand_index(a['labels'], b['labels']) for a,b in combinations(group,2)]),
            train_qe=np.mean([r['train_metrics']['quantization_error'] for r in group]),
            excluded_qe=np.mean([r['excluded_metrics']['quantization_error'] for r in group]),
            qe_gap=np.mean([r['excluded_metrics']['quantization_error']-r['train_metrics']['quantization_error'] for r in group]),
            excluded_te=np.mean([r['excluded_metrics']['topographic_error'] for r in group])))
    report['subsamples'] = {column: quantiles([r[column] for r in sample_rows]) for column in
                            ('ari_reference', 'ari_between_seeds', 'train_qe', 'excluded_qe', 'qe_gap', 'excluded_te')}
    report['subsamples']['negative_qe_gap_fraction'] = np.mean([r['qe_gap'] < 0 for r in sample_rows])
    conditions = sorted({(r['task']['stage'], r['task']['rate'], r['task'].get('removed', 0), r['task'].get('scaling', 'full'))
                         for r in runs if r['task']['stage'].startswith('oja')})
    for stage, rate, removed, scaling in conditions:
        group = [r for r in runs if (r['task']['stage'], r['task'].get('rate'), r['task'].get('removed', 0), r['task'].get('scaling', 'full')) == (stage, rate, removed, scaling)]
        successful = [r for r in group if r['status'] == 'ok']
        row = dict(stage=stage, rate=rate, removed=removed, scaling=scaling, runs=len(group), successful=len(successful))
        if successful:
            for metric in ('pc1_alignment', 'pc1_drift_cosine', 'learned_reference_cosine',
                           'score_rank_correlation', 'pca_score_rank_correlation', 'pc1_explained_variance'):
                row.update({f'{metric}_{stat}': value for stat, value in quantiles([r[metric] for r in successful]).items()})
            for metric in ('first_epoch_0999', 'first_epoch_099999'):
                reached = [r[metric] for r in successful if r[metric] is not None]
                row[metric+'_reached'] = len(reached)
                row[metric+'_median'] = np.median(reached) if reached else None
                row[metric+'_min'] = min(reached) if reached else None
                row[metric+'_max'] = max(reached) if reached else None
        oja_rows.append(row)
    tables = dict(finalists=finalist_rows, paired_finalists=paired_rows, subsamples=sample_rows,
                  creator_stability=stable_rows, campaigns=campaign_rows, oja=oja_rows)
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(output/f'{name}.csv', index=False)
    report['finalists'] = finalist_rows
    report['paired_finalists'] = paired_rows
    report['campaigns'] = campaign_rows
    report['oja'] = oja_rows
    (output/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    names = ['rect4x4', 'hex4x4', 'hex2x8']
    for axis, metric, label in zip(axes[:2], ('quantization_error', 'topographic_error'), ('Error de cuantización', 'Error topográfico')):
        axis.boxplot([[r['train_metrics'][metric] for r in som_groups[name]] for name in names], tick_labels=names)
        axis.set_title(label)
        axis.grid(axis='y', alpha=.2)
    axes[2].hist([r['ari_reference'] for r in sample_rows], bins=12, color='#347a91', edgecolor='white')
    axes[2].set(xlabel='ARI contra referencia (media de 3 semillas)', ylabel='Submuestras', title='100 submuestras de 40/50')
    fig.suptitle('Confirmación del ejercicio 1 — 1.000 épocas; cinco regiones', fontsize=12)
    fig.tight_layout()
    fig.savefig(output/'confirmation.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(planned=len(runs), successful=len(ok), bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    summarize(parser.parse_args().output)
