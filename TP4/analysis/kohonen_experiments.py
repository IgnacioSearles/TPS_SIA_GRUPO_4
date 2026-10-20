"""Compara barridos SOM: ocupación, cohesión y estabilidad sin alinear la grilla."""
import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.reporting import group_id
from experiments.session import file_digest
from src.tp4.numerics import distances
from src.tp4.preprocessing import Preprocessor

METRICS = ['quantization_error', 'topographic_error', 'common_centroid_distance',
           'occupied_neurons', 'empty_fraction', 'singleton_creator_fraction',
           'occupancy_entropy', 'epochs_completed']


@dataclass
class DatasetReference:
    frame: pd.DataFrame
    standardized: np.ndarray


def adjusted_rand_index(first, second):
    """ARI entre asignaciones; no depende del número/nombre de las neuronas."""
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 1 or first.shape != second.shape or len(first) < 2:
        raise ValueError('ARI requiere dos asignaciones iguales en longitud, con al menos dos observaciones')
    _, a = np.unique(first, return_inverse=True)
    _, b = np.unique(second, return_inverse=True)
    shape = (a.max() + 1, b.max() + 1)
    contingency = np.bincount(a * shape[1] + b, minlength=shape[0] * shape[1]).reshape(shape)
    choose_two = lambda values: (values * (values - 1) / 2).sum()
    observed = choose_two(contingency)
    row_pairs = choose_two(contingency.sum(axis=1))
    col_pairs = choose_two(contingency.sum(axis=0))
    expected = row_pairs * col_pairs / (len(first) * (len(first) - 1) / 2)
    denominator = (row_pairs + col_pairs) / 2 - expected
    return 1.0 if denominator == 0 else float((observed - expected) / denominator)


def occupancy_metrics(winners, neuron_count, canonical):
    membership = winners[:, None] == np.arange(neuron_count)[None, :]
    counts = membership.sum(axis=0)
    occupied = counts[counts > 0]
    centroids = membership.T @ canonical / np.maximum(counts[:, None], 1)
    cohesion = np.linalg.norm(canonical - centroids[winners], axis=1).mean()
    probabilities = occupied / len(winners)
    return {
        'common_centroid_distance': float(cohesion),
        'occupied_neurons': int(len(occupied)),
        'empty_fraction': float((counts == 0).mean()),
        'singleton_creator_fraction': float((occupied == 1).sum() / len(winners)),
        'occupancy_entropy': float(-(probabilities * np.log(probabilities)).sum() / np.log(neuron_count)),
    }


def flatten(parameters, prefix):
    values = {}
    for key, value in parameters.items():
        name = f'{prefix}.{key}'
        if isinstance(value, dict):
            values.update(flatten(value, name))
        else:
            values[name] = json.dumps(value, sort_keys=True) if isinstance(value, list) else value
    return values


def descriptor(config, dataset_digest, directory):
    config_group = group_id(config)
    group = hashlib.sha256((config_group + dataset_digest).encode()).hexdigest()[:12]
    representation = json.dumps(config['preprocessing'], sort_keys=True)
    return {
        'run_directory': str(directory), 'group': group, 'config_group': config_group,
        'dataset_digest': dataset_digest,
        'representation': hashlib.sha256(representation.encode()).hexdigest()[:12],
        'seed': config['model']['seed'],
        'neuron_count': config['model']['rows'] * config['model']['cols'],
        **flatten(config['model'], 'model'), **flatten(config['preprocessing'], 'preprocessing'),
    }


def dataset_reference(config, expected_digest, cache):
    key = (config['data'], expected_digest)
    if key not in cache:
        if file_digest(config['data']) != expected_digest:
            raise ValueError('El catálogo actual no coincide con el usado por el experimento')
        frame = pd.read_csv(config['data'])
        cache[key] = DatasetReference(frame, Preprocessor().fit_transform(frame))
    return cache[key]


def evaluate_saved_run(directory, config, reference):
    preprocessing = Preprocessor.from_state(json.loads((directory / 'preprocessing.json').read_text()))
    samples = preprocessing.transform(reference.frame)
    with np.load(directory / 'model.npz', allow_pickle=False) as arrays:
        weights = arrays['weights']
    count = config['model']['rows'] * config['model']['cols']
    if weights.shape != (count, samples.shape[1]) or not np.isfinite(weights).all():
        raise ValueError('Pesos incompatibles con el preprocesamiento')
    winners = distances(samples, weights).argmin(axis=1)
    metrics = json.loads((directory / 'metrics.json').read_text())
    return {
        'quantization_error': metrics['quantization_error'],
        'topographic_error': metrics['topographic_error'],
        'epochs_completed': metrics['epochs'],
        **occupancy_metrics(winners, count, reference.standardized),
    }, winners


def collect_runs(root):
    rows, assignments, errors, cache = [], {}, [], {}
    plans = sorted(Path(root).rglob('plan.json'))
    for path in plans:
        plan = json.loads(path.read_text())
        for index, config in enumerate(plan['configs']):
            if config['algorithm'] != 'kohonen':
                continue
            directory = path.parent / f'run_{index:04d}'
            digest = plan['datasets'][config['data']]
            row = descriptor(config, digest, directory)
            try:
                row['status'] = _collect_run(directory, config, digest, cache, row, assignments)
            except Exception as error:
                row['status'] = 'invalid'
                errors.append({'run_directory': str(directory), 'error': str(error)})
            rows.append(row)
    if not rows:
        raise ValueError(f'No hay planes de Kohonen en {root}')
    return pd.DataFrame(rows), assignments, pd.DataFrame(errors, columns=['run_directory', 'error'])


def _collect_run(directory, config, digest, cache, row, assignments):
    marker = directory / 'record.json'
    if not marker.exists():
        return 'paused' if (directory / 'checkpoint.npz').exists() else 'pending'
    record = json.loads(marker.read_text())
    if record['failure'] is not None:
        return 'failed'
    reference = dataset_reference(config, digest, cache)
    metrics, winners = evaluate_saved_run(directory, config, reference)
    row.update(metrics)
    assignments.setdefault((row['group'], row['seed']), winners)
    return 'ok'


def stability_table(assignments):
    rows = []
    groups = sorted({group for group, _ in assignments})
    for group in groups:
        seeds = sorted(seed for candidate, seed in assignments if candidate == group)
        for first, second in combinations(seeds, 2):
            rows.append({'group': group, 'seed_a': first, 'seed_b': second,
                         'ari': adjusted_rand_index(assignments[group, first], assignments[group, second])})
    return pd.DataFrame(rows, columns=['group', 'seed_a', 'seed_b', 'ari'])


def summarize_runs(runs, stability):
    counts = runs.groupby('group').agg(
        planned_runs=('status', 'size'),
        successful_runs=('status', lambda values: (values == 'ok').sum()),
        failed_runs=('status', lambda values: (values == 'failed').sum()),
        pending_runs=('status', lambda values: values.isin(['pending', 'paused']).sum()),
        invalid_runs=('status', lambda values: (values == 'invalid').sum()),
    )
    success = runs[runs.status == 'ok'].drop_duplicates(['group', 'seed'])
    if success.empty:
        return counts.reset_index()
    metrics = success.groupby('group')[METRICS].agg(['mean', 'std'])
    metrics.columns = ['_'.join(column) for column in metrics.columns]
    metadata = [column for column in runs if column.startswith(('model.', 'preprocessing.')) and column != 'model.seed']
    metadata += ['representation', 'neuron_count', 'dataset_digest', 'config_group']
    summary = counts.join(metrics).join(success.groupby('group').size().rename('successful_seeds'))
    summary = summary.join(runs.groupby('group')[metadata].first())
    if not stability.empty:
        summary = summary.join(stability.groupby('group').ari.agg(ari_mean='mean', ari_std='std', ari_min='min'))
    return summary.reset_index()


def compare(root, output):
    runs, assignments, errors = collect_runs(root)
    stability = stability_table(assignments)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / 'runs.csv', index=False)
    summarize_runs(runs, stability).to_csv(output / 'summary.csv', index=False)
    stability.to_csv(output / 'stability.csv', index=False)
    errors.to_csv(output / 'errors.csv', index=False)
    print(f'{len(runs)} corridas encontradas; {len(errors)} inválidas. Comparación: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/kohonen_suite')
    parser.add_argument('--output', default='results/kohonen_comparison')
    args = parser.parse_args()
    compare(args.root, args.output)


if __name__ == '__main__':
    main()
