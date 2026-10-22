"""Compara ambas pseudoinversas con un candidato ya confirmado, sin reajustarlo."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.hopfield_qr import ROOT, load_patterns
from analysis.hopfield_qr_unlearning import evaluate
from src.tp4.hopfield import Hopfield
from src.tp4.queries import make_queries
from src.tp4.shared_modules import SharedModules
from src.tp4.unlearning import UnlearningTrainer


def paired_statistics(table):
    """Bootstrap por semilla de consulta; QR y redes entrenadas permanecen fijos."""
    means = table[table.noise > 0].groupby(
        ['query_seed', 'method']).correct.mean().unstack('method').sort_index()
    indices = np.random.default_rng(20261022).integers(0, len(means), size=(20000, len(means)))
    statistics = {}
    for method in ('pseudoinverse_full', 'pseudoinverse_stripped'):
        differences = (means.unlearning - means[method]).to_numpy()
        statistics[method] = {
            'delta_unlearning_minus_pseudoinverse': float(differences.mean()),
            'interval95': [float(v) for v in np.quantile(
                differences[indices].mean(axis=1), [.025, .975])]}
    return statistics


def compare(confirmation, output):
    confirmation, output = Path(confirmation), Path(output)
    if not (confirmation / 'COMPLETED').is_file():
        raise ValueError('Falta una confirmación completa')
    config = json.loads((confirmation / 'config.json').read_text())
    patterns = load_patterns().astype(float)
    if sha256(patterns.tobytes()).hexdigest() != config['patterns_sha256']:
        raise ValueError('Cambió el conjunto de QR')
    queries = make_queries(patterns, config['noise_levels'], config['query_seeds'])
    existing = pd.read_csv(confirmation / 'resultados.csv')
    existing = existing[existing.method == 'unlearning'].copy()
    seeds = config['training_seeds'] if config['modes'] == ['dream'] else config['training_seeds'][:1]
    expected = {(q.origin, q.noise, q.seed) for q in queries}
    if sorted(existing.training_seed.unique()) != sorted(seeds):
        raise ValueError('Faltan repeticiones del candidato')
    for _, group in existing.groupby('training_seed'):
        actual = set(group[['origin', 'noise', 'query_seed']].itertuples(index=False, name=None))
        if len(group) != len(expected) or actual != expected:
            raise ValueError('Las consultas del candidato no coinciden con la confirmación')
    output.mkdir(parents=True, exist_ok=False)
    source = confirmation.resolve()
    source_label = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
    (output / 'config.json').write_text(json.dumps({
        'source_confirmation': source_label, 'reference_config': config,
        'comparison': 'Frozen unlearning candidate versus both pseudoinverse representations; identical queries.'
    }, indent=2) + '\n')
    rows = []
    for method, shared in [('pseudoinverse_full', None),
                           ('pseudoinverse_stripped', SharedModules(patterns))]:
        active = shared.strip(patterns) if shared is not None else patterns
        network = Hopfield(active, rule='pseudoinverse')
        mixture = UnlearningTrainer(active).mixture
        rows.extend(evaluate(network, patterns, queries, mixture,
                             {'method': method, 'training_seed': -1, 'active_neurons': active.shape[1]},
                             config['max_iterations'], shared))
        print(f'{method}: {len(queries)} consultas', flush=True)
    existing['active_neurons'] = config['active_neurons']
    combined = pd.concat([pd.DataFrame(rows), existing], ignore_index=True)
    combined.to_csv(output / 'resultados.csv', index=False)
    combined.groupby(['method', 'noise']).agg(
        accuracy=('correct', 'mean'), correct=('correct', 'sum'), count=('correct', 'size'),
        cycles=('cycle', 'sum'), caps=('capped', 'sum')).to_csv(output / 'resumen.csv')
    (output / 'comparacion_emparejada.json').write_text(
        json.dumps(paired_statistics(combined), indent=2) + '\n')
    (output / 'COMPLETED').write_text('Comparación emparejada sin reajustar el candidato.\n')
    print(combined[combined.noise > 0].groupby('method').correct.mean().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.confirmation_results, args.output)


if __name__ == '__main__':
    main()
