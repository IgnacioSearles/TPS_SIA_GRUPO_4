"""Barrido de desaprendizaje con consultas emparejadas y checkpoints acumulativos.

Uso: OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config ...
Ver docs/experimentos_desaprendizaje.md para exploración y confirmación separadas.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.hopfield_qr import QRS, ROOT, load_patterns
from src.tp4.hopfield import Hopfield
from src.tp4.queries import make_queries
from src.tp4.recall_outcome import classify
from src.tp4.shared_modules import SharedModules
from src.tp4.unlearning import UnlearningTrainer


def load_config(path):
    config = json.loads(Path(path).read_text())
    keys = {'name', 'modes', 'learning_rates', 'checkpoints', 'training_seeds',
            'noise_levels', 'query_seeds', 'max_iterations'}
    if not keys <= set(config) or set(config) - keys - {'remove_shared'}:
        raise ValueError(f'Claves requeridas: {sorted(keys)}; opcional: remove_shared')
    config.setdefault('remove_shared', False)
    if type(config['remove_shared']) is not bool:
        raise ValueError('remove_shared debe ser booleano')
    if not isinstance(config['name'], str) or not config['name']:
        raise ValueError('name debe ser texto no vacío')
    for key in keys - {'name', 'max_iterations'}:
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f'{key}: se requiere una lista no vacía sin duplicados')
    if not set(config['modes']) <= {'directed', 'dream'}:
        raise ValueError('Modos desconocidos')
    for key in ('checkpoints', 'training_seeds', 'query_seeds'):
        if any(type(v) is not int or v < 0 for v in config[key]):
            raise ValueError(f'{key}: enteros no negativos')
    if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0
           for v in config['learning_rates']):
        raise ValueError('learning_rates: valores finitos positivos')
    if any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= .5
           for v in config['noise_levels']):
        raise ValueError('noise_levels: valores entre 0 y 0.5')
    if (type(config['max_iterations']) is not int or config['max_iterations'] < 1):
        raise ValueError('max_iterations: entero positivo')
    if sorted(config['checkpoints'])[0] != 0:
        raise ValueError('checkpoints debe incluir 0 como control sin desaprendizaje')
    if 0 not in config['noise_levels']:
        raise ValueError('noise_levels debe incluir 0 para comprobar los originales')
    return config


def run_specs(config):
    # directed no usa aleatoriedad: no contar copias idénticas como repeticiones.
    for mode, rate in itertools.product(config['modes'], config['learning_rates']):
        seeds = config['training_seeds'] if mode == 'dream' else config['training_seeds'][:1]
        for seed in seeds:
            yield mode, rate, seed


def evaluate(network, patterns, queries, mixture, metadata, max_iterations, shared=None):
    """Consultas completas; recuperar/clasificar en espacio activo y restaurar salida."""
    active_patterns = shared.strip(patterns) if shared is not None else patterns
    rows = []
    for query in queries:
        query_state = shared.strip(query.pattern) if shared is not None else query.pattern
        if network is None:
            distances = np.count_nonzero(active_patterns != query_state, axis=1)
            state = active_patterns[int(distances.argmin())]
            category = 'correct' if np.array_equal(state, active_patterns[query.origin]) else 'other_pattern'
            stopping = {'iterations': 0, 'stop_reason': 'reference', 'energy_initial': np.nan,
                        'energy_final': np.nan, 'energy_rises': 0}
        else:
            result = network.recall(query_state, max_iterations=max_iterations)
            state = result.state
            category = classify(state, result.stop_reason, active_patterns, query.origin).category
            stopping = {'iterations': result.iterations, 'stop_reason': result.stop_reason,
                        'energy_initial': result.energies[0], 'energy_final': result.energies[-1],
                        'energy_rises': result.energy_rises}
        restored = shared.restore(state) if shared is not None else state
        rows.append({**metadata, 'origin': query.origin, 'qr': list(QRS)[query.origin],
                     'noise': query.noise, 'query_seed': query.seed,
                     'query_hamming': int(np.count_nonzero(query.pattern != patterns[query.origin])),
                     'network_query_hamming': int(np.count_nonzero(query_state != active_patterns[query.origin])),
                     'category': category, 'correct': category == 'correct',
                     'hamming': int(np.count_nonzero(restored != patterns[query.origin])),
                     'mixture': bool(np.array_equal(state, mixture)),
                     'inverse_mixture': bool(np.array_equal(state, -mixture)),
                     'cycle': category == 'cycle', 'capped': category == 'max_iterations', **stopping})
    return rows


def diagnostics(network, patterns, mixture, metadata):
    rows = []
    names = [*QRS, 'mezcla', 'inversa_mezcla']
    states = np.vstack([patterns, mixture, -mixture])
    fields = states @ network.weights_.T
    for name, state, field in zip(names, states, fields):
        margin = state * field
        wrong = int(np.count_nonzero(margin < 0))
        rows.append({**metadata, 'state': name, 'stable': wrong == 0,
                     'wrong_fields': wrong, 'zero_fields': int(np.count_nonzero(field == 0)),
                     'min_margin': float(margin.min()), 'energy': network.energy(state)})
    return rows


def run(config, output):
    patterns = load_patterns().astype(float)
    queries = make_queries(patterns, config['noise_levels'], config['query_seeds'])
    shared = SharedModules(patterns) if config.get('remove_shared', False) else None
    active_patterns = shared.strip(patterns) if shared is not None else patterns
    if active_patterns.shape[1] == 0:
        raise ValueError('No hay módulos diferentes para entrenar una red')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**config, 'patterns_sha256': sha256(patterns.tobytes()).hexdigest(),
                'remove_shared': shared is not None,
                'full_neurons': patterns.shape[1], 'active_neurons': active_patterns.shape[1],
                'removed_shared': patterns.shape[1] - active_patterns.shape[1],
                'active_patterns_sha256': sha256(active_patterns.tobytes()).hexdigest(),
                'qr_names': list(QRS), 'created_utc': datetime.now(timezone.utc).isoformat(),
                'python_version': sys.version, 'numpy_version': np.__version__,
                'pandas_version': pd.__version__}
    (output / 'config.json').write_text(json.dumps(manifest, indent=2) + '\n')
    mixture = UnlearningTrainer(active_patterns).mixture
    results, stability, training = [], [], []
    hebb = Hopfield(active_patterns)
    controls = {'mode': 'none', 'learning_rate': 0., 'training_seed': -1,
                'attempts': 0, 'updates': 0, 'remove_shared': shared is not None}
    for method, network in [('hebb', hebb), ('nearest', None)]:
        metadata = {'method': method, **controls}
        results.extend(evaluate(network, patterns, queries, mixture, metadata, config['max_iterations'], shared))
        if network is not None:
            stability.extend(diagnostics(network, active_patterns, mixture, metadata))
    for mode, rate, seed in run_specs(config):
        trainer = UnlearningTrainer(active_patterns, mode=mode, learning_rate=rate, seed=seed,
                                    max_iterations=config['max_iterations'])
        for checkpoint in sorted(config['checkpoints']):
            while trainer.attempts < checkpoint:
                step = trainer.step()
                training.append({'mode': mode, 'learning_rate': rate, 'training_seed': seed,
                                 'remove_shared': shared is not None, **asdict(step)})
            if checkpoint == 0:
                continue  # El checkpoint cero es el control Hebb ya evaluado.
            metadata = {'method': 'unlearning', 'mode': mode, 'learning_rate': rate,
                        'training_seed': seed, 'attempts': checkpoint, 'updates': trainer.updates,
                        'remove_shared': shared is not None}
            results.extend(evaluate(trainer.network, patterns, queries, mixture, metadata,
                                    config['max_iterations'], shared))
            stability.extend(diagnostics(trainer.network, active_patterns, mixture, metadata))
            print(f'{mode}, eta={rate}, semilla={seed}, intentos={checkpoint}, '
                  f'updates={trainer.updates}', flush=True)
        # Guardar progreso entre entrenamientos; no conservar matrices N x N por checkpoint.
        pd.DataFrame(results).to_csv(output / 'resultados.csv', index=False)
        pd.DataFrame(stability).to_csv(output / 'estabilidad.csv', index=False)
        pd.DataFrame(training).to_csv(output / 'entrenamiento.csv', index=False)
    table = pd.DataFrame(results)
    groups = ['method', 'mode', 'learning_rate', 'training_seed', 'attempts', 'updates', 'noise']
    metrics = ['correct', 'hamming', 'iterations', 'mixture', 'inverse_mixture', 'cycle', 'capped']
    table.groupby(groups)[metrics].mean().to_csv(output / 'resumen.csv')
    table.groupby([*groups, 'qr'])[metrics].mean().to_csv(output / 'resumen_por_qr.csv')
    baseline = table[table.method == 'hebb'][['origin', 'noise', 'query_seed', 'correct']]
    paired = table.merge(baseline, on=['origin', 'noise', 'query_seed'], suffixes=('', '_hebb'),
                         validate='many_to_one')
    paired['wins_over_hebb'] = paired.correct & ~paired.correct_hebb
    paired['losses_to_hebb'] = ~paired.correct & paired.correct_hebb
    paired.groupby(groups)[['wins_over_hebb', 'losses_to_hebb']].sum().to_csv(output / 'pares_hebb.csv')
    (output / 'COMPLETED').write_text('Todos los entrenamientos y evaluaciones finalizaron.\n')
    print(f'Resultados: {output}\nRecuperaciones evaluadas: {len(table)}', flush=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    candidates = len(list(run_specs(config))) * (len(config['checkpoints']) - 1)
    queries = len(QRS) * len(config['noise_levels']) * len(config['query_seeds'])
    print(f'{candidates} checkpoints + 2 controles; {queries} consultas por modelo; '
          f'{(candidates + 2) * queries} recuperaciones', flush=True)
    if args.dry_run:
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT / 'results' / 'hopfield_unlearning' / stamp
    run(config, output)


if __name__ == '__main__':
    main()
