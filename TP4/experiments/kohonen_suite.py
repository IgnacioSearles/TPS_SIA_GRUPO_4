"""Genera una batería amplia y reproducible; nunca inicia entrenamientos."""
import argparse
from itertools import product
import json
from pathlib import Path
import random

from src.tp4.preprocessing import DEFAULT_DROP, DEFAULT_LOG
from .configuration import expand, resolve

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(5))
SHAPES = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
          (2, 6), (6, 2), (3, 4), (4, 3), (2, 8), (8, 2), (4, 6), (6, 4)]


def shape_case(rows, cols):
    return {'model.rows': rows, 'model.cols': cols}


def base(stage, *, epochs=500, seeds=SEEDS):
    return {
        'algorithm': 'kohonen', 'data': 'data/creadores.csv',
        'output': f'results/kohonen_suite/{stage}',
        'model': {'rows': 4, 'cols': 4, 'epochs': epochs, 'shuffle': True,
                  'learning_rate_start': .3, 'learning_rate_end': .01,
                  'decay': 'exponential', 'radius_start': None, 'radius_end': .5,
                  'radius_decay': 'exponential', 'neighborhood': 'gaussian',
                  'initialization': 'samples', 'early_stopping': {'enabled': False}},
        'preprocessing': {'exclude': DEFAULT_DROP.copy(), 'log_columns': [], 'pca_components': None},
        'seeds': list(seeds), 'grid': {'model.topology': ['rectangular', 'hexagonal']},
    }


def schedule_cases(prefix, starts, ends):
    cases = [{f'model.{prefix}_start': start, f'model.{prefix}_end': start,
              'model.decay' if prefix == 'learning_rate' else 'model.radius_decay': 'constant'} for start in starts]
    decay_key = 'model.decay' if prefix == 'learning_rate' else 'model.radius_decay'
    for start, end, decay in product(starts, ends, ('linear', 'exponential')):
        if end < start:
            cases.append({f'model.{prefix}_start': start, f'model.{prefix}_end': end, decay_key: decay})
    return cases


def representation_cases():
    cases = [
        {'preprocessing.exclude': DEFAULT_DROP.copy(), 'preprocessing.log_columns': [], 'preprocessing.pca_components': None},
        {'preprocessing.exclude': [], 'preprocessing.log_columns': [], 'preprocessing.pca_components': None},
        {'preprocessing.exclude': DEFAULT_DROP.copy(), 'preprocessing.log_columns': DEFAULT_LOG.copy(), 'preprocessing.pca_components': None},
    ]
    for log_columns, components in product(([], DEFAULT_LOG.copy()), (2, 6, 10, 13, 25)):
        if log_columns and components in (2, 25):
            continue
        cases.append({'preprocessing.exclude': DEFAULT_DROP.copy(),
                      'preprocessing.log_columns': log_columns, 'preprocessing.pca_components': components})
    cases.append({'preprocessing.exclude': DEFAULT_DROP + ['followers', 'average_impressions', 'average_video_views'],
                  'preprocessing.log_columns': [], 'preprocessing.pca_components': None})
    return cases


def interaction_cases(count=120, seed=20261018):
    """Muestra reproducible sin construir el producto cartesiano completo."""
    rng = random.Random(seed)
    cases, seen = [], set()
    representations = representation_cases()
    while len(cases) < count:
        case = shape_case(*rng.choice([(3, 3), (4, 4), (5, 5), (3, 4), (4, 6)]))
        case.update(rng.choice(representations))
        case.update({
            'model.topology': rng.choice(['rectangular', 'hexagonal']),
            'model.learning_rate_start': rng.choice([.05, .1, .3, .6]),
            'model.learning_rate_end': rng.choice([.001, .01, .05]),
            'model.decay': rng.choice(['linear', 'exponential']),
            'model.radius_start': rng.choice([.5, 1., 2., 4.]),
            'model.radius_end': rng.choice([.1, .5]),
            'model.radius_decay': rng.choice(['linear', 'exponential']),
            'model.neighborhood': rng.choice(['gaussian', 'uniform']),
            'model.initialization': rng.choice(['samples', 'uniform']),
            'model.shuffle': rng.choice([True, False]),
        })
        key = json.dumps(case, sort_keys=True)
        if key not in seen:
            cases.append(case)
            seen.add(key)
    return cases


def build_suite():
    smoke = base('00_smoke', epochs=10, seeds=[0, 1])
    smoke['model'].update(rows=3, cols=3)
    smoke['cases'] = [{'preprocessing.pca_components': value} for value in (None, 10)]
    suite = {'00_smoke': smoke}
    geometry = base('01_geometry')
    geometry['cases'] = [shape_case(*shape) for shape in SHAPES]
    geometry['grid']['model.initialization'] = ['samples', 'uniform']
    suite['01_geometry'] = geometry

    learning = base('02_learning')
    learning['cases'] = schedule_cases('learning_rate', [.05, .1, .3, .6], [.001, .01, .05])
    suite['02_learning'] = learning

    neighborhood = base('03_neighborhood')
    neighborhood['cases'] = schedule_cases('radius', [.5, 1., 2., 4., 8.], [.1, .5, 1.])
    neighborhood['grid']['model.neighborhood'] = ['gaussian', 'uniform']
    suite['03_neighborhood'] = neighborhood

    representation = base('04_representation')
    representation['cases'] = [{**shape_case(*shape), **parameters} for shape, parameters in
                               product([(3, 3), (4, 4), (5, 5)], representation_cases())]
    suite['04_representation'] = representation

    duration = base('05_duration')
    duration['grid']['model.epochs'] = [50, 100, 250, 500, 1000, 2000]
    suite['05_duration'] = duration

    stopping = base('06_early_stopping', epochs=1000)
    stopping['model']['early_stopping']['enabled'] = True
    stopping['grid'].update({'model.early_stopping.min_delta': [1e-6, 1e-4, 1e-3, 1e-2],
                             'model.early_stopping.patience': [10, 30, 60],
                             'model.early_stopping.min_epochs': [50, 150]})
    suite['06_early_stopping'] = stopping

    order = base('07_order_initialization', epochs=1000, seeds=range(10))
    order['grid'].update({'model.shuffle': [True, False], 'model.initialization': ['samples', 'uniform']})
    suite['07_order_initialization'] = order

    stability = base('08_reference_stability', epochs=1000, seeds=range(20))
    stability['cases'] = [shape_case(*shape) for shape in [(3, 3), (4, 4), (5, 5)]]
    suite['08_reference_stability'] = stability

    interactions = base('09_interactions', epochs=1000)
    interactions['grid'] = {}
    interactions['cases'] = interaction_cases()
    suite['09_interactions'] = interactions
    return suite


def suite_budget(suite):
    rows = []
    for name, config in suite.items():
        runs = [resolve(candidate) for candidate in expand(config)]
        epochs = sum(run['model']['epochs'] for run in runs)
        rows.append((name, len(runs), epochs, epochs * 50))
    return rows


def confirmation_config(paths, seed_start=100, seed_count=30):
    """Repite candidatos seleccionados con semillas distintas de las exploratorias."""
    if seed_start < 0 or seed_count < 1:
        raise ValueError('Las semillas deben ser no negativas y la cantidad positiva')
    candidates = [resolve(json.loads(Path(path).read_text())) for path in paths]
    if not candidates or any(item['algorithm'] != 'kohonen' for item in candidates):
        raise ValueError('Se requieren configuraciones finales de Kohonen')
    datasets = {str(Path(item['data']).resolve()) for item in candidates}
    if len(datasets) != 1:
        raise ValueError('Los candidatos deben usar el mismo catálogo')
    config = base('10_confirmation', seeds=range(seed_start, seed_start + seed_count))
    config['data'] = datasets.pop()
    config['grid'] = {}
    config['cases'] = []
    for candidate in candidates:
        parameters = {f'model.{key}': value for key, value in candidate['model'].items() if key != 'seed'}
        parameters.update({f'preprocessing.{key}': value for key, value in candidate['preprocessing'].items()})
        config['cases'].append(parameters)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Guardar los JSON; sin esta opción solo muestra el presupuesto')
    parser.add_argument('--confirm-from', nargs='+', help='config.json de candidatos seleccionados para confirmar')
    parser.add_argument('--seed-start', type=int, default=100)
    parser.add_argument('--seed-count', type=int, default=30)
    args = parser.parse_args()
    suite = ({'10_confirmation': confirmation_config(args.confirm_from, args.seed_start, args.seed_count)}
             if args.confirm_from else build_suite())
    if args.write:
        destination = ROOT / 'configs' / 'kohonen_suite'
        destination.mkdir(parents=True, exist_ok=True)
        for name, config in suite.items():
            (destination / f'{name}.json').write_text(json.dumps(config, indent=2) + '\n')
    rows = suite_budget(suite)
    for name, runs, epochs, updates in rows:
        print(f'{name:25} {runs:5} corridas | {epochs:9,} épocas máximas | {updates:11,} actualizaciones (50 creadores)')
    print(f'Total: {sum(row[1] for row in rows)} corridas; {sum(row[2] for row in rows):,} épocas máximas')
    print('No se ejecutó ningún entrenamiento.')


if __name__ == '__main__':
    main()
