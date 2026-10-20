"""Validación y expansión de barridos y casos con parámetros acoplados."""
from copy import deepcopy
from dataclasses import asdict
from itertools import product

from src.tp4.config import KohonenConfig, OjaConfig
from src.tp4.preprocessing import PreprocessingConfig


def set_nested(config, key, value):
    parts = key.split('.')
    if len(parts) < 2 or parts[0] not in ('model', 'preprocessing'):
        raise ValueError('El barrido admite model.* o preprocessing.*')
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _sweep_inputs(config):
    base = {key: deepcopy(value) for key, value in config.items() if key not in ('grid', 'seeds', 'cases')}
    grid = config.get('grid', {})
    cases = config.get('cases', [{}])
    seeds = config.get('seeds', [base.get('model', {}).get('seed', 0)])
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds debe ser una lista no vacía sin repeticiones')
    if not isinstance(grid, dict) or any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError('grid debe contener listas no vacías')
    if not isinstance(cases, list) or not cases or any(not isinstance(case, dict) for case in cases):
        raise ValueError('cases debe ser una lista no vacía de diccionarios')
    for case in cases:
        keys = list(grid) + list(case)
        if 'model.seed' in keys:
            raise ValueError('Usar seeds para variar semillas')
        for case_key in case:
            if any(case_key == key or case_key.startswith(key + '.') or key.startswith(case_key + '.') for key in grid):
                raise ValueError(f'Parámetro compartido entre grid y cases: {case_key}')
    return base, grid, cases, seeds


def expand(config):
    """Producto cases × grid × seeds; cada caso mantiene sus valores acoplados."""
    base, grid, cases, seeds = _sweep_inputs(config)
    runs = []
    for case in cases:
        for combination in product(*grid.values()):
            candidate = deepcopy(base)
            for key, value in {**case, **dict(zip(grid, combination))}.items():
                set_nested(candidate, key, value)
            for seed in seeds:
                item = deepcopy(candidate)
                item.setdefault('model', {})['seed'] = seed
                resolve(item)
                runs.append(item)
    return runs


def resolve(config):
    allowed = {'algorithm', 'data', 'output', 'model', 'preprocessing'}
    if set(config) - allowed:
        raise ValueError(f'Claves desconocidas: {set(config) - allowed}')
    algorithm = config.get('algorithm')
    if algorithm not in ('kohonen', 'oja'):
        raise ValueError('algorithm debe ser kohonen u oja')
    cls = KohonenConfig if algorithm == 'kohonen' else OjaConfig
    model = cls(**config.get('model', {}))
    prep = PreprocessingConfig(**config.get('preprocessing', {}))
    return {**config, 'data': config.get('data', 'data/creadores.csv'),
            'output': config.get('output', 'results'), 'model': asdict(model), 'preprocessing': asdict(prep)}
