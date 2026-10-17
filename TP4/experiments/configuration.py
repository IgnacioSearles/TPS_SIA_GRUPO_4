"""Validación y expansión del producto cartesiano de experimentos."""
from copy import deepcopy
from dataclasses import asdict
from itertools import product

from src.tp4.config import KohonenConfig, OjaConfig
from src.tp4.preprocessing import PreprocessingConfig


def set_nested(config, key, value):
    parts = key.split('.')
    if parts[0] not in ('model', 'preprocessing'):
        raise ValueError('El barrido admite model.* o preprocessing.*')
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def expand(config):
    base = {k:deepcopy(v) for k,v in config.items() if k not in ('grid','seeds')}
    grid = config.get('grid', {})
    if 'model.seed' in grid:
        raise ValueError('Usar seeds para variar semillas')
    seeds = config.get('seeds', [base.get('model', {}).get('seed', 0)])
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds debe ser una lista no vacía sin repeticiones')
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f'{key}: se requiere lista no vacía')
    runs = []
    for combination in product(*grid.values()):
        candidate = deepcopy(base)
        for key, value in zip(grid, combination):
            set_nested(candidate, key, value)
        for seed in seeds:
            item = deepcopy(candidate)
            item.setdefault('model', {})['seed'] = seed
            resolve(item)  # Validar todo antes de iniciar el barrido.
            runs.append(item)
    return runs


def resolve(config):
    allowed = {'algorithm', 'data', 'output', 'model', 'preprocessing'}
    if set(config)-allowed:
        raise ValueError(f'Claves desconocidas: {set(config)-allowed}')
    algorithm = config.get('algorithm')
    if algorithm not in ('kohonen', 'oja'):
        raise ValueError('algorithm debe ser kohonen u oja')
    cls = KohonenConfig if algorithm == 'kohonen' else OjaConfig
    model = cls(**config.get('model', {}))
    prep = PreprocessingConfig(**config.get('preprocessing', {}))
    return {**config, 'data':config.get('data','data/creadores.csv'),
            'output':config.get('output','results'), 'model':asdict(model), 'preprocessing':asdict(prep)}
