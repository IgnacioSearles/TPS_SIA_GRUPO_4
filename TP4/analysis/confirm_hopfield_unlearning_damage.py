"""Validación espacial del candidato congelado, sin seleccionar nuevos parámetros."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.hopfield_qr import ROOT, load_patterns
from analysis.hopfield_qr_unlearning import evaluate, load_config
from src.tp4.hopfield import Hopfield
from src.tp4.qr import SIDE
from src.tp4.shared_modules import SharedModules
from src.tp4.unlearning import UnlearningTrainer


class DamageQuery(NamedTuple):
    origin: int
    noise: int  # Adaptador para evaluate: se renombra block_side en la salida.
    seed: int
    pattern: np.ndarray
    kind: str
    top: int
    left: int


def damage_queries(patterns, plan):
    queries = []
    for origin, pattern in enumerate(patterns):
        for side in plan['block_sizes']:
            for corner in range(4):
                top = (corner // 2) * (SIDE - side)
                left = (corner % 2) * (SIDE - side)
                damaged = pattern.reshape(SIDE, SIDE).copy()
                damaged[top:top + side, left:left + side] = plan['corner_fill']
                queries.append(DamageQuery(origin, side, corner, damaged.ravel(),
                                           'corner_white', top, left))
            for seed in plan['spot_seeds']:
                rng = np.random.default_rng((20261022, origin, side, seed))
                top, left = map(int, rng.integers(0, SIDE - side + 1, size=2))
                damaged = pattern.reshape(SIDE, SIDE).copy()
                damaged[top:top + side, left:left + side] = plan['spot_fill']
                queries.append(DamageQuery(origin, side, seed, damaged.ravel(),
                                           'spot_black', top, left))
    return queries


def run(selection_path, confirmation_directory, plan_path, output):
    selection = json.loads(Path(selection_path).read_text())
    config_path = Path(selection['confirmation'])
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_bytes = config_path.read_bytes()
    if sha256(config_bytes).hexdigest() != selection['confirmation_sha256']:
        raise ValueError('El candidato fue modificado después de congelarlo')
    config = load_config(config_path)
    confirmation = Path(confirmation_directory)
    if not (confirmation / 'COMPLETED').is_file():
        raise ValueError('Falta la confirmación completa')
    manifest = json.loads((confirmation / 'config.json').read_text())
    for key, value in config.items():
        if manifest[key] != value:
            raise ValueError(f'La confirmación no corresponde al candidato: {key}')
    results = pd.read_csv(confirmation / 'resultados.csv')
    stability = pd.read_csv(confirmation / 'estabilidad.csv')
    stored = stability[(stability.method == 'unlearning') &
                       stability.state.isin(manifest['qr_names'])]
    expected_seeds = config['training_seeds'] if config['modes'][0] == 'dream' else config['training_seeds'][:1]
    if not stored.stable.all() or len(stored) != 5 * len(expected_seeds):
        raise ValueError('El candidato no preservó los originales en todas las semillas')
    scores = results[results.noise > 0].groupby('method').correct.mean()
    if scores['unlearning'] <= scores['hebb']:
        raise ValueError('El candidato no mejoró a Hebb en confirmación')
    patterns = load_patterns().astype(float)
    if sha256(patterns.tobytes()).hexdigest() != selection['patterns_sha256']:
        raise ValueError('Cambió el conjunto de QR')
    plan_bytes = Path(plan_path).read_bytes()
    plan = json.loads(plan_bytes)
    queries = damage_queries(patterns, plan)
    if len(queries) != plan['queries_per_model']:
        raise ValueError('Presupuesto espacial diferente del plan')
    shared = SharedModules(patterns) if config['remove_shared'] else None
    active = shared.strip(patterns) if shared is not None else patterns
    mixture = UnlearningTrainer(active).mixture
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps({
        'candidate': config, 'plan': plan, 'plan_sha256': sha256(plan_bytes).hexdigest(),
        'selection': selection, 'confirmation_directory': str(confirmation.resolve()),
        'created_utc': datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2) + '\n')
    rows = []

    def measure(network, metadata):
        measured = evaluate(network, patterns, queries, mixture, metadata,
                            config['max_iterations'], shared)
        for row, query in zip(measured, queries):
            row['block_side'] = row.pop('noise')
            row.update(kind=query.kind, top=query.top, left=query.left,
                       covered_modules=query.noise ** 2)
        rows.extend(measured)
        pd.DataFrame(rows).to_csv(output / 'resultados.csv', index=False)
        print(f'{metadata}: {len(measured)} consultas espaciales', flush=True)

    for method, network in [('hebb', Hopfield(active)), ('nearest', None)]:
        measure(network, {'method': method, 'training_seed': -1})
    seeds = config['training_seeds'] if config['modes'][0] == 'dream' else config['training_seeds'][:1]
    for seed in seeds:
        trainer = UnlearningTrainer(active, mode=config['modes'][0],
                                    learning_rate=config['learning_rates'][0], seed=seed,
                                    max_iterations=config['max_iterations'])
        for _ in range(max(config['checkpoints'])):
            trainer.step()
        measure(trainer.network, {'method': 'unlearning', 'training_seed': seed,
                                 'attempts': trainer.attempts, 'updates': trainer.updates})
    table = pd.DataFrame(rows)
    table.groupby(['method', 'training_seed', 'kind', 'block_side'])[
        ['correct', 'hamming', 'query_hamming', 'cycle', 'capped']].mean().to_csv(output / 'resumen.csv')
    (output / 'COMPLETED').write_text('Validación espacial completa, sin reajustar parámetros.\n')
    print(f'Resultados: {output}; filas: {len(table)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--confirmation-results', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.selection, args.confirmation_results, args.plan, args.output)


if __name__ == '__main__':
    main()
