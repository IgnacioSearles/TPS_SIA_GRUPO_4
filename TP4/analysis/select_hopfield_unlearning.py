"""Selecciona por el protocolo prefijado y congela una confirmación independiente.

Uso: python3 analysis/select_hopfield_unlearning.py DIRECTORIO --confirmation JSON
Sólo lee exploraciones con COMPLETED; no entrena ni observa ruido de confirmación.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from analysis.hopfield_qr import ROOT
from analysis.hopfield_qr_unlearning import run_specs

CONFIG_KEYS = ('name', 'modes', 'learning_rates', 'checkpoints', 'training_seeds',
               'noise_levels', 'query_seeds', 'max_iterations', 'remove_shared')


def project_path(path):
    """Rutas portables dentro del proyecto; conserva rutas externas explícitas."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def rank_candidates(directory):
    directory = Path(directory)
    if not (directory / 'COMPLETED').is_file():
        raise ValueError('La exploración no está completa')
    config = json.loads((directory / 'config.json').read_text())
    results = pd.read_csv(directory / 'resultados.csv')
    stability = pd.read_csv(directory / 'estabilidad.csv')
    query_count = len(config['qr_names']) * len(config['noise_levels']) * len(config['query_seeds'])
    checkpoints = len(list(run_specs(config))) * (len(config['checkpoints']) - 1)
    if len(results) != (checkpoints + 2) * query_count:
        raise ValueError('Cantidad de evaluaciones incompatible con el protocolo')
    identity = ['method', 'mode', 'learning_rate', 'training_seed', 'attempts',
                'origin', 'noise', 'query_seed']
    if results.duplicated(identity).any():
        raise ValueError('Hay consultas duplicadas dentro de un modelo')
    originals = stability[(stability.method == 'unlearning') &
                           stability.state.isin(config['qr_names'])]
    candidates = results[(results.method == 'unlearning') & (results.noise > 0)]
    rows = []
    for (mode, rate, attempts), group in candidates.groupby(['mode', 'learning_rate', 'attempts']):
        stored = originals[(originals['mode'] == mode) & (originals.learning_rate == rate) &
                           (originals.attempts == attempts)]
        expected_seeds = config['training_seeds'] if mode == 'dream' else config['training_seeds'][:1]
        if sorted(group.training_seed.unique()) != sorted(expected_seeds):
            raise ValueError('Faltan semillas de entrenamiento')
        if len(stored) != len(config['qr_names']) * len(expected_seeds):
            raise ValueError('Faltan diagnósticos de originales')
        counts = stored.groupby('training_seed').stable.sum()
        scores = group.groupby('training_seed').correct.mean()
        rows.append({'mode': mode, 'learning_rate': rate, 'attempts': int(attempts),
                     'eligible': bool((counts == len(config['qr_names'])).all()),
                     'min_stable_originals': int(counts.min()),
                     'mean_accuracy': float(scores.mean()),
                     'min_training_seed_accuracy': float(scores.min()),
                     'max_training_seed_accuracy': float(scores.max()),
                     'cycle_fraction': float(group.cycle.mean()),
                     'capped_fraction': float(group.capped.mean()),
                     'mean_updates': float(group.groupby('training_seed').updates.first().mean())})
    ranking = pd.DataFrame(rows).sort_values(
        ['eligible', 'mean_accuracy', 'attempts', 'learning_rate', 'mode'],
        ascending=[False, False, True, True, True], kind='stable').reset_index(drop=True)
    ranking.to_csv(directory / 'candidatos.csv', index=False)
    return config, ranking


def freeze_confirmation(directory, destination):
    directory, destination = Path(directory), Path(destination)
    config, ranking = rank_candidates(directory)
    viable = ranking[ranking.eligible]
    selection_path = directory / 'seleccion.json'
    if selection_path.exists() or destination.exists():
        raise FileExistsError('La selección/confirmación ya existe; no se sobrescribe')
    selection = {'selected_utc': datetime.now(timezone.utc).isoformat(),
                 'pilot_directory': project_path(directory),
                 'patterns_sha256': config['patterns_sha256'],
                 'criterion': '5 originales estables en cada semilla; media ruido>0; menos intentos; menor eta',
                 'viable_candidates': len(viable)}
    if viable.empty:
        selection.update(status='no_viable_candidate', candidate=None, confirmation=None)
    else:
        winner = viable.iloc[0].to_dict()
        confirmation = {key: config[key] for key in CONFIG_KEYS}
        confirmation.update(name='Confirmación congelada de desaprendizaje',
                            modes=[winner['mode']], learning_rates=[winner['learning_rate']],
                            checkpoints=[0, winner['attempts']], query_seeds=list(range(100, 120)))
        if set(confirmation['query_seeds']) & set(config['query_seeds']):
            raise ValueError('Las semillas de confirmación se solapan con exploración')
        content = json.dumps(confirmation, ensure_ascii=False, indent=2) + '\n'
        with destination.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(content)
        selection.update(status='frozen', candidate=winner,
                         confirmation=project_path(destination),
                         confirmation_sha256=sha256(content.encode()).hexdigest())
    with selection_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(selection, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(ranking.to_string(index=False))
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--confirmation', type=Path, required=True)
    args = parser.parse_args()
    freeze_confirmation(args.directory, args.confirmation)


if __name__ == '__main__':
    main()
