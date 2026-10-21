"""Extiende corridas terminadas sin repetir épocas ni reiniciar el decaimiento."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd

from src.tp4.checkpoint import load_checkpoint, save_checkpoint
from src.tp4.config import KohonenConfig
from src.tp4.export import write_json
from src.tp4.preprocessing import Preprocessor
from .configuration import resolve
from .runner import _drive, create_output_directory
from .session import create_plan, file_digest, runtime_versions


def completed_runs(source):
    plan = json.loads((source / 'plan.json').read_text())
    if plan['runtime'] != runtime_versions():
        raise ValueError('Python/NumPy no coinciden con la ejecución original')
    for path, expected in plan['datasets'].items():
        if file_digest(path) != expected:
            raise ValueError('El catálogo cambió desde el entrenamiento original')
    runs = []
    for index, config in enumerate(plan['configs']):
        directory = source / f'run_{index:04d}'
        marker = directory / 'record.json'
        if not marker.exists() or config['algorithm'] != 'kohonen':
            continue
        if json.loads(marker.read_text())['failure'] is None:
            runs.append((directory, config))
    if not runs:
        raise ValueError('No hay corridas completas de Kohonen para extender')
    return plan, runs


def validate_origin(directory, config, cache):
    state = json.loads((directory / 'preprocessing.json').read_text())
    key = (config['data'], json.dumps(state, sort_keys=True))
    if key not in cache:
        frame = pd.read_csv(config['data'])
        cache[key] = Preprocessor.from_state(state).transform(frame)
    checkpoint = load_checkpoint(directory / 'checkpoint.npz')
    expected = {**asdict(KohonenConfig(**config['model'])), 'layout': config['model']['topology']}
    # Migración explícita: los checkpoints previos no tenían este parámetro opcional.
    parameters = {**checkpoint.parameters}
    parameters.setdefault('schedule_epochs', None)
    checkpoint = replace(checkpoint, parameters=parameters)
    checkpoint.validate('kohonen', expected, cache[key])
    if checkpoint.stopped:
        raise ValueError('La corrida terminó por early stopping; no se fuerza su continuación')
    return checkpoint, state


def prepare_extension(source, target_epochs, output_root):
    source = Path(source).resolve()
    original_plan, origins = completed_runs(source)
    output = create_output_directory(output_root, f'extension_{target_epochs}')
    configs, provenance, cache = [], [], {}
    for index, (directory, original) in enumerate(origins):
        checkpoint, preprocessing = validate_origin(directory, original, cache)
        if target_epochs <= checkpoint.epoch:
            raise ValueError('El objetivo debe superar las épocas ya completadas')
        config = deepcopy(original)
        schedule_epochs = config['model'].get('schedule_epochs') or config['model']['epochs']
        config['model'].update(epochs=target_epochs, schedule_epochs=schedule_epochs)
        config['output'] = str(output)
        config = resolve(config)
        destination = output / f'run_{index:04d}'
        destination.mkdir()
        parameters = {**asdict(KohonenConfig(**config['model'])), 'layout': config['model']['topology']}
        save_checkpoint(destination / 'checkpoint.npz', replace(checkpoint, parameters=parameters))
        write_json(destination / 'preprocessing.json', preprocessing)
        origin = {'source_run': str(directory), 'source_epoch': checkpoint.epoch,
                  'target_epochs': target_epochs, 'schedule_epochs': schedule_epochs,
                  'checkpoint_sha256': file_digest(directory / 'checkpoint.npz')}
        write_json(destination / 'continuation.json', origin)
        configs.append(config)
        provenance.append(origin)
    requested = {'operation': 'continue_kohonen', 'source': str(source), 'target_epochs': target_epochs}
    plan = create_plan(requested, configs, output)
    write_json(output / 'continuation.json', {'origins': provenance,
               'source_digest_original': original_plan['source_digest'],
               'mode': 'Preservar épocas previas y mantener aprendizaje/radio finales al agotar el decaimiento'})
    return output, plan


def extend(source, target_epochs, output_root='results/kohonen_continuation'):
    output, plan = prepare_extension(source, target_epochs, output_root)
    return _drive(output, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target-epochs', required=True, type=int)
    parser.add_argument('--output', default='results/kohonen_continuation')
    args = parser.parse_args()
    output = extend(args.source, args.target_epochs, args.output)
    status = json.loads((output / 'experiment.json').read_text())
    if status['status'] == 'paused':
        raise SystemExit(130)
    if status['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
