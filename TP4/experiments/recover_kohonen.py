"""Reconstruye un plan actual desde artefactos validados, sin entrenar ni sobrescribir el origen."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

from src.tp4.checkpoint import save_checkpoint
from src.tp4.config import KohonenConfig
from src.tp4.export import write_json
from .configuration import resolve
from .extend_kohonen import completed_runs, validate_origin
from .reporting import group_id, write_reports
from .results import RunRecord, restore_record
from .session import create_plan, file_digest, runtime_versions


def validate_source(source):
    plan = json.loads((source / 'plan.json').read_text())
    if plan['runtime'] != runtime_versions():
        raise ValueError('Python/NumPy no coinciden con el plan original')
    for path, expected in plan['datasets'].items():
        if file_digest(path) != expected:
            raise ValueError(f'El catálogo cambió: {path}')
    if any(config['algorithm'] != 'kohonen' for config in plan['configs']):
        raise ValueError('La recuperación está implementada para planes de Kohonen')
    return plan


def target_config(original, target_epochs):
    config = deepcopy(original)
    if target_epochs < config['model']['epochs']:
        raise ValueError('No se reduce el presupuesto de una corrida existente')
    schedule = config['model'].get('schedule_epochs') or config['model']['epochs']
    config['model'].update(epochs=target_epochs, schedule_epochs=schedule)
    return resolve(config)


def config_key(config):
    return group_id(config), config['model']['seed']


def completed_candidates(sources, expected_keys, cache):
    candidates = {}
    for source in sources:
        _, runs = completed_runs(source)
        for directory, config in runs:
            config = resolve(config)
            key = config_key(config)
            if key not in expected_keys:
                raise ValueError(f'La corrida completada no corresponde al objetivo: {directory}')
            checkpoint, _ = validate_origin(directory, config, cache)
            if checkpoint.epoch != config['model']['epochs']:
                raise ValueError(f'La corrida no completó el presupuesto anunciado: {directory}')
            for filename in ('model.npz', 'metrics.json', 'history.csv'):
                if not (directory / filename).exists():
                    raise ValueError(f'Falta un artefacto de la corrida completada: {directory / filename}')
            candidates.setdefault(key, directory)
    return candidates


def validate_pending(source, original_configs, targets, completed, cache):
    pending = {}
    for index, (original, target) in enumerate(zip(original_configs, targets)):
        if config_key(target) in completed:
            continue
        directory = source / f'run_{index:04d}'
        if not (directory / 'checkpoint.npz').exists():
            continue
        checkpoint, preprocessing = validate_origin(directory, original, cache)
        if checkpoint.epoch > target['model']['epochs']:
            raise ValueError('El checkpoint supera el nuevo presupuesto')
        pending[index] = (directory, checkpoint, preprocessing)
    return pending


def import_completed(origin, destination, config):
    shutil.copytree(origin, destination)
    old = restore_record(json.loads((destination / 'record.json').read_text()))
    record = RunRecord(destination.name, group_id(config), config['model']['seed'], metrics=old.metrics)
    write_json(destination / 'record.json', asdict(record))
    write_json(destination / 'config.json', config)
    return record


def import_pending(origin, destination, checkpoint, preprocessing, config):
    destination.mkdir()
    parameters = {**asdict(KohonenConfig(**config['model'])), 'layout': config['model']['topology']}
    save_checkpoint(destination / 'checkpoint.npz', replace(checkpoint, parameters=parameters))
    write_json(destination / 'preprocessing.json', preprocessing)
    write_json(destination / 'config.json', config)


def materialize(output, configs, completed, pending):
    records, provenance = [], []
    for index, config in enumerate(configs):
        destination = output / f'run_{index:04d}'
        origin = completed.get(config_key(config))
        if origin is not None:
            records.append(import_completed(origin, destination, config))
            epoch = records[-1].metrics.epochs
            kind = 'completed'
        elif index in pending:
            origin, checkpoint, preprocessing = pending[index]
            import_pending(origin, destination, checkpoint, preprocessing, config)
            epoch, kind = checkpoint.epoch, 'checkpoint'
        else:
            continue
        details = {'source_run': str(origin), 'source_epoch': epoch, 'kind': kind,
                   'source_checkpoint_sha256': file_digest(origin / 'checkpoint.npz')}
        write_json(destination / 'recovery.json', details)
        provenance.append({'run': destination.name, **details})
    return records, provenance


def prepare_recovery(source, output, target_epochs, completed_from=()):
    source, output = Path(source).resolve(), Path(output).resolve()
    plan = validate_source(source)
    targets = [target_config(config, target_epochs) for config in plan['configs']]
    cache = {}
    completed = completed_candidates([Path(path).resolve() for path in completed_from],
                                     {config_key(config) for config in targets}, cache)
    pending = validate_pending(source, plan['configs'], targets, completed, cache)
    for config in targets:
        config['output'] = str(output)
    output.mkdir(parents=True, exist_ok=False)
    records, provenance = materialize(output, targets, completed, pending)
    requested = {'operation': 'recover_kohonen', 'source': str(source),
                 'target_epochs': target_epochs, 'completed_from': [str(path) for path in completed_from]}
    create_plan(requested, targets, output)
    write_json(output / 'recovery.json', {'source_digest_original': plan['source_digest'],
               'validation': 'Runtime, catálogo, parámetros, digest transformado e historial de checkpoints',
               'imported': provenance})
    write_reports(output, requested, records, status='paused', total=len(targets))
    print(f'Plan preparado: {output}')
    print(f'{len(records)} completas reutilizadas; {len(pending)} checkpoints pendientes recuperados; '
          f'{len(targets) - len(records) - len(pending)} sin iniciar')
    print(f'Continuar: python3 main.py resume --experiment {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target-epochs', required=True, type=int)
    parser.add_argument('--completed-from', nargs='*', default=[])
    args = parser.parse_args()
    prepare_recovery(args.source, args.output, args.target_epochs, args.completed_from)


if __name__ == '__main__':
    main()
