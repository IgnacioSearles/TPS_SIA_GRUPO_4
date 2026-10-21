"""Coordina entrenamiento, aislamiento de fallos y persistencia de resultados."""
from dataclasses import asdict
from datetime import datetime
from itertools import count
from pathlib import Path
import traceback
import json

import numpy as np
import pandas as pd

from src.tp4.checkpoint import load_checkpoint
from src.tp4.config import KohonenConfig, OjaConfig
from src.tp4.export import export_model, write_json
from src.tp4.kohonen import Kohonen
from src.tp4.oja import Oja
from src.tp4.preprocessing import PreprocessingConfig, Preprocessor
from .configuration import expand, resolve
from .reporting import group_id, write_reports
from .results import ExecutionResult, RunFailure, RunRecord, restore_record
from .session import ExperimentInterrupted, StopControl, create_plan, handle_interrupts, load_plan
from .progress import epoch_callback, export_partial_history
from src.tp4.results import RunMetrics


_MODEL_TYPES = {
    'kohonen': (KohonenConfig, Kohonen),
    'oja': (OjaConfig, Oja),
}


def create_model(algorithm, parameters):
    config_type, model_type = _MODEL_TYPES[algorithm]
    return model_type(config_type(**parameters))


def prepare_data(resolved, output):
    frame = pd.read_csv(resolved['data'])
    if frame.isna().any().any():
        raise ValueError('El catálogo contiene faltantes')
    state_path = output / 'preprocessing.json'
    if (output / 'checkpoint.npz').exists() and state_path.exists():
        preprocessing = Preprocessor.from_state(json.loads(state_path.read_text()))
        samples = preprocessing.transform(frame)
    else:
        preprocessing = Preprocessor(PreprocessingConfig(**resolved['preprocessing']))
        samples = preprocessing.fit_transform(frame)
    write_json(state_path, preprocessing.state())
    return frame, preprocessing, samples


def collect_metrics(model, samples, algorithm) -> RunMetrics:
    metrics = RunMetrics(
        evaluation=model.evaluate(samples),
        epochs=model.epochs_,
        updates=model.updates_,
        stop_reason=model.stop_reason_,
        features=samples.shape[1],
        selected_epoch=model.selected_epoch_ if algorithm == 'kohonen' else None,
        best_epoch=model.best_epoch_ if algorithm == 'kohonen' else None,
    )
    numbers = (value for value in metrics.to_dict().values() if isinstance(value, (float, int)))
    if not all(np.isfinite(value) for value in numbers):
        raise FloatingPointError('Métricas no finitas')
    return metrics


def execute(config, output, control=None) -> ExecutionResult:
    """Una corrida; el coordinador registra las excepciones y continúa."""
    resolved = resolve(config)
    write_json(output / 'config.json', resolved)
    frame, preprocessing, samples = prepare_data(resolved, output)
    model = create_model(resolved['algorithm'], resolved['model'])
    checkpoint_path = output / 'checkpoint.npz'
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    model.fit(samples, checkpoint=checkpoint, on_epoch=epoch_callback(output, control))
    metrics = collect_metrics(model, samples, resolved['algorithm'])
    export_model(output, model, preprocessing, frame, samples, resolved['algorithm'])
    write_json(output / 'metrics.json', metrics.to_dict())
    return ExecutionResult(resolved, metrics)


def create_output_directory(root, name):
    """Crea root/<name>_<fecha>_<hora> en hora local; agrega _2, _3... si ya existe."""
    if not name or any(separator in name for separator in '/\\'):
        raise ValueError(f'Nombre de experimento inválido: {name!r}')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    base = f'{name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    for attempt in count(1):
        directory = root / (base if attempt == 1 else f'{base}_{attempt}')
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            continue


def execute_record(candidate, output, index, control=None) -> RunRecord:
    resolved = resolve(candidate)
    directory = output / f'run_{index:04d}'
    directory.mkdir(exist_ok=True)
    (directory / 'record.json').unlink(missing_ok=True)
    group = group_id(resolved)
    seed = resolved['model']['seed']
    try:
        result = execute(candidate, directory, control=control)
        record = RunRecord(directory.name, group, seed, metrics=result.metrics)
        (directory / 'error.json').unlink(missing_ok=True)
    except ExperimentInterrupted:
        raise
    except Exception as error:
        failure = RunFailure(type(error).__name__, str(error), traceback.format_exc())
        write_json(directory / 'error.json', asdict(failure))
        record = RunRecord(directory.name, group, seed, failure=failure)
    write_json(directory / 'record.json', asdict(record))
    return record


def existing_records(output, configs, retry_failed=False):
    records = {}
    for index, candidate in enumerate(configs):
        path = output / f'run_{index:04d}' / 'record.json'
        if not path.exists():
            continue
        record = restore_record(json.loads(path.read_text()))
        if record.group != group_id(candidate) or record.seed != candidate['model']['seed']:
            raise ValueError(f'El registro no corresponde al plan: {path}')
        if record.status == 'ok' or not retry_failed:
            records[index] = record
    return records


def _save_progress(output, plan, records, status, interrupted_run=None):
    ordered = [record for _, record in sorted(records.items())]
    write_reports(output, plan['requested'], ordered, status=status,
                  total=len(plan['configs']), interrupted_run=interrupted_run)


def _run_pending(output, plan, records, control):
    for index, candidate in enumerate(plan['configs']):
        if index in records:
            continue
        control.check()
        record = execute_record(candidate, output, index, control)
        records[index] = record
        _save_progress(output, plan, records, 'running')
        print(f'[{index + 1}/{len(plan["configs"])}] {record.run}: {record.status}', flush=True)


def _drive(output, plan, retry_failed=False):
    records = existing_records(output, plan['configs'], retry_failed)
    control = StopControl()
    print(f'Resultados: {output}', flush=True)
    print(f'Corridas pendientes: {len(plan["configs"]) - len(records)}', flush=True)
    _save_progress(output, plan, records, 'running')
    with handle_interrupts(control):
        try:
            _run_pending(output, plan, records, control)
        except (ExperimentInterrupted, KeyboardInterrupt):
            pending = next((index for index in range(len(plan['configs'])) if index not in records), None)
            interrupted = f'run_{pending:04d}' if pending is not None else None
            _save_progress(output, plan, records, 'paused', interrupted)
            if interrupted is not None:
                export_partial_history(output / interrupted)
            print(f'Experimento pausado. Continuar: python3 main.py resume --experiment {output}', flush=True)
            return output
    _save_progress(output, plan, records, 'completed')
    return output


def run(config, experiment=False, dry_run=False, name=None):
    """name identifica el directorio de resultados; por defecto, el algoritmo."""
    configs = expand(config) if experiment else [resolve(config)]
    print(f'Corridas: {len(configs)}', flush=True)
    if dry_run:
        return configs
    output = create_output_directory(config.get('output', 'results'), name or config['algorithm'])
    plan = create_plan(config, configs, output)
    return _drive(output, plan)


def resume(output, *, retry_failed=False, dry_run=False):
    output = Path(output).resolve()
    plan = load_plan(output)
    records = existing_records(output, plan['configs'], retry_failed)
    if dry_run:
        print(f'Corridas pendientes: {len(plan["configs"]) - len(records)}', flush=True)
        return output
    return _drive(output, plan, retry_failed)
