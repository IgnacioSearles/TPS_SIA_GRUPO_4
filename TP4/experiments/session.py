"""Plan persistente, control de interrupciones y verificación al reanudar."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import signal
import threading

import numpy as np

from src.tp4.export import write_json
from .configuration import resolve

ROOT = Path(__file__).resolve().parents[1]


class ExperimentInterrupted(Exception):
    """Pausa solicitada por el usuario; no es una corrida fallida."""


@dataclass
class StopControl:
    requested: bool = False

    def check(self):
        if self.requested:
            raise ExperimentInterrupted()


@contextmanager
def handle_interrupts(control):
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        if control.requested:
            raise KeyboardInterrupt()
        control.requested = True
        print('\nPausa solicitada: guardando al finalizar la época actual. Ctrl+C otra vez: corte inmediato.', flush=True)

    signal.signal(signal.SIGINT, request_stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest():
    files = [ROOT / 'main.py']
    for folder in ('src/tp4', 'experiments'):
        files.extend(sorted((ROOT / folder).glob('*.py')))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_versions():
    return {'numpy': np.__version__, 'python': platform.python_version()}


def create_plan(config, configs, output):
    resolved = [resolve(candidate) for candidate in configs]
    for candidate in resolved:
        candidate['data'] = str(Path(candidate['data']).resolve())
        candidate['output'] = str(output.resolve())
    plan = {
        'version': 1, 'requested': config, 'configs': resolved,
        'datasets': {path: file_digest(path) for path in sorted({candidate['data'] for candidate in resolved})},
        'source_digest': source_digest(), 'runtime': runtime_versions(),
    }
    write_json(output / 'plan.json', plan)
    return plan


def load_plan(output):
    path = output / 'plan.json'
    if not path.exists():
        raise ValueError('No hay plan reanudable: los experimentos anteriores a esta función no lo incluyen')
    plan = json.loads(path.read_text())
    if plan['version'] != 1:
        raise ValueError('Versión de plan incompatible')
    if plan['source_digest'] != source_digest() or plan['runtime'] != runtime_versions():
        raise ValueError('Cambió el código o la versión de Python/NumPy; no se reanuda el entrenamiento')
    for path, expected in plan['datasets'].items():
        if file_digest(path) != expected:
            raise ValueError(f'El catálogo cambió desde el inicio del experimento: {path}')
    return plan
