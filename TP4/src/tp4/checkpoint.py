"""Estado de entrenamiento portable; guarda únicamente la última época completa."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .results import KohonenEpoch, OjaEpoch
from .storage import atomic_file


@dataclass
class TrainingCheckpoint:
    algorithm: str
    parameters: dict
    sample_digest: str
    epoch: int
    weights: np.ndarray
    rng_state: dict
    history: list[KohonenEpoch] | list[OjaEpoch]
    stopped: bool = False
    mean: np.ndarray | None = None
    best_weights: np.ndarray | None = None
    best_error: float = float('inf')
    best_epoch: int = 0
    patience_reference: float = float('inf')
    bad_epochs: int = 0
    stable_epochs: int = 0

    def validate(self, algorithm, parameters, samples):
        if self.algorithm != algorithm or self.parameters != parameters:
            raise ValueError('El checkpoint pertenece a otro modelo o configuración')
        if self.sample_digest != sample_digest(samples):
            raise ValueError('Los datos transformados no coinciden con el checkpoint')
        if self.epoch != len(self.history) or self.epoch < 1:
            raise ValueError('Historial de checkpoint inconsistente')


def sample_digest(samples):
    contiguous = np.ascontiguousarray(samples, dtype=float)
    digest = hashlib.sha256(str(contiguous.shape).encode())
    digest.update(memoryview(contiguous).cast('B'))
    return digest.hexdigest()


def save_checkpoint(path, checkpoint: TrainingCheckpoint):
    """Un NPZ sin pickle; reemplazo atómico del checkpoint anterior."""
    metadata = {key: value for key, value in vars(checkpoint).items()
                if key not in ('weights', 'mean', 'best_weights', 'history')}
    metadata['history'] = [asdict(epoch) for epoch in checkpoint.history]
    metadata['version'] = 1
    # Los infinitos representan referencias aún no inicializadas.
    for key in ('best_error', 'patience_reference'):
        if not np.isfinite(metadata[key]):
            metadata[key] = None
    arrays = {'weights': checkpoint.weights, 'metadata': json.dumps(metadata, allow_nan=False).encode('utf-8')}
    for name in ('mean', 'best_weights'):
        value = getattr(checkpoint, name)
        if value is not None:
            arrays[name] = value
    with atomic_file(path, binary=True) as stream:
        np.savez_compressed(stream, **arrays)


def load_checkpoint(path) -> TrainingCheckpoint:
    with np.load(Path(path), allow_pickle=False) as arrays:
        metadata = json.loads(arrays['metadata'].item().decode('utf-8'))
        if metadata.pop('version') != 1:
            raise ValueError('Versión de checkpoint incompatible')
        epoch_type = {'kohonen': KohonenEpoch, 'oja': OjaEpoch}[metadata['algorithm']]
        metadata['history'] = [epoch_type(**epoch) for epoch in metadata['history']]
        for key in ('best_error', 'patience_reference'):
            if metadata[key] is None:
                metadata[key] = float('inf')
        return TrainingCheckpoint(
            **metadata, weights=arrays['weights'].copy(),
            mean=arrays['mean'].copy() if 'mean' in arrays else None,
            best_weights=arrays['best_weights'].copy() if 'best_weights' in arrays else None,
        )
