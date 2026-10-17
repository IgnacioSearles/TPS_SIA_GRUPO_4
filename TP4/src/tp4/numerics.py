"""Operaciones vectorizadas compartidas."""
import numpy as np

def matrix(x, features=None):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError('Se requiere una matriz no vacía de valores finitos')
    if features is not None and x.shape[1] != features:
        raise ValueError('Cantidad de variables incompatible')
    return x

def distances(x, w):
    """Distancias euclídeas sin tensor muestras × neuronas × variables."""
    squared = (x*x).sum(axis=1)[:, None] + (w*w).sum(axis=1)[None, :] - 2*x@w.T
    return np.sqrt(np.maximum(squared, 0))

def schedule(start, end, progress, kind):
    if kind == 'constant':
        return start
    if kind == 'linear':
        return start + (end-start)*progress
    return start * (end/start)**progress

class Patience:
    def __init__(self, config):
        self.config = config
        self.reference = float('inf')
        self.bad_epochs = 0

    def update(self, value, epoch):
        if self.reference - value > self.config.min_delta:
            self.reference = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return (self.config.enabled and epoch >= self.config.min_epochs
                and self.bad_epochs >= self.config.patience)
