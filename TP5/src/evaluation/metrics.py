"""Métricas de calidad de reconstrucción."""

from __future__ import annotations

import numpy as np

from src.data.dataset import Dataset
from src.models.autoencoder import Autoencoder

Array = np.ndarray
BINARY_THRESHOLD = 0.5
# Meta de la consigna: a lo sumo un píxel incorrecto por patrón.
PIXEL_ERROR_TARGET = 1


def binarize(reconstructions: Array) -> Array:
    return (reconstructions >= BINARY_THRESHOLD).astype(float)


def pixel_errors(targets: Array, reconstructions: Array) -> Array:
    """Cantidad de píxeles incorrectos de cada muestra tras binarizar."""
    return np.count_nonzero(targets != binarize(reconstructions), axis=1)


def squared_errors(targets: Array, reconstructions: Array) -> Array:
    """Error cuadrático medio de cada muestra."""
    return np.mean((targets - reconstructions) ** 2, axis=1)


def sample_errors(dataset: Dataset, targets: Array, reconstructions: Array) -> Array:
    """Error por muestra en la unidad natural del dataset (píxeles o MSE)."""
    if dataset.is_binary:
        return pixel_errors(targets, reconstructions)
    return squared_errors(targets, reconstructions)


def error_unit(dataset: Dataset) -> str:
    return "píxeles incorrectos" if dataset.is_binary else "error cuadrático medio"


def reconstruction_quality(model: Autoencoder, dataset: Dataset) -> dict[str, float]:
    """Resumen de qué tan bien se reconstruyen los datos limpios."""
    reconstructions = model.reconstruct(dataset.samples)
    if not dataset.is_binary:
        return {
            "mean_squared_error": float(
                squared_errors(dataset.samples, reconstructions).mean()
            )
        }
    errors = pixel_errors(dataset.samples, reconstructions)
    return {
        "max_pixel_errors": int(errors.max()),
        "mean_pixel_errors": float(errors.mean()),
        "patterns_within_target": int(np.count_nonzero(errors <= PIXEL_ERROR_TARGET)),
    }
