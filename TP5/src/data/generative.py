"""Datos de prueba y metadata para experimentos de generación sin notas omitidas."""

from dataclasses import dataclass
import hashlib

import numpy as np

from src.options import ConfigurationError, Options


@dataclass(frozen=True)
class GenerativeStudy:
    test_samples: np.ndarray
    test_labels: tuple[str, ...]
    train_records: tuple[dict, ...]
    test_records: tuple[dict, ...]
    metadata: dict
    test_waveforms: np.ndarray | None = None


def positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{name} debe ser un entero ≥ {minimum}")
    return value


def split_options(options: Options):
    train = positive_integer(options.take("train_count", 1024), "train_count", 2)
    test = positive_integer(options.take("test_count", 256), "test_count")
    seed = options.take("data_seed", 20261113)
    if type(seed) is not int or seed < 0:
        raise ConfigurationError("data_seed debe ser un entero no negativo")
    return train, test, seed


def assemble_generative(samples, records, train_count, image_shape, metadata,
                        codec=None, test_waveforms=None):
    """Sólo las primeras filas se entregan al Trainer; las restantes son evaluación."""
    from src.data.dataset import Dataset

    samples = np.asarray(samples, dtype=np.float32)
    train_fingerprints = {hashlib.sha256(row.tobytes()).digest() for row in samples[:train_count]}
    if any(hashlib.sha256(row.tobytes()).digest() in train_fingerprints for row in samples[train_count:]):
        raise ConfigurationError("Entrenamiento y prueba contienen imágenes idénticas")
    metadata = {"version": 1, **metadata,
                "train_sha256": hashlib.sha256(samples[:train_count].astype("<f4").tobytes()).hexdigest(),
                "test_sha256": hashlib.sha256(samples[train_count:].astype("<f4").tobytes()).hexdigest()}
    labels = [record["label"] for record in records]
    if len(set(labels)) != len(labels) or len(records) != len(samples):
        raise ConfigurationError("Las muestras generativas requieren etiquetas únicas y metadata alineada")
    return Dataset(samples=samples[:train_count], labels=labels[:train_count],
                   groups=[r["family"] for r in records[:train_count]],
                   image_shape=image_shape, is_binary=False, audio_codec=codec,
                   generative_study=GenerativeStudy(samples[train_count:], tuple(labels[train_count:]),
                       tuple(records[:train_count]), tuple(records[train_count:]), metadata, test_waveforms))
