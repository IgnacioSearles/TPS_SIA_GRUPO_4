"""Patrones de consulta: patrones almacenados con ruido aleatorio por módulo."""
from typing import NamedTuple

import numpy as np


class Query(NamedTuple):
    """Un patrón de consulta y de dónde salió: el índice del patrón almacenado
    (`origin`), el nivel de ruido y la semilla de ruido."""
    origin: int
    noise: float
    seed: int
    pattern: np.ndarray


def noisy_query(pattern, noise, rng):
    """Invierte cada módulo, de forma independiente, con probabilidad `noise`."""
    flips = rng.random(len(pattern)) < noise
    return np.where(flips, -pattern, pattern)


def make_queries(patterns, noise_levels, seeds) -> list[Query]:
    """Un patrón de consulta por patrón almacenado, nivel de ruido y semilla.

    El generador de cada uno se siembra con (índice del patrón, índice del nivel
    de ruido, semilla), así cada consulta se puede rehacer a partir de esos datos.
    """
    return [Query(origin, noise, seed,
                  noisy_query(pattern, noise, np.random.default_rng((origin, level, seed))))
            for origin, pattern in enumerate(patterns)
            for level, noise in enumerate(noise_levels)
            for seed in seeds]
