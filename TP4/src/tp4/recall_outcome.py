"""Clasificación del resultado de una recuperación, separada de la dinámica."""
from typing import NamedTuple

import numpy as np


class Outcome(NamedTuple):
    """`category`: correct / other_pattern / inverse / spurious para un estado estable;
    si la recuperación terminó por ciclo o tope, el motivo de parada ('cycle' /
    'max_iterations'), porque sólo un estado estable puede ser estado espurio.
    `hamming`: módulos distintos entre el estado final y el patrón de origen."""
    category: str
    hamming: int


def classify(state, stop_reason, patterns, origin) -> Outcome:
    """Clasifica el estado final respecto del patrón almacenado `patterns[origin]`."""
    patterns = np.asarray(patterns)
    hamming = int(np.sum(state != patterns[origin]))
    if stop_reason != 'stable':
        return Outcome(stop_reason, hamming)
    if hamming == 0:
        return Outcome('correct', 0)
    if (patterns == state).all(axis=1).any():
        return Outcome('other_pattern', hamming)
    if (patterns == -state).all(axis=1).any():
        return Outcome('inverse', hamming)
    return Outcome('spurious', hamming)
