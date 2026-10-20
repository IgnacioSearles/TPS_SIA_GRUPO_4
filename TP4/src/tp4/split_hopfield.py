"""Varias redes de Hopfield, cada una con un grupo de los patrones almacenados.

La consulta pasa por todas las redes. Entre las salidas que son un patrón almacenado
de su propia red se elige la más cercana a la consulta (distancia de Hamming); si
ninguna red devolvió un patrón almacenado, la más cercana entre todas las salidas.
"""
from typing import NamedTuple

import numpy as np

from .hopfield import Hopfield, LearningRule, StopReason


class SplitRecall(NamedTuple):
    """`state`: la salida elegida, completa. `match`: índice del patrón almacenado
    igual a la salida, o None. `stored_outputs`: cuántas redes devolvieron uno de sus
    patrones almacenados. `group`: índice de la red de la que salió `state`.
    `iterations`, `energies` y `stop_reason` son los de esa red."""
    state: np.ndarray
    match: int | None
    stored_outputs: int
    group: int
    iterations: int
    energies: tuple[float, ...]
    stop_reason: StopReason


class SplitHopfield:
    def __init__(self, patterns, groups, shared=None, rule: LearningRule = 'hebb'):
        """`groups`: índices de `patterns` que guarda cada red. Con `shared` (un
        `SharedModules`), cada red trabaja sólo con los módulos que difieren y las
        salidas se completan con los compartidos."""
        self.patterns_ = np.asarray(patterns, dtype=float)
        self.groups_ = [tuple(group) for group in groups]
        self._strip = shared.strip if shared is not None else np.asarray
        self._restore = shared.restore if shared is not None else np.asarray
        self.networks_ = [Hopfield(self._strip(self.patterns_[list(group)]), rule=rule) for group in self.groups_]

    def recall(self, query, max_iterations=100) -> SplitRecall:
        outputs = []
        for index, (group, network) in enumerate(zip(self.groups_, self.networks_)):
            result = network.recall(self._strip(query), max_iterations=max_iterations)
            state = self._restore(result.state)
            match = next((i for i in group if np.array_equal(self.patterns_[i], state)), None)
            outputs.append((state, match, index, result))
        stored = [output for output in outputs if output[1] is not None]
        state, match, group, result = min(stored or outputs, key=lambda output: np.sum(output[0] != query))
        return SplitRecall(state, match, len(stored), group, result.iterations, result.energies, result.stop_reason)
