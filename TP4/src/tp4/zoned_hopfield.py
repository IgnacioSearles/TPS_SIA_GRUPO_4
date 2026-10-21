"""Una red de Hopfield por zona del QR: cada red guarda los mismos patrones, recortados a su zona.

La imagen de lado `side` se corta en cuadrados de lado `zone` (los del borde pueden quedar más chicos).
Una zona en la que todos los patrones valen lo mismo no necesita red: se devuelve su valor conocido.
Con `only_differing`, cada red usa sólo los módulos de su zona que difieren en algún patrón, como la
variante sin módulos compartidos pero aplicada zona por zona.

La salida es la unión de las salidas de las zonas. Cada zona puede terminar en un patrón distinto, así
que la salida completa sólo es un patrón almacenado si todas las zonas coinciden en el mismo.
"""
from typing import NamedTuple

import numpy as np

from .hopfield import Hopfield, bipolar


class Zone(NamedTuple):
    """`modules`: índices (en el patrón aplanado) de la zona. `neurons`: los que ve su red; vacío si
    la zona es igual en todos los patrones. `network`: None en ese caso."""
    row: int
    column: int
    modules: np.ndarray
    neurons: np.ndarray
    network: Hopfield | None


class ZonedRecall(NamedTuple):
    """`state`: salida completa. `match`: índice del patrón almacenado igual a `state`, o None.
    `zone_matches`: por zona con red, los patrones almacenados que coinciden con su salida en esa
    zona (puede haber varios si son iguales ahí, o ninguno si cayó en un estado espurio)."""
    state: np.ndarray
    match: int | None
    zone_matches: tuple[frozenset[int], ...]

    @property
    def zones_on_stored(self):
        """Zonas cuya salida es la de algún patrón almacenado."""
        return sum(bool(matches) for matches in self.zone_matches)

    @property
    def consistent(self):
        """Hay un patrón almacenado con el que coinciden todas las zonas."""
        return bool(frozenset.intersection(*self.zone_matches)) if self.zone_matches else True


class ZonedHopfield:
    def __init__(self, patterns, side, zone, only_differing=False):
        self.patterns_ = bipolar(patterns, 2)
        if self.patterns_.shape[1] != side * side:
            raise ValueError(f'Los patrones deben tener {side}×{side} módulos')
        if not 1 <= zone <= side:
            raise ValueError(f'El lado de la zona debe estar entre 1 y {side}')
        self.zones_ = [self._zone(row, column, side, zone, only_differing)
                       for row in range(0, side, zone) for column in range(0, side, zone)]

    def _zone(self, row, column, side, zone, only_differing):
        grid = np.arange(side * side).reshape(side, side)
        modules = grid[row:row + zone, column:column + zone].ravel()
        local = self.patterns_[:, modules]
        differing = ~np.all(local == local[0], axis=0)
        if not differing.any():
            return Zone(row, column, modules, modules[:0], None)
        neurons = modules[differing] if only_differing else modules
        return Zone(row, column, modules, neurons, Hopfield(self.patterns_[:, neurons]))

    @property
    def learned_zones(self):
        return [zone for zone in self.zones_ if zone.network is not None]

    def stable_patterns(self, zone: Zone):
        """Cuántos patrones almacenados son estados estables de la red de la zona."""
        return sum(zone.network.recall(pattern).iterations == 0 for pattern in zone.network.patterns_)

    def recall(self, query, max_iterations=100) -> ZonedRecall:
        query = bipolar(query, 1)
        state = query.copy()
        matches = []
        for zone in self.zones_:
            if zone.network is None:
                state[zone.modules] = self.patterns_[0, zone.modules]
                continue
            state[zone.neurons] = zone.network.recall(query[zone.neurons], max_iterations=max_iterations).state
            # Los módulos de la zona que la red no ve son iguales en todos los patrones: se reponen.
            known = np.setdiff1d(zone.modules, zone.neurons)
            state[known] = self.patterns_[0, known]
            matches.append(frozenset(np.flatnonzero(np.all(self.patterns_[:, zone.neurons] == state[zone.neurons],
                                                           axis=1)).tolist()))
        match = next((index for index, pattern in enumerate(self.patterns_) if np.array_equal(pattern, state)), None)
        return ZonedRecall(state, match, tuple(matches))
