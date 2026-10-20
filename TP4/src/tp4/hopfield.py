"""Red de Hopfield con la regla de Hebb del apunte (Unidad 9).

Recibe patrones de ±1 como matrices NumPy; no sabe de archivos, imágenes ni gráficos.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

StopReason = Literal['stable', 'cycle', 'max_iterations']
UpdateMode = Literal['synchronous', 'asynchronous']


@dataclass(frozen=True)
class Recall:
    """Resultado de una recuperación.

    `iterations` cuenta las iteraciones que cambiaron el estado: el recálculo que
    confirma un estado estable no suma. `energies` tiene la energía del estado
    inicial seguida de la de cada iteración (largo `iterations + 1`). `states`, sólo
    si se pidió, tiene el patrón de consulta seguido del estado tras cada iteración
    (mismo largo que `energies`); si no, es None.
    """
    state: np.ndarray
    iterations: int
    energies: tuple[float, ...]
    stop_reason: StopReason
    states: tuple[np.ndarray, ...] | None = None

    @property
    def energy_rises(self) -> int:
        """Iteraciones en las que la energía subió respecto de la anterior.

        La demostración del apunte (H nunca sube) supone un cambio por vez; en
        modo sincrónico cambian varias neuronas juntas y H puede subir. La
        tolerancia descarta diferencias de redondeo entre energías iguales.
        """
        return int(np.sum(np.diff(self.energies) > 1e-9))


def bipolar(x, ndim):
    x = np.asarray(x, dtype=float)
    if x.ndim != ndim or not x.size or not np.isin(x, (-1, 1)).all():
        raise ValueError('Se requieren valores ±1 con las dimensiones correctas')
    return x


class Hopfield:
    def __init__(self, patterns):
        """Almacena los patrones (p × N) con Hebb: W = (1/N)·Σ ξξᵀ, diagonal cero."""
        self.patterns_ = bipolar(patterns, 2)
        neurons = self.patterns_.shape[1]
        self.weights_ = self.patterns_.T @ self.patterns_ / neurons
        np.fill_diagonal(self.weights_, 0)

    def overlaps(self):
        """Matriz p × p de solapamientos: m^μν = (1/N)·Σⱼ ξⱼ^μ ξⱼ^ν."""
        return self.patterns_ @ self.patterns_.T / self.patterns_.shape[1]

    def crosstalk(self, index):
        """Término de crosstalk del apunte para el patrón almacenado `index` (ν), por neurona.

        Cᵢ^ν = (1/N)·Σⱼ Σ_{μ≠ν} ξᵢ^μ ξⱼ^μ ξⱼ^ν = Σ_{μ≠ν} m^μν ξᵢ^μ, de modo que el
        desarrollo del apunte queda hᵢ^ν = ξᵢ^ν + Cᵢ^ν.
        """
        others = np.delete(np.arange(len(self.patterns_)), index)
        return self.overlaps()[others, index] @ self.patterns_[others]

    def energy(self, state):
        """H = −½·Σᵢⱼ wᵢⱼ Sᵢ Sⱼ."""
        return float(-.5 * state @ self.weights_ @ state)

    def recall(self, query, mode: UpdateMode = 'synchronous', max_iterations=100, seed=0,
               return_states=False) -> Recall:
        """Itera desde el patrón de consulta hasta un estado estable, un ciclo o el tope.

        Cada paso se clasifica en ese orden: si no cambia el estado es estable; si
        vuelve a un estado ya visitado es ciclo, también cuando llega en el paso
        que se calcula ya en el tope (ese paso no se aplica: la red no pasa de
        `max_iterations`); si no, recién ahí el tope corta la recuperación.

        `seed` siembra el generador local de los órdenes del modo asincrónico (acepta
        lo mismo que `np.random.default_rng`); el modo sincrónico no la usa.
        `return_states` agrega al resultado la secuencia de estados.
        """
        state = bipolar(query, 1)
        if len(state) != len(self.weights_):
            raise ValueError('El patrón de consulta no tiene N elementos')
        step = self._updater(mode, seed)
        visited = {state.tobytes()}
        states, energies = [state], [self.energy(state)]

        def result(stop_reason: StopReason) -> Recall:
            return Recall(states[-1], len(states) - 1, tuple(energies), stop_reason,
                          tuple(states) if return_states else None)

        while True:
            new_state = step(state)
            if np.array_equal(new_state, state):
                return result('stable')
            revisits = new_state.tobytes() in visited
            if len(states) - 1 == max_iterations:
                return result('cycle' if revisits else 'max_iterations')
            state = new_state
            states.append(state)
            energies.append(self.energy(state))
            if revisits:
                return result('cycle')
            visited.add(state.tobytes())

    def _updater(self, mode, seed):
        """Devuelve la función que hace una iteración en el modo pedido."""
        if mode == 'synchronous':
            return self._synchronous
        if mode == 'asynchronous':
            rng = np.random.default_rng(seed)
            return lambda state: self._asynchronous(state, rng)
        raise ValueError(f'Modo de actualización desconocido: {mode}')

    def _synchronous(self, state):
        """S(t+1) = sign(W·S(t)); con campo cero la neurona conserva su estado."""
        fields = self.weights_ @ state
        return np.where(fields > 0, 1., np.where(fields < 0, -1., state))

    def _asynchronous(self, state, rng):
        """Una pasada por las N neuronas en un orden aleatorio nuevo: cada neurona
        usa los estados ya actualizados en esta pasada. Campo cero → conserva."""
        state = state.copy()
        for i in rng.permutation(len(state)):
            field = self.weights_[i] @ state
            if field:
                state[i] = 1. if field > 0 else -1.
        return state
