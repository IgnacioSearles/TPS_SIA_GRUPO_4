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
    inicial seguida de la de cada iteración (largo `iterations + 1`).
    """
    state: np.ndarray
    iterations: int
    energies: tuple[float, ...]
    stop_reason: StopReason

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

    def energy(self, state):
        """H = −½·Σᵢⱼ wᵢⱼ Sᵢ Sⱼ."""
        return float(-.5 * state @ self.weights_ @ state)

    def recall(self, query, mode: UpdateMode = 'synchronous', max_iterations=100, seed=0) -> Recall:
        """Itera desde el patrón de consulta hasta un estado estable, un ciclo o el tope.

        `seed` siembra el generador local de los órdenes del modo asincrónico (acepta
        lo mismo que `np.random.default_rng`); el modo sincrónico no la usa.
        """
        state = bipolar(query, 1)
        if len(state) != len(self.weights_):
            raise ValueError('El patrón de consulta no tiene N elementos')
        step = self._update(mode, seed)
        visited = {state.tobytes()}
        energies = [self.energy(state)]
        while True:
            new_state = step(state)
            if np.array_equal(new_state, state):
                return Recall(state, len(energies) - 1, tuple(energies), 'stable')
            if len(energies) - 1 == max_iterations:
                return Recall(state, max_iterations, tuple(energies), 'max_iterations')
            state = new_state
            energies.append(self.energy(state))
            if state.tobytes() in visited:
                return Recall(state, len(energies) - 1, tuple(energies), 'cycle')
            visited.add(state.tobytes())

    def _update(self, mode, seed):
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
