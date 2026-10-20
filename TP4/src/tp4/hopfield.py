"""Red de Hopfield con la regla de Hebb del apunte (Unidad 9) y, como comparación, la
regla de la pseudoinversa (no está en el apunte).

Recibe patrones de ±1 como matrices NumPy; no sabe de archivos, imágenes ni gráficos.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

StopReason = Literal['stable', 'cycle', 'max_iterations']
LearningRule = Literal['hebb', 'pseudoinverse']


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

        La demostración del apunte (H nunca sube) supone un cambio por vez; la
        actualización sincrónica cambia varias neuronas juntas y H puede subir. La
        tolerancia descarta diferencias de redondeo entre energías iguales.
        """
        return int(np.sum(np.diff(self.energies) > 1e-9))


def bipolar(x, ndim):
    x = np.asarray(x, dtype=float)
    if x.ndim != ndim or not x.size or not np.isin(x, (-1, 1)).all():
        raise ValueError('Se requieren valores ±1 con las dimensiones correctas')
    return x


class Hopfield:
    def __init__(self, patterns, rule: LearningRule = 'hebb'):
        """Almacena los patrones (p × N), con diagonal cero en las dos reglas.

        Hebb: W = (1/N)·Σ ξξᵀ. Pseudoinversa: W = Ξᵀ(ΞΞᵀ)⁻¹Ξ, con Ξ de p × N, que
        proyecta sobre el espacio de los patrones: deja estable a cada uno aunque
        estén correlacionados, y con patrones ortogonales (ΞΞᵀ = N·I) es igual a Hebb.
        """
        self.patterns_ = bipolar(patterns, 2)
        neurons = self.patterns_.shape[1]
        if rule == 'hebb':
            self.weights_ = self.patterns_.T @ self.patterns_ / neurons
        elif rule == 'pseudoinverse':
            gram = self.patterns_ @ self.patterns_.T
            self.weights_ = self.patterns_.T @ np.linalg.pinv(gram) @ self.patterns_
        else:
            raise ValueError(f'Regla de aprendizaje desconocida: {rule}')
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

    def recall(self, query, max_iterations=100, return_states=False) -> Recall:
        """Itera sincrónicamente desde el patrón de consulta hasta un estado estable,
        un ciclo o el tope.

        Cada paso se clasifica en ese orden: si no cambia el estado es estable; si
        vuelve a un estado ya visitado es ciclo, también cuando llega en el paso
        que se calcula ya en el tope (ese paso no se aplica: la red no pasa de
        `max_iterations`); si no, recién ahí el tope corta la recuperación.

        `return_states` agrega al resultado la secuencia de estados.
        """
        state = bipolar(query, 1)
        if len(state) != len(self.weights_):
            raise ValueError('El patrón de consulta no tiene N elementos')
        visited = {state.tobytes()}
        states, energies = [state], [self.energy(state)]

        def result(stop_reason: StopReason) -> Recall:
            return Recall(states[-1], len(states) - 1, tuple(energies), stop_reason,
                          tuple(states) if return_states else None)

        while True:
            new_state = self._synchronous(state)
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

    def _synchronous(self, state):
        """S(t+1) = sign(W·S(t)); con campo cero la neurona conserva su estado."""
        fields = self.weights_ @ state
        return np.where(fields > 0, 1., np.where(fields < 0, -1., state))
