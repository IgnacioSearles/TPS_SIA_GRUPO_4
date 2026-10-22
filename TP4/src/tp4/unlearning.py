"""Desaprendizaje antihebbiano sobre una única red de Hebb.

No cambia la recuperación sincrónica de Hopfield. Sólo penaliza estados estables
espurios encontrados durante el entrenamiento, sin usar consultas de evaluación.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .hopfield import Hopfield, bipolar

UnlearningMode = Literal['directed', 'dream']


@dataclass(frozen=True)
class UnlearningStep:
    attempt: int
    updates: int
    action: str
    recall_iterations: int
    energy_before: float
    energy_after: float


def anti_hebbian_update(network: Hopfield, state, learning_rate: float):
    """W <- W - eta/N * aa.T, conservando diagonal cero.

    Modifica la red recibida. El llamador decide qué estado penalizar; el entrenador
    de abajo excluye recuerdos, sus inversos y salidas que no sean estables.
    Para un estado bipolar a, su energía aumenta exactamente eta*(N-1)/2.
    """
    state = bipolar(state, 1)
    if state.size != network.weights_.shape[0]:
        raise ValueError('El estado a desaprender no tiene N elementos')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate debe ser finito y positivo')
    network.weights_ -= learning_rate / state.size * np.outer(state, state)
    np.fill_diagonal(network.weights_, 0)


class UnlearningTrainer:
    """Entrenamiento reproducible desde Hebb, independiente de las consultas.

    directed: inicia cada intento en la mezcla mayoritaria original y penaliza el
    atractor espurio actual al que llega. dream: inicia desde un vector uniforme
    aleatorio de ±1. Un ciclo o tope consume un intento, pero no modifica pesos.
    Tampoco modifica pesos si llega a un recuerdo o su inverso. Protegerlos como
    objetivos NO garantiza que otros updates no los desestabilicen.
    """
    def __init__(self, patterns, *, mode: UnlearningMode = 'directed',
                 learning_rate: float = .05, seed: int = 0, max_iterations: int = 100):
        if mode not in ('directed', 'dream'):
            raise ValueError('mode debe ser directed o dream')
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate debe ser finito y positivo')
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
            raise ValueError('max_iterations debe ser un entero positivo')
        self.network = Hopfield(bipolar(patterns, 2).copy())
        self.mode = mode
        self.learning_rate = float(learning_rate)
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)
        total = self.network.patterns_.sum(axis=0)
        # Para conjuntos pares, los empates de mayoría usan el primer recuerdo.
        self.mixture = np.where(total > 0, 1.,
                                np.where(total < 0, -1., self.network.patterns_[0]))
        self.attempts = 0
        self.updates = 0

    def step(self) -> UnlearningStep:
        query = (self.mixture.copy() if self.mode == 'directed' else
                 self.rng.choice([-1., 1.], size=self.mixture.size))
        result = self.network.recall(query, max_iterations=self.max_iterations)
        state = result.state
        before = self.network.energy(state)
        protected = (np.all(self.network.patterns_ == state, axis=1).any() or
                     np.all(self.network.patterns_ == -state, axis=1).any())
        action = result.stop_reason
        if result.stop_reason == 'stable':
            if protected:
                action = 'protected'
            else:
                anti_hebbian_update(self.network, state, self.learning_rate)
                self.updates += 1
                action = 'updated'
        self.attempts += 1
        return UnlearningStep(self.attempts, self.updates, action, result.iterations,
                             before, self.network.energy(state))
