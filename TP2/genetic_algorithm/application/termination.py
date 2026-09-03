"""Implementaciones de condiciones de corte del algoritmo genético."""

from __future__ import annotations

from typing import Any

from genetic_algorithm.application.contracts import TerminationCondition
from genetic_algorithm.domain.contracts import EvolutionContext, EvolutionState, Fitness, Individual, FitnessComparator


class MaxGenerationsTermination[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    TerminationCondition[IndividualT, FitnessT]
):
    """Corta la ejecución al alcanzar un límite de generaciones."""

    def __init__(self, max_generations: int) -> None:
        if max_generations < 0:
            raise ValueError("max_generations must be non-negative")
        self._max_generations = max_generations

    def should_stop(
        self, state: EvolutionState[IndividualT, FitnessT], context: EvolutionContext
    ) -> bool:
        return state.generation >= self._max_generations


class TargetFitnessTermination[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    TerminationCondition[IndividualT, FitnessT]
):
    """Corta la ejecución cuando el mejor individuo alcanza un fitness objetivo."""

    def __init__(self, target_fitness: FitnessT, fitness_comparator: FitnessComparator[FitnessT]) -> None:
        self._target_fitness = target_fitness
        self._fitness_comparator = fitness_comparator

    def should_stop(
        self, state: EvolutionState[IndividualT, FitnessT], context: EvolutionContext
    ) -> bool:
        pop = tuple(state.population)
        if not pop:
            return False
        
        # Encontrar el mejor fitness de la población actual
        best_fitness = pop[0].fitness
        for candidate in pop[1:]:
            if self._fitness_comparator.is_better(candidate.fitness, best_fitness):
                best_fitness = candidate.fitness
                
        # Si el mejor es mejor o igual al objetivo, cortamos
        # is_better no cubre igualdad, así que chequeamos que no sea peor que el objetivo
        return not self._fitness_comparator.is_better(self._target_fitness, best_fitness)


class StagnationTermination[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    TerminationCondition[IndividualT, FitnessT]
):
    """Corta luego de N generaciones sin una mejora significativa."""

    def __init__(self, generations: int, improvement: float = 0.0) -> None:
        if generations <= 0:
            raise ValueError("stagnation generations must be positive")
        if improvement < 0:
            raise ValueError("stagnation improvement must be non-negative")
        self._generations = generations
        self._improvement = improvement
        self._best: float | None = None
        self._last_improvement = 0

    def should_stop(self, state: EvolutionState[IndividualT, FitnessT], context: EvolutionContext) -> bool:
        population = tuple(state.population)
        if not population:
            return False
        value = float(population[0].fitness.value)
        if self._best is None or value > self._best + self._improvement:
            self._best = value
            self._last_improvement = state.generation
        return state.generation - self._last_improvement >= self._generations
