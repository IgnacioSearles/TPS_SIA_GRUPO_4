"""Implementaciones de estrategias de supervivencia."""

from __future__ import annotations

from collections.abc import Collection
from functools import cmp_to_key
from typing import Any

from genetic_algorithm.application.contracts import SurvivalStrategy
from genetic_algorithm.domain.contracts import (EvolutionContext, Fitness,
                                                 FitnessComparator, Individual,
                                                 ScoredIndividual)


def _take_best[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    population: Collection[ScoredIndividual[IndividualT, FitnessT]],
    amount: int,
    fitness_comparator: FitnessComparator[FitnessT],
) -> tuple[ScoredIndividual[IndividualT, FitnessT], ...]:
    """Helper interno para ordenar y truncar candidatos sin lógica de repetición."""
    if amount > len(population):
        raise ValueError("amount cannot exceed population size")

    def compare(
        left: ScoredIndividual[IndividualT, FitnessT],
        right: ScoredIndividual[IndividualT, FitnessT],
    ) -> int:
        if fitness_comparator.is_better(left.fitness, right.fitness):
            return -1
        if fitness_comparator.is_better(right.fitness, left.fitness):
            return 1
        return 0

    return tuple(sorted(population, key=cmp_to_key(compare))[:amount])


class AdditiveSurvival[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SurvivalStrategy[IndividualT, FitnessT]
):
    """Combina padres y descendencia; sobreviven los mejores del conjunto."""

    def __init__(self, fitness_comparator: FitnessComparator[FitnessT]) -> None:
        self._fitness_comparator = fitness_comparator

    def build_next_generation(
        self,
        current_population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        offspring: Collection[ScoredIndividual[IndividualT, FitnessT]],
        population_size: int,
        context: EvolutionContext,
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        """Aplica supervivencia aditiva sobre la unión de ambas poblaciones."""
        if population_size < 0:
            raise ValueError("population_size must be non-negative")
        candidates = tuple(current_population) + tuple(offspring)
        if population_size > len(candidates):
            raise ValueError("not enough candidates to fill the next generation")
        
        return _take_best(candidates, population_size, self._fitness_comparator)


class ExclusiveSurvival[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SurvivalStrategy[IndividualT, FitnessT]
):
    """Considera solo descendencia; si no alcanza, completa con los mejores padres."""

    def __init__(self, fitness_comparator: FitnessComparator[FitnessT]) -> None:
        self._fitness_comparator = fitness_comparator

    def build_next_generation(
        self,
        current_population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        offspring: Collection[ScoredIndividual[IndividualT, FitnessT]],
        population_size: int,
        context: EvolutionContext,
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        """Aplica supervivencia exclusiva priorizando a la descendencia."""
        if population_size < 0:
            raise ValueError("population_size must be non-negative")
            
        candidates = tuple(offspring)
        if len(candidates) >= population_size:
            return _take_best(candidates, population_size, self._fitness_comparator)
            
        needed = population_size - len(candidates)
        if needed > len(current_population):
            raise ValueError("not enough candidates to fill the next generation")
            
        best_parents = _take_best(current_population, needed, self._fitness_comparator)
        return candidates + best_parents
