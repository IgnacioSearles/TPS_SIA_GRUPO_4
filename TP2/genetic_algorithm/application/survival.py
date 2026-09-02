"""Implementaciones de estrategias de supervivencia."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from genetic_algorithm.application.contracts import SelectionStrategy, SurvivalStrategy
from genetic_algorithm.domain.contracts import (EvolutionContext, Fitness, Individual,
                                                 ScoredIndividual)


class AdditiveSurvival[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SurvivalStrategy[IndividualT, FitnessT]
):
    """Combina padres y descendencia; sobreviven los mejores del conjunto."""

    def __init__(self, selection_strategy: SelectionStrategy[IndividualT, FitnessT]) -> None:
        self._selection_strategy = selection_strategy

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
        return self._selection_strategy.select(candidates, population_size, context)


class ExclusiveSurvival[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SurvivalStrategy[IndividualT, FitnessT]
):
    """Considera solo descendencia; si no alcanza, completa con los mejores padres."""

    def __init__(self, selection_strategy: SelectionStrategy[IndividualT, FitnessT]) -> None:
        self._selection_strategy = selection_strategy

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
            return self._selection_strategy.select(candidates, population_size, context)
            
        needed = population_size - len(candidates)
        if needed > len(current_population):
            raise ValueError("not enough candidates to fill the next generation")
            
        best_parents = self._selection_strategy.select(
            current_population, needed, context
        )
        return candidates + tuple(best_parents)
