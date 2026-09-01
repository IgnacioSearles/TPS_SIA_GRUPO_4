"""Implementaciones de estrategias de supervivencia."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from genetic_algorithm.application.contracts import SelectionStrategy, SurvivalStrategy
from genetic_algorithm.application.selection import EliteSelection
from genetic_algorithm.domain.contracts import (EvolutionContext, Fitness,
                                                 FitnessComparator, Individual,
                                                 ScoredIndividual)


class AdditiveSurvival[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SurvivalStrategy[IndividualT, FitnessT]
):
    """Combina padres y descendencia; sobreviven los mejores del conjunto."""

    def __init__(self, fitness_comparator: FitnessComparator[FitnessT]) -> None:
        self._elite_selection: SelectionStrategy[IndividualT, FitnessT] = EliteSelection(
            fitness_comparator
        )

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
        return self._elite_selection.select(candidates, population_size, context)
