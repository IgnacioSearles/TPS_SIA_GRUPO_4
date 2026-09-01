"""Implementaciones de estrategias de selección."""

from __future__ import annotations

from collections.abc import Collection
from functools import cmp_to_key
from typing import Any

from genetic_algorithm.application.contracts import SelectionStrategy
from genetic_algorithm.domain.contracts import (EvolutionContext, Fitness,
                                                 FitnessComparator, Individual,
                                                 ScoredIndividual)


class EliteSelection[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SelectionStrategy[IndividualT, FitnessT]
):
    """Selecciona los candidatos mejor evaluados según un comparador inyectado."""

    def __init__(self, fitness_comparator: FitnessComparator[FitnessT]) -> None:
        self._fitness_comparator = fitness_comparator

    def select(
        self,
        population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        amount: int,
        context: EvolutionContext,
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        """Devuelve los ``amount`` mejores individuos de la población."""
        if amount < 0:
            raise ValueError("amount must be non-negative")
        if amount > len(population):
            raise ValueError("amount cannot exceed population size")

        def compare(
            left: ScoredIndividual[IndividualT, FitnessT],
            right: ScoredIndividual[IndividualT, FitnessT],
        ) -> int:
            if self._fitness_comparator.is_better(left.fitness, right.fitness):
                return -1
            if self._fitness_comparator.is_better(right.fitness, left.fitness):
                return 1
            return 0

        return tuple(sorted(population, key=cmp_to_key(compare))[:amount])
