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
        if not population and amount > 0:
            raise ValueError("cannot select from an empty population")

        def compare(
            left: ScoredIndividual[IndividualT, FitnessT],
            right: ScoredIndividual[IndividualT, FitnessT],
        ) -> int:
            if self._fitness_comparator.is_better(left.fitness, right.fitness):
                return -1
            if self._fitness_comparator.is_better(right.fitness, left.fitness):
                return 1
            return 0

        sorted_pop = sorted(population, key=cmp_to_key(compare))
        return tuple(sorted_pop[i % len(sorted_pop)] for i in range(amount))


class ProbabilisticTournamentSelection[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SelectionStrategy[IndividualT, FitnessT]
):
    """Selecciona padres mediante torneos con presión selectiva configurable.

    En cada torneo se ordenan participantes por fitness. El mejor gana con
    ``win_probability``; si no, se prueba con el siguiente y así sucesivamente.
    El muestreo de participantes es sin reemplazo dentro de cada torneo.
    """

    def __init__(
        self,
        fitness_comparator: FitnessComparator[FitnessT],
        tournament_size: int = 3,
        win_probability: float = 0.85,
    ) -> None:
        if tournament_size <= 0:
            raise ValueError("tournament_size must be positive")
        if not 0.0 < win_probability <= 1.0:
            raise ValueError("win_probability must be in (0.0, 1.0]")
        self._fitness_comparator = fitness_comparator
        self._tournament_size = tournament_size
        self._win_probability = win_probability

    def select(
        self,
        population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        amount: int,
        context: EvolutionContext,
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        candidates = tuple(population)
        if not candidates and amount > 0:
            raise ValueError("cannot select from an empty population")

        def compare(left: ScoredIndividual[IndividualT, FitnessT], right: ScoredIndividual[IndividualT, FitnessT]) -> int:
            if self._fitness_comparator.is_better(left.fitness, right.fitness):
                return -1
            if self._fitness_comparator.is_better(right.fitness, left.fitness):
                return 1
            return 0

        selected = []
        rng = context.random_generator
        for _ in range(amount):
            indices = rng.sample(range(len(candidates)), min(self._tournament_size, len(candidates)))
            tournament = sorted((candidates[index] for index in indices), key=cmp_to_key(compare))
            winner = tournament[-1]
            for participant in tournament[:-1]:
                if rng.random() < self._win_probability:
                    winner = participant
                    break
            selected.append(winner)
        return tuple(selected)
