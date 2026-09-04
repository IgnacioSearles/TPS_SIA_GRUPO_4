"""Implementaciones de estrategias de selección."""

from __future__ import annotations

from collections.abc import Collection
from functools import cmp_to_key
import math
from typing import Any

from genetic_algorithm.application.contracts import SelectionStrategy
from genetic_algorithm.domain.contracts import (EvolutionContext, Fitness,
                                                 FitnessComparator, Individual,
                                                 ScoredIndividual)


def _validate_selection_request[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    population: Collection[ScoredIndividual[IndividualT, FitnessT]],
    amount: int,
) -> tuple[ScoredIndividual[IndividualT, FitnessT], ...]:
    if amount < 0:
        raise ValueError("amount must be non-negative")
    candidates = tuple(population)
    if not candidates and amount > 0:
        raise ValueError("cannot select from an empty population")
    return candidates


def _compare_by_fitness[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    fitness_comparator: FitnessComparator[FitnessT],
):
    def compare(
        left: ScoredIndividual[IndividualT, FitnessT],
        right: ScoredIndividual[IndividualT, FitnessT],
    ) -> int:
        if fitness_comparator.is_better(left.fitness, right.fitness):
            return -1
        if fitness_comparator.is_better(right.fitness, left.fitness):
            return 1
        return 0

    return compare


def _ranked_by_fitness[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    population: Collection[ScoredIndividual[IndividualT, FitnessT]],
    fitness_comparator: FitnessComparator[FitnessT],
) -> tuple[ScoredIndividual[IndividualT, FitnessT], ...]:
    return tuple(sorted(population, key=cmp_to_key(_compare_by_fitness(fitness_comparator))))


def _numeric_fitness_values[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    candidates: tuple[ScoredIndividual[IndividualT, FitnessT], ...],
) -> tuple[float, ...]:
    try:
        return tuple(float(candidate.fitness.value) for candidate in candidates)
    except (TypeError, ValueError) as error:
        raise ValueError("weighted selection requires numeric fitness values") from error


def _non_negative_weights(values: tuple[float, ...]) -> tuple[float, ...]:
    if any(not math.isfinite(value) for value in values):
        raise ValueError("weighted selection requires finite fitness values")
    if not values:
        return ()
    minimum = min(values)
    shifted = tuple(value - minimum for value in values) if minimum < 0.0 else values
    return tuple(max(0.0, value) for value in shifted)


def _select_by_weight[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    candidates: tuple[ScoredIndividual[IndividualT, FitnessT], ...],
    weights: tuple[float, ...],
    point: float,
) -> ScoredIndividual[IndividualT, FitnessT]:
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights):
        cumulative += weight
        if point < cumulative:
            return candidate
    return candidates[-1]


def _select_weighted_with_replacement[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    candidates: tuple[ScoredIndividual[IndividualT, FitnessT], ...],
    weights: tuple[float, ...],
    amount: int,
    context: EvolutionContext,
) -> tuple[ScoredIndividual[IndividualT, FitnessT], ...]:
    if amount == 0:
        return ()
    total = sum(weights)
    rng = context.random_generator
    if total <= 0.0:
        return tuple(rng.choice(candidates) for _ in range(amount))
    return tuple(
        _select_by_weight(candidates, weights, rng.random() * total)
        for _ in range(amount)
    )


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
        sorted_pop = _ranked_by_fitness(
            _validate_selection_request(population, amount), self._fitness_comparator
        )
        if amount == 0:
            return ()
        return tuple(sorted_pop[i % len(sorted_pop)] for i in range(amount))


class RouletteSelection[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SelectionStrategy[IndividualT, FitnessT]
):
    """Selecciona con probabilidad proporcional al valor numérico del fitness.

    Si todos los pesos son cero, cae a muestreo uniforme. Los fitness negativos se
    desplazan para que el peor valor pese cero y el resto mantenga diferencias
    relativas.
    """

    def select(
        self,
        population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        amount: int,
        context: EvolutionContext,
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        candidates = _validate_selection_request(population, amount)
        weights = _non_negative_weights(_numeric_fitness_values(candidates))
        return _select_weighted_with_replacement(candidates, weights, amount, context)


class UniversalSelection[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SelectionStrategy[IndividualT, FitnessT]
):
    """Selección universal estocástica con agujas equiespaciadas."""

    def select(
        self,
        population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        amount: int,
        context: EvolutionContext,
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        candidates = _validate_selection_request(population, amount)
        if amount == 0:
            return ()
        weights = _non_negative_weights(_numeric_fitness_values(candidates))
        total = sum(weights)
        rng = context.random_generator
        if total <= 0.0:
            return tuple(rng.choice(candidates) for _ in range(amount))
        step = total / amount
        start = rng.random() * step
        return tuple(
            _select_by_weight(candidates, weights, start + index * step)
            for index in range(amount)
        )


class BoltzmannSelection[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SelectionStrategy[IndividualT, FitnessT]
):
    """Selección probabilística con pesos exponenciales controlados por temperatura."""

    def __init__(
        self,
        temperature: float = 1.0,
        standardize: bool = False,
    ) -> None:
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("temperature must be positive and finite")
        self._temperature = temperature
        self._standardize = standardize

    def _scores(self, values: tuple[float, ...]) -> tuple[float, ...]:
        if not self._standardize:
            return values
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std_dev = math.sqrt(variance)
        if std_dev <= 1e-12:
            return tuple(0.0 for _ in values)
        return tuple((value - mean) / std_dev for value in values)

    def select(
        self,
        population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        amount: int,
        context: EvolutionContext,
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        candidates = _validate_selection_request(population, amount)
        values = _numeric_fitness_values(candidates)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("boltzmann selection requires finite fitness values")
        if not values:
            return ()
        scores = self._scores(values)
        maximum = max(scores)
        weights = tuple(math.exp((score - maximum) / self._temperature) for score in scores)
        return _select_weighted_with_replacement(candidates, weights, amount, context)


class RankingSelection[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SelectionStrategy[IndividualT, FitnessT]
):
    """Selecciona por ranking, no por magnitud absoluta del fitness."""

    def __init__(self, fitness_comparator: FitnessComparator[FitnessT]) -> None:
        self._fitness_comparator = fitness_comparator

    def select(
        self,
        population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        amount: int,
        context: EvolutionContext,
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        ranked = _ranked_by_fitness(
            _validate_selection_request(population, amount), self._fitness_comparator
        )
        if amount == 0:
            return ()
        weights = tuple(float(len(ranked) - index) for index in range(len(ranked)))
        return _select_weighted_with_replacement(ranked, weights, amount, context)


class DeterministicTournamentSelection[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    SelectionStrategy[IndividualT, FitnessT]
):
    """Selecciona siempre al mejor participante de cada torneo aleatorio."""

    def __init__(
        self,
        fitness_comparator: FitnessComparator[FitnessT],
        tournament_size: int = 3,
    ) -> None:
        if tournament_size <= 0:
            raise ValueError("tournament_size must be positive")
        self._fitness_comparator = fitness_comparator
        self._tournament_size = tournament_size

    def select(
        self,
        population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        amount: int,
        context: EvolutionContext,
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        candidates = _validate_selection_request(population, amount)
        if amount == 0:
            return ()
        rng = context.random_generator
        selected = []
        for _ in range(amount):
            indices = rng.sample(range(len(candidates)), min(self._tournament_size, len(candidates)))
            tournament = _ranked_by_fitness(
                tuple(candidates[index] for index in indices), self._fitness_comparator
            )
            selected.append(tournament[0])
        return tuple(selected)


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
        candidates = _validate_selection_request(population, amount)
        if amount == 0:
            return ()

        selected = []
        rng = context.random_generator
        for _ in range(amount):
            indices = rng.sample(range(len(candidates)), min(self._tournament_size, len(candidates)))
            tournament = _ranked_by_fitness(
                tuple(candidates[index] for index in indices), self._fitness_comparator
            )
            winner = tournament[-1]
            for participant in tournament[:-1]:
                if rng.random() < self._win_probability:
                    winner = participant
                    break
            selected.append(winner)
        return tuple(selected)
