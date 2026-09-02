"""Estrategias de emparejamiento de padres."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from genetic_algorithm.application.contracts import ParentPair, ParentPairingStrategy
from genetic_algorithm.domain.contracts import EvolutionContext, Fitness, Individual, ScoredIndividual


class SimpleParentPair[IndividualT: Individual[Any]](ParentPair[IndividualT]):
    def __init__(self, first: IndividualT, second: IndividualT) -> None:
        self._first = first
        self._second = second

    @property
    def first_parent(self) -> IndividualT:
        return self._first

    @property
    def second_parent(self) -> IndividualT:
        return self._second


class RandomPairingStrategy[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    ParentPairingStrategy[IndividualT, FitnessT]
):
    """Empareja a los padres seleccionados de forma aleatoria.
    
    Genera tantas parejas como la mitad de los padres seleccionados.
    """

    def pair(
        self,
        selected: Collection[ScoredIndividual[IndividualT, FitnessT]],
        context: EvolutionContext,
    ) -> Collection[ParentPair[IndividualT]]:
        pool = list(selected)
        context.random_generator.shuffle(pool)
        
        pairs = []
        # Agrupamos de a dos
        for i in range(0, len(pool) - 1, 2):
            pairs.append(SimpleParentPair(pool[i].individual, pool[i + 1].individual))
            
        return tuple(pairs)
