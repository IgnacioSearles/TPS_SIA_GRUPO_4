"""Definición del problema y configuración para aproximar imágenes con triángulos."""

from __future__ import annotations

from typing import Any

from genetic_algorithm.domain.contracts import EvolutionContext, GeneticProblem, FitnessEvaluator
from genetic_algorithm.application.contracts import EvolutionConfiguration
from triangle_image.fitness import TriangleImageTarget, MSEFitness, MSEComparator
from triangle_image.gene import TriangleIndividual


class TriangleConfiguration(EvolutionConfiguration):
    """Configuración concreta del algoritmo."""
    
    def __init__(self, pop_size: int, parent_count: int) -> None:
        self._pop_size = pop_size
        self._parent_count = parent_count

    @property
    def population_size(self) -> int:
        return self._pop_size

    @property
    def selected_parent_count(self) -> int:
        return self._parent_count

    @property
    def data(self) -> object:
        return self


class TriangleContext(EvolutionContext):
    """Contexto de evolución vacío (no se usa info extra)."""
    @property
    def data(self) -> object:
        return self


class TriangleProblem(GeneticProblem[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Empaqueta el objetivo y el evaluador de fitness."""
    
    def __init__(
        self,
        target: TriangleImageTarget,
        fitness_evaluator: FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness],
        fitness_comparator: MSEComparator,
    ) -> None:
        self._target = target
        self._fitness_evaluator = fitness_evaluator
        self._fitness_comparator = fitness_comparator

    @property
    def target(self) -> TriangleImageTarget:
        return self._target

    @property
    def fitness_evaluator(self) -> FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]:
        return self._fitness_evaluator

    @property
    def fitness_comparator(self) -> MSEComparator:
        return self._fitness_comparator
