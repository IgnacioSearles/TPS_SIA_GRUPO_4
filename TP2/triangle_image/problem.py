"""Definición del problema y configuración para aproximar imágenes con triángulos."""

from __future__ import annotations

from random import Random
from typing import Any

import numpy as np
from PIL import Image

from genetic_algorithm.domain.contracts import EvolutionContext, GeneticProblem, FitnessEvaluator
from genetic_algorithm.application.contracts import EvolutionConfiguration
from triangle_image.fitness import TriangleImageTarget, MSEFitness, MSEComparator
from triangle_image.gene import TriangleIndividual
from triangle_image.rendering import render


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
    """Contexto de evolución con una fuente aleatoria reproducible opcional."""

    def __init__(self, seed: int | None = None) -> None:
        self._random_generator = Random(seed)
        self._generation = 0
        self._render_cache: dict[tuple[TriangleIndividual, int, int], Image.Image] = {}
        self._array_cache: dict[tuple[TriangleIndividual, int, int], np.ndarray] = {}
        self._mse_cache: dict[tuple[TriangleIndividual, int, int], float] = {}
        self._render_scope_active = False

    @property
    def random_generator(self) -> Random:
        return self._random_generator

    @property
    def generation(self) -> int:
        """Generación actual, actualizada por el orquestador antes de reproducir."""
        return self._generation

    def set_generation(self, generation: int) -> None:
        self._generation = generation

    def begin_render_scope(self) -> None:
        """Inicia un cache efímero para los componentes de un fitness compuesto."""
        self._render_cache.clear()
        self._array_cache.clear()
        self._mse_cache.clear()
        self._render_scope_active = True

    def render_individual(
        self, individual: TriangleIndividual, width: int, height: int
    ) -> Image.Image:
        """Devuelve el fenotipo actual, reutilizándolo solo durante el scope activo."""
        if not self._render_scope_active:
            return render(individual, width, height)
        key = (individual, width, height)
        if key not in self._render_cache:
            self._render_cache[key] = render(individual, width, height)
        return self._render_cache[key]

    def render_array(
        self, individual: TriangleIndividual, width: int, height: int
    ) -> np.ndarray:
        """Devuelve el render como array, reutilizándolo dentro del scope actual."""
        key = (individual, width, height)
        if not self._render_scope_active:
            return np.asarray(self.render_individual(individual, width, height))
        if key not in self._array_cache:
            self._array_cache[key] = np.asarray(
                self.render_individual(individual, width, height)
            )
        return self._array_cache[key]

    def global_mse(
        self, individual: TriangleIndividual, target: np.ndarray, width: int, height: int
    ) -> float:
        """Calcula una sola vez el MSE RGB global durante un fitness compuesto."""
        key = (individual, width, height)
        if self._render_scope_active and key in self._mse_cache:
            return self._mse_cache[key]
        rendered = self.render_array(individual, width, height).astype(np.float32)
        mse = float(np.mean(np.square(target.astype(np.float32) - rendered)))
        if self._render_scope_active:
            self._mse_cache[key] = mse
        return mse

    def end_render_scope(self) -> None:
        """Evicta todos los renders al terminar de evaluar un individuo."""
        self._render_cache.clear()
        self._array_cache.clear()
        self._mse_cache.clear()
        self._render_scope_active = False

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
