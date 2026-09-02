"""Contratos genéricos del dominio, sin estructuras concretas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from random import Random
from typing import Any


class Individual[GeneT](ABC):
    """Candidato con una representación genética opaca."""
    @property
    @abstractmethod
    def genome(self) -> object:
        """Representación del genoma, sin imponer su estructura."""


class Fitness[FitnessValueT](ABC):
    """Resultado de evaluar un individuo, sin escala predefinida."""
    @property
    @abstractmethod
    def value(self) -> FitnessValueT:
        """Valor de fitness en el formato concreto elegido."""


class ImageTarget[ImageValueT](ABC):
    """Imagen objetivo independiente de librerías gráficas."""
    @property
    @abstractmethod
    def image(self) -> ImageValueT:
        """Imagen en el formato decidido por la implementación."""


class EvolutionContext(ABC):
    """Datos y fuente aleatoria compartidos por los operadores de una ejecución."""
    @property
    @abstractmethod
    def random_generator(self) -> Random:
        """Generador pseudoaleatorio aislado de esta ejecución."""
    @property
    @abstractmethod
    def data(self) -> object:
        """Contexto opaco de la ejecución."""


class ScoredIndividual[IndividualT: Individual[Any], FitnessT: Fitness[Any]](ABC):
    """Asocia un individuo con el fitness obtenido al evaluarlo."""
    @property
    @abstractmethod
    def individual(self) -> IndividualT:
        """Individuo evaluado."""
    @property
    @abstractmethod
    def fitness(self) -> FitnessT:
        """Fitness asociado."""


class FitnessEvaluator[
    IndividualT: Individual[Any],
    TargetT: ImageTarget[Any],
    FitnessT: Fitness[Any],
](ABC):
    """Evalúa individuos contra una imagen objetivo."""
    @abstractmethod
    def evaluate(self, individual: IndividualT, target: TargetT,
                 context: EvolutionContext) -> FitnessT:
        """Calcula el fitness de un individuo."""


class FitnessComparator[FitnessT: Fitness[Any]](ABC):
    """Define el orden de preferencia de fitness."""
    @abstractmethod
    def is_better(self, left: FitnessT, right: FitnessT) -> bool:
        """Indica si ``left`` es mejor que ``right``."""


class GeneticProblem[
    IndividualT: Individual[Any],
    TargetT: ImageTarget[Any],
    FitnessT: Fitness[Any],
](ABC):
    """Agrupa imagen objetivo, evaluación y comparación."""
    @property
    @abstractmethod
    def target(self) -> TargetT:
        """Imagen a aproximar."""
    @property
    @abstractmethod
    def fitness_evaluator(self) -> FitnessEvaluator[IndividualT, TargetT, FitnessT]:
        """Evaluador de fitness."""
    @property
    @abstractmethod
    def fitness_comparator(self) -> FitnessComparator[FitnessT]:
        """Criterio de comparación."""


class AlgorithmConfiguration(ABC):
    """Hiperparámetros sin imponer campos ni formato."""
    @property
    @abstractmethod
    def data(self) -> object:
        """Configuración opaca."""


class EvolutionState[IndividualT: Individual[Any], FitnessT: Fitness[Any]](ABC):
    """Instantánea observable con población ordenada de mejor a peor fitness."""
    @property
    @abstractmethod
    def generation(self) -> int:
        """Número de generación actual."""
    @property
    @abstractmethod
    def population(self) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        """Población evaluada actual, cuyo primer elemento es el mejor."""


class EvolutionResult[IndividualT: Individual[Any], FitnessT: Fitness[Any]](ABC):
    """Resultado final de una ejecución."""
    @property
    @abstractmethod
    def final_state(self) -> EvolutionState[IndividualT, FitnessT]:
        """Último estado alcanzado."""
