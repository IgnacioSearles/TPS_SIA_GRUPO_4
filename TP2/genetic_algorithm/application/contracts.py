"""Puntos de extensión del motor de algoritmos genéticos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from typing import Any

from genetic_algorithm.domain.contracts import (AlgorithmConfiguration,
    EvolutionContext, EvolutionResult, EvolutionState, Fitness, GeneticProblem,
    Individual, ScoredIndividual)

class SelectionStrategy[IndividualT: Individual[Any], FitnessT: Fitness[Any]](ABC):
    """Elige individuos evaluados para su reproducción."""
    @abstractmethod
    def select(self, population: Collection[ScoredIndividual[IndividualT, FitnessT]],
               amount: int, context: EvolutionContext
               ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        """Selecciona candidatos de la población."""


class ParentPair[IndividualT: Individual[Any]](ABC):
    """Par de progenitores sin especificar cómo se eligió."""
    @property
    @abstractmethod
    def first_parent(self) -> IndividualT:
        """Primer progenitor."""
    @property
    @abstractmethod
    def second_parent(self) -> IndividualT:
        """Segundo progenitor."""


class ParentPairingStrategy[IndividualT: Individual[Any], FitnessT: Fitness[Any]](ABC):
    """Agrupa los seleccionados para cruzarlos sin fijar el emparejamiento."""
    @abstractmethod
    def pair(self, selected: Collection[ScoredIndividual[IndividualT, FitnessT]],
             context: EvolutionContext) -> Collection[ParentPair[IndividualT]]:
        """Construye pares de progenitores."""


class PopulationInitializer[IndividualT: Individual[Any]](ABC):
    """Crea la población inicial sin asumir cómo se construyen individuos."""
    @abstractmethod
    def create_initial_population(self, population_size: int,
                                  context: EvolutionContext) -> Collection[IndividualT]:
        """Crea la población inicial de la ejecución."""


class GenomeCodec[IndividualT: Individual[Any], GeneT](ABC):
    """Traduce entre la representación concreta del individuo y sus genes ordenados."""
    @abstractmethod
    def extract_genes(self, individual: IndividualT) -> Sequence[GeneT]:
        """Extrae genes ordenados de un individuo."""
    @abstractmethod
    def build_individual(self, genes: Sequence[GeneT]) -> IndividualT:
        """Reconstruye un individuo desde genes ordenados."""


class CutPointSelector(ABC):
    """Elige el punto de corte para una cruza de un punto."""
    @abstractmethod
    def select_cut_point(self, genome_size: int, context: EvolutionContext) -> int:
        """Devuelve un punto de corte válido para el genoma."""


class TwoCutPointSelector(ABC):
    """Elige dos puntos internos para una cruza de dos puntos."""

    @abstractmethod
    def select_cut_points(self, genome_size: int, context: EvolutionContext) -> tuple[int, int]:
        """Devuelve dos puntos distintos y ordenados."""


class RingCutPointSelector(ABC):
    """Elige inicio y fin para un segmento de un genoma circular."""

    @abstractmethod
    def select_ring_cut_points(self, genome_size: int, context: EvolutionContext) -> tuple[int, int]:
        """Devuelve dos posiciones distintas en el anillo."""


class GenePositionSelector(ABC):
    """Elige las posiciones a alterar durante una mutación MultiGen."""
    @abstractmethod
    def select_positions(self, genome_size: int,
                         context: EvolutionContext) -> Collection[int]:
        """Devuelve las posiciones de genes a mutar."""


class GeneMutator[GeneT](ABC):
    """Define cómo se transforma un gen sin conocer al individuo."""
    @abstractmethod
    def mutate_gene(self, gene: GeneT, context: EvolutionContext) -> GeneT:
        """Devuelve el gen mutado."""


class SurvivalStrategy[IndividualT: Individual[Any], FitnessT: Fitness[Any]](ABC):
    """Construye una generación desde población y descendencia."""
    @abstractmethod
    def build_next_generation(
        self, current_population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        offspring: Collection[ScoredIndividual[IndividualT, FitnessT]],
        population_size: int, context: EvolutionContext,
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        """Devuelve la siguiente población evaluada."""


class CrossoverStrategy[IndividualT: Individual[Any]](ABC):
    """Combina progenitores sin imponer estructura genética."""
    @abstractmethod
    def cross(self, parent_a: IndividualT, parent_b: IndividualT,
              context: EvolutionContext) -> Collection[IndividualT]:
        """Genera descendencia de dos progenitores."""


class MutationStrategy[IndividualT: Individual[Any]](ABC):
    """Produce una variante sin fijar el tipo de mutación."""
    @abstractmethod
    def mutate(self, individual: IndividualT, context: EvolutionContext) -> IndividualT:
        """Devuelve un individuo mutado."""


class TerminationCondition[IndividualT: Individual[Any], FitnessT: Fitness[Any]](ABC):
    """Decide si la ejecución debe finalizar."""
    @abstractmethod
    def should_stop(self, state: EvolutionState[IndividualT, FitnessT],
                    context: EvolutionContext) -> bool:
        """Indica si no se debe generar otra población."""


class EvolutionObserver[IndividualT: Individual[Any], FitnessT: Fitness[Any]](ABC):
    """Recibe estados evaluados para informar o registrar una ejecución."""

    @abstractmethod
    def on_generation(
        self, state: EvolutionState[IndividualT, FitnessT], context: EvolutionContext
    ) -> None:
        """Procesa el estado inicial o el estado resultante de una generación."""


class EvolutionConfiguration(AlgorithmConfiguration, ABC):
    """Configuración mínima que necesita el orquestador genérico."""
    @property
    @abstractmethod
    def population_size(self) -> int:
        """Cantidad esperada de individuos por generación."""
    @property
    @abstractmethod
    def selected_parent_count(self) -> int:
        """Cantidad de candidatos que la selección debe devolver."""


class GeneticAlgorithm[
    ProblemT: GeneticProblem[Any, Any, Any],
    IndividualT: Individual[Any],
    FitnessT: Fitness[Any],
    ConfigurationT: AlgorithmConfiguration,
](ABC):
    """Orquestador abstracto del ciclo evolutivo."""
    @abstractmethod
    def run(self, problem: ProblemT, configuration: ConfigurationT
            ) -> EvolutionResult[IndividualT, FitnessT]:
        """Ejecuta el ciclo y devuelve su resultado final."""
