"""Implementaciones de estrategias de mutación."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from genetic_algorithm.application.contracts import (GeneMutator,
                                                      GenePositionSelector,
                                                      GenomeCodec, MutationStrategy)
from genetic_algorithm.domain.contracts import EvolutionContext, Individual


class MultiGeneMutation[IndividualT: Individual[Any], GeneT](MutationStrategy[IndividualT]):
    """Muta las posiciones elegidas, delegando selección y cambio de cada gen."""

    def __init__(
        self,
        genome_codec: GenomeCodec[IndividualT, GeneT],
        position_selector: GenePositionSelector,
        gene_mutator: GeneMutator[GeneT],
    ) -> None:
        self._genome_codec = genome_codec
        self._position_selector = position_selector
        self._gene_mutator = gene_mutator

    def mutate(self, individual: IndividualT, context: EvolutionContext) -> IndividualT:
        """Reconstruye el individuo tras mutar las posiciones seleccionadas."""
        genes = list(self._genome_codec.extract_genes(individual))
        positions = tuple(self._position_selector.select_positions(len(genes), context))
        if len(set(positions)) != len(positions):
            raise ValueError("gene positions must not be repeated")
        if any(position < 0 or position >= len(genes) for position in positions):
            raise ValueError("gene position is outside the genome")

        for position in positions:
            genes[position] = self._gene_mutator.mutate_gene(genes[position], context)
        return self._genome_codec.build_individual(genes)


class GenMutation(MultiGeneMutation):
    """Mutación de un solo gen; reutiliza toda la implementación MultiGen."""


class MultiGenMutation(MultiGeneMutation):
    """Nombre explícito para la variante MultiGen del enunciado."""


class UniformMutation(MultiGeneMutation):
    """Mutación uniforme parametrizada por un selector y un mutador de genes."""


class NonUniformMutation(MultiGeneMutation):
    """Mutación no uniforme parametrizada por un schedule."""


class RandomGenePositionSelector(GenePositionSelector):
    """Elige posiciones aleatorias con base en una probabilidad de mutación por gen."""
    
    def __init__(
        self,
        mutation_probability: float,
        final_probability: float | None = None,
        decay_generations: int = 0,
    ) -> None:
        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0.0 and 1.0")
        if final_probability is not None and not 0.0 <= final_probability <= 1.0:
            raise ValueError("final_probability must be between 0.0 and 1.0")
        if decay_generations < 0:
            raise ValueError("decay_generations must be non-negative")
        self._probability = mutation_probability
        self._final_probability = mutation_probability if final_probability is None else final_probability
        self._decay_generations = decay_generations
        
    def select_positions(self, genome_size: int, context: EvolutionContext) -> Collection[int]:
        generation = getattr(context, "generation", 0)
        progress = min(generation / self._decay_generations, 1.0) if self._decay_generations else 0.0
        probability = self._probability + progress * (self._final_probability - self._probability)
        positions = []
        for i in range(genome_size):
            if context.random_generator.random() < probability:
                positions.append(i)
        return tuple(positions)


class SingleGenePositionSelector(GenePositionSelector):
    """Selecciona exactamente una posición al azar."""

    def select_positions(self, genome_size: int, context: EvolutionContext) -> Collection[int]:
        if genome_size <= 0:
            return ()
        return (context.random_generator.randrange(genome_size),)


class AllGenePositionSelector(GenePositionSelector):
    """Selecciona todos los genes del individuo."""

    def select_positions(self, genome_size: int, context: EvolutionContext) -> Collection[int]:
        return tuple(range(genome_size))
