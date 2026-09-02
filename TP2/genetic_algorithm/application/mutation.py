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


class RandomGenePositionSelector(GenePositionSelector):
    """Elige posiciones aleatorias con base en una probabilidad de mutación por gen."""
    
    def __init__(self, mutation_probability: float) -> None:
        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be between 0.0 and 1.0")
        self._probability = mutation_probability
        
    def select_positions(self, genome_size: int, context: EvolutionContext) -> Collection[int]:
        import random
        positions = []
        for i in range(genome_size):
            if random.random() < self._probability:
                positions.append(i)
        return tuple(positions)
