"""Implementaciones de estrategias de cruza."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from genetic_algorithm.application.contracts import (CrossoverStrategy,
                                                      CutPointSelector, GenomeCodec)
from genetic_algorithm.domain.contracts import EvolutionContext, Individual


class OnePointCrossover[IndividualT: Individual[Any], GeneT](
    CrossoverStrategy[IndividualT]
):
    """Cruza dos genomas ordenados en un punto elegido por una dependencia."""

    def __init__(
        self,
        genome_codec: GenomeCodec[IndividualT, GeneT],
        cut_point_selector: CutPointSelector,
    ) -> None:
        self._genome_codec = genome_codec
        self._cut_point_selector = cut_point_selector

    def cross(
        self,
        parent_a: IndividualT,
        parent_b: IndividualT,
        context: EvolutionContext,
    ) -> Collection[IndividualT]:
        """Crea dos hijos intercambiando las colas desde el punto de corte."""
        genes_a = tuple(self._genome_codec.extract_genes(parent_a))
        genes_b = tuple(self._genome_codec.extract_genes(parent_b))
        if len(genes_a) != len(genes_b):
            raise ValueError("parents must have genomes of the same length")
        if len(genes_a) < 2:
            raise ValueError("one-point crossover requires at least two genes")

        cut_point = self._cut_point_selector.select_cut_point(len(genes_a), context)
        if not 0 < cut_point < len(genes_a):
            raise ValueError("cut point must be strictly inside the genome")

        child_a_genes = genes_a[:cut_point] + genes_b[cut_point:]
        child_b_genes = genes_b[:cut_point] + genes_a[cut_point:]
        return tuple(
            self._genome_codec.build_individual(genes)
            for genes in (child_a_genes, child_b_genes)
        )


class RandomCutPointSelector(CutPointSelector):
    """Elige un punto de corte aleatorio (excluyendo extremos)."""
    
    def select_cut_point(self, genome_size: int, context: EvolutionContext) -> int:
        import random
        # Si el tamaño es muy pequeño, cortamos en el medio
        if genome_size <= 2:
            return genome_size // 2
        # Elegir un punto entre 1 y genome_size - 1
        return random.randint(1, genome_size - 1)
