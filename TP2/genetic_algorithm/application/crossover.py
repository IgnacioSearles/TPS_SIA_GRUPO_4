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

        child_a = self._genome_codec.build_individual(genes_a[:cut_point] + genes_b[cut_point:])
        child_b = self._genome_codec.build_individual(genes_b[:cut_point] + genes_a[cut_point:])
        return child_a, child_b
