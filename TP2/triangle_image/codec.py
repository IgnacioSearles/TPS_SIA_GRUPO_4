"""Codec entre `TriangleIndividual` y su secuencia ordenada de genes."""

from __future__ import annotations

from collections.abc import Sequence

from genetic_algorithm.application.contracts import GenomeCodec
from triangle_image.gene import TriangleGene, TriangleIndividual


class TriangleCodec(GenomeCodec[TriangleIndividual, TriangleGene]):
    """Implementación del codec para individuos basados en triángulos."""

    def extract_genes(self, individual: TriangleIndividual) -> Sequence[TriangleGene]:
        return individual.genome

    def build_individual(self, genes: Sequence[TriangleGene]) -> TriangleIndividual:
        return TriangleIndividual(tuple(genes))
