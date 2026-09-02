"""Implementaciones de estrategias de cruza."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from genetic_algorithm.application.contracts import (
    CrossoverStrategy,
    CutPointSelector,
    GenomeCodec,
    RingCutPointSelector,
    TwoCutPointSelector,
)
from genetic_algorithm.domain.contracts import EvolutionContext, Individual


def _parent_genes[IndividualT: Individual[Any], GeneT](
    genome_codec: GenomeCodec[IndividualT, GeneT],
    parent_a: IndividualT,
    parent_b: IndividualT,
    minimum_size: int,
) -> tuple[tuple[GeneT, ...], tuple[GeneT, ...]]:
    genes_a = tuple(genome_codec.extract_genes(parent_a))
    genes_b = tuple(genome_codec.extract_genes(parent_b))
    if len(genes_a) != len(genes_b):
        raise ValueError("parents must have genomes of the same length")
    if len(genes_a) < minimum_size:
        raise ValueError(f"crossover requires at least {minimum_size} genes")
    return genes_a, genes_b


def _build_children[IndividualT: Individual[Any], GeneT](
    genome_codec: GenomeCodec[IndividualT, GeneT],
    genes_a: tuple[GeneT, ...],
    genes_b: tuple[GeneT, ...],
) -> tuple[IndividualT, IndividualT]:
    return genome_codec.build_individual(genes_a), genome_codec.build_individual(genes_b)


class OnePointCrossover[IndividualT: Individual[Any], GeneT](CrossoverStrategy[IndividualT]):
    """Intercambia las colas de dos genomas desde un punto interno."""

    def __init__(self, genome_codec: GenomeCodec[IndividualT, GeneT], cut_point_selector: CutPointSelector) -> None:
        self._genome_codec = genome_codec
        self._cut_point_selector = cut_point_selector

    def cross(self, parent_a: IndividualT, parent_b: IndividualT, context: EvolutionContext) -> Collection[IndividualT]:
        genes_a, genes_b = _parent_genes(self._genome_codec, parent_a, parent_b, 2)
        cut_point = self._cut_point_selector.select_cut_point(len(genes_a), context)
        if not 0 < cut_point < len(genes_a):
            raise ValueError("cut point must be strictly inside the genome")
        return _build_children(
            self._genome_codec,
            genes_a[:cut_point] + genes_b[cut_point:],
            genes_b[:cut_point] + genes_a[cut_point:],
        )


class TwoPointCrossover[IndividualT: Individual[Any], GeneT](CrossoverStrategy[IndividualT]):
    """Intercambia el segmento comprendido entre dos puntos de corte internos."""

    def __init__(self, genome_codec: GenomeCodec[IndividualT, GeneT], cut_point_selector: TwoCutPointSelector) -> None:
        self._genome_codec = genome_codec
        self._cut_point_selector = cut_point_selector

    def cross(self, parent_a: IndividualT, parent_b: IndividualT, context: EvolutionContext) -> Collection[IndividualT]:
        genes_a, genes_b = _parent_genes(self._genome_codec, parent_a, parent_b, 3)
        first, second = self._cut_point_selector.select_cut_points(len(genes_a), context)
        if not 0 < first < second < len(genes_a):
            raise ValueError("two cut points must be distinct and strictly inside the genome")
        return _build_children(
            self._genome_codec,
            genes_a[:first] + genes_b[first:second] + genes_a[second:],
            genes_b[:first] + genes_a[first:second] + genes_b[second:],
        )


class UniformCrossover[IndividualT: Individual[Any], GeneT](CrossoverStrategy[IndividualT]):
    """Decide independientemente para cada posición si intercambia ambos genes."""

    def __init__(self, genome_codec: GenomeCodec[IndividualT, GeneT], swap_probability: float = 0.5) -> None:
        if not 0.0 <= swap_probability <= 1.0:
            raise ValueError("swap_probability must be between 0.0 and 1.0")
        self._genome_codec = genome_codec
        self._swap_probability = swap_probability

    def cross(self, parent_a: IndividualT, parent_b: IndividualT, context: EvolutionContext) -> Collection[IndividualT]:
        genes_a, genes_b = _parent_genes(self._genome_codec, parent_a, parent_b, 1)
        child_a: list[GeneT] = []
        child_b: list[GeneT] = []
        for gene_a, gene_b in zip(genes_a, genes_b):
            if context.random_generator.random() < self._swap_probability:
                child_a.append(gene_b)
                child_b.append(gene_a)
            else:
                child_a.append(gene_a)
                child_b.append(gene_b)
        return _build_children(self._genome_codec, tuple(child_a), tuple(child_b))


class AnnularCrossover[IndividualT: Individual[Any], GeneT](CrossoverStrategy[IndividualT]):
    """Intercambia un segmento circular; el intervalo puede atravesar el final del genoma."""

    def __init__(self, genome_codec: GenomeCodec[IndividualT, GeneT], cut_point_selector: RingCutPointSelector) -> None:
        self._genome_codec = genome_codec
        self._cut_point_selector = cut_point_selector

    def cross(self, parent_a: IndividualT, parent_b: IndividualT, context: EvolutionContext) -> Collection[IndividualT]:
        genes_a, genes_b = _parent_genes(self._genome_codec, parent_a, parent_b, 2)
        start, end = self._cut_point_selector.select_ring_cut_points(len(genes_a), context)
        if not 0 <= start < len(genes_a) or not 0 <= end < len(genes_a) or start == end:
            raise ValueError("annular cut points must be distinct genome positions")

        selected = set()
        position = start
        while position != end:
            selected.add(position)
            position = (position + 1) % len(genes_a)

        child_a = tuple(gene_b if index in selected else gene_a for index, (gene_a, gene_b) in enumerate(zip(genes_a, genes_b)))
        child_b = tuple(gene_a if index in selected else gene_b for index, (gene_a, gene_b) in enumerate(zip(genes_a, genes_b)))
        return _build_children(self._genome_codec, child_a, child_b)


class RandomCutPointSelector(CutPointSelector):
    """Elige un punto de corte aleatorio, excluyendo extremos."""

    def select_cut_point(self, genome_size: int, context: EvolutionContext) -> int:
        if genome_size < 2:
            raise ValueError("one-point crossover requires at least two genes")
        return context.random_generator.randint(1, genome_size - 1)


class RandomTwoCutPointSelector(TwoCutPointSelector):
    """Elige dos puntos internos y distintos para cruza de dos puntos."""

    def select_cut_points(self, genome_size: int, context: EvolutionContext) -> tuple[int, int]:
        if genome_size < 3:
            raise ValueError("two-point crossover requires at least three genes")
        first, second = context.random_generator.sample(range(1, genome_size), 2)
        return min(first, second), max(first, second)


class RandomRingCutPointSelector(RingCutPointSelector):
    """Elige inicio y fin distintos en un genoma tratado como circular."""

    def select_ring_cut_points(self, genome_size: int, context: EvolutionContext) -> tuple[int, int]:
        if genome_size < 2:
            raise ValueError("annular crossover requires at least two genes")
        return tuple(context.random_generator.sample(range(genome_size), 2))
