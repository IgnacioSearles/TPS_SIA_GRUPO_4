"""Mutaciones especializadas para genes de triángulos."""

from __future__ import annotations

from dataclasses import replace
from random import Random

from genetic_algorithm.application.contracts import GeneMutator
from genetic_algorithm.domain.contracts import EvolutionContext
from triangle_image.gene import TriangleGene
from triangle_image.mutation_schedule import TriangleMutationSchedule


def _validate_dimensions_and_strength(width: int, height: int, strength: float) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if strength < 0:
        raise ValueError("mutation_strength must be non-negative")


def _add_noise(value: float, limit: float, strength: float, random_generator: Random) -> float:
    return value + (random_generator.random() * 2 - 1) * limit * strength


def _random_triangle(width: int, height: int, random_generator: Random) -> TriangleGene:
    angle_a = random_generator.uniform(1.0, 178.0)
    angle_b = random_generator.uniform(1.0, 179.0 - angle_a)
    return TriangleGene(
        center_x=random_generator.uniform(0, width),
        center_y=random_generator.uniform(0, height),
        size=random_generator.uniform(1, max(width, height)),
        angle_a=angle_a,
        angle_b=angle_b,
        rotation=random_generator.uniform(0, 360),
        r=random_generator.randint(0, 255),
        g=random_generator.randint(0, 255),
        b=random_generator.randint(0, 255),
        alpha=random_generator.uniform(0.0, 1.0),
    )


class TriangleColorMutator(GeneMutator[TriangleGene]):
    """Modifica solamente RGB y opacidad; preserva la geometría."""

    def __init__(self, mutation_strength: float = 0.1) -> None:
        if mutation_strength < 0:
            raise ValueError("mutation_strength must be non-negative")
        self._strength = mutation_strength

    def mutate_gene(self, gene: TriangleGene, context: EvolutionContext) -> TriangleGene:
        random_generator = context.random_generator

        def mutate_color(value: int) -> int:
            noisy = _add_noise(value, 255.0, self._strength, random_generator)
            return round(max(0.0, min(255.0, noisy)))

        alpha = _add_noise(gene.alpha, 1.0, self._strength, random_generator)
        return replace(
            gene,
            r=mutate_color(gene.r),
            g=mutate_color(gene.g),
            b=mutate_color(gene.b),
            alpha=max(0.0, min(1.0, alpha)),
        )


class TriangleOrientationMutator(GeneMutator[TriangleGene]):
    """Modifica solamente la rotación del triángulo."""

    def __init__(self, mutation_strength: float = 0.1) -> None:
        if mutation_strength < 0:
            raise ValueError("mutation_strength must be non-negative")
        self._strength = mutation_strength

    def mutate_gene(self, gene: TriangleGene, context: EvolutionContext) -> TriangleGene:
        rotation = _add_noise(gene.rotation, 360.0, self._strength, context.random_generator)
        return replace(gene, rotation=rotation % 360.0)


class TriangleShapeMutator(GeneMutator[TriangleGene]):
    """Modifica tamaño y ángulos internos, manteniendo posición y color."""

    def __init__(self, width: int, height: int, mutation_strength: float = 0.1) -> None:
        _validate_dimensions_and_strength(width, height, mutation_strength)
        self._max_size = float(max(width, height))
        self._strength = mutation_strength

    def mutate_gene(self, gene: TriangleGene, context: EvolutionContext) -> TriangleGene:
        random_generator = context.random_generator
        size = _add_noise(gene.size, self._max_size, self._strength, random_generator)
        angle_a = _add_noise(gene.angle_a, 180.0, self._strength, random_generator)
        angle_b = _add_noise(gene.angle_b, 180.0, self._strength, random_generator)

        angle_a = max(1.0, min(178.0, angle_a))
        angle_b = max(1.0, min(179.0 - angle_a, angle_b))
        return replace(
            gene,
            size=max(1.0, min(self._max_size, size)),
            angle_a=angle_a,
            angle_b=angle_b,
        )


class TrianglePositionMutator(GeneMutator[TriangleGene]):
    """Modifica solamente el centro del triángulo."""

    def __init__(self, width: int, height: int, mutation_strength: float = 0.1) -> None:
        _validate_dimensions_and_strength(width, height, mutation_strength)
        self._width = float(width)
        self._height = float(height)
        self._strength = mutation_strength

    def mutate_gene(self, gene: TriangleGene, context: EvolutionContext) -> TriangleGene:
        random_generator = context.random_generator
        center_x = _add_noise(gene.center_x, self._width, self._strength, random_generator)
        center_y = _add_noise(gene.center_y, self._height, self._strength, random_generator)
        return replace(
            gene,
            center_x=max(0.0, min(self._width, center_x)),
            center_y=max(0.0, min(self._height, center_y)),
        )


class TriangleReplacementMutator(GeneMutator[TriangleGene]):
    """Descarta un triángulo y lo reemplaza por uno aleatorio válido."""

    def __init__(self, width: int, height: int) -> None:
        _validate_dimensions_and_strength(width, height, 0.0)
        self._width = width
        self._height = height

    def mutate_gene(self, gene: TriangleGene, context: EvolutionContext) -> TriangleGene:
        return _random_triangle(self._width, self._height, context.random_generator)


class MixedTriangleGeneMutator(GeneMutator[TriangleGene]):
    """Elige una mutación local o, raramente, reemplaza el triángulo completo."""

    def __init__(
        self,
        width: int,
        height: int,
        mutation_strength: float = 0.1,
        replacement_probability: float = 0.02,
        final_mutation_strength: float | None = None,
        final_replacement_probability: float | None = None,
        decay_generations: int = 0,
    ) -> None:
        _validate_dimensions_and_strength(width, height, mutation_strength)
        if not 0.0 <= replacement_probability <= 1.0:
            raise ValueError("replacement_probability must be between 0.0 and 1.0")
        if final_mutation_strength is not None and final_mutation_strength < 0:
            raise ValueError("final_mutation_strength must be non-negative")
        if (final_replacement_probability is not None
                and not 0.0 <= final_replacement_probability <= 1.0):
            raise ValueError("final_replacement_probability must be between 0.0 and 1.0")
        if decay_generations < 0:
            raise ValueError("decay_generations must be non-negative")
        self._width = width
        self._height = height
        self._mutation_strength = mutation_strength
        self._final_mutation_strength = (
            mutation_strength if final_mutation_strength is None else final_mutation_strength
        )
        self._replacement_probability = replacement_probability
        self._final_replacement_probability = (
            replacement_probability
            if final_replacement_probability is None
            else final_replacement_probability
        )
        self._decay_generations = decay_generations
        self._replacement_mutator = TriangleReplacementMutator(width, height)

    def _scheduled_values(self, context: EvolutionContext) -> tuple[float, float]:
        generation = getattr(context, "generation", 0)
        progress = min(generation / self._decay_generations, 1.0) if self._decay_generations else 0.0
        strength = self._mutation_strength + progress * (
            self._final_mutation_strength - self._mutation_strength
        )
        replacement_probability = self._replacement_probability + progress * (
            self._final_replacement_probability - self._replacement_probability
        )
        return strength, replacement_probability

    def mutate_gene(self, gene: TriangleGene, context: EvolutionContext) -> TriangleGene:
        random_generator = context.random_generator
        strength, replacement_probability = self._scheduled_values(context)
        if random_generator.random() < replacement_probability:
            return self._replacement_mutator.mutate_gene(gene, context)
        local_mutator = random_generator.choice((
            TriangleColorMutator(strength),
            TriangleOrientationMutator(strength),
            TriangleShapeMutator(self._width, self._height, strength),
            TrianglePositionMutator(self._width, self._height, strength),
        ))
        return local_mutator.mutate_gene(gene, context)


class TriangleGeneMutator(MixedTriangleGeneMutator):
    """Alias retrocompatible del mutador mixto de triángulos."""


class ScheduledTriangleGeneMutator(GeneMutator[TriangleGene]):
    """Aplica el mutador mixto con parámetros elegidos por una política."""

    def __init__(self, width: int, height: int, schedule: TriangleMutationSchedule) -> None:
        self._width = width
        self._height = height
        self._schedule = schedule

    def mutate_gene(self, gene: TriangleGene, context: EvolutionContext) -> TriangleGene:
        generation = getattr(context, "generation", 0)
        parameters = self._schedule.parameters_at(generation)
        return MixedTriangleGeneMutator(
            self._width,
            self._height,
            parameters.strength,
            parameters.replacement_probability,
        ).mutate_gene(gene, context)
