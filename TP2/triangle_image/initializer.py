"""Inicializador de población para el problema de aproximación con triángulos."""

from __future__ import annotations

from collections.abc import Collection

from genetic_algorithm.application.contracts import PopulationInitializer
from genetic_algorithm.domain.contracts import EvolutionContext
from triangle_image.gene import TriangleGene, TriangleIndividual


class RandomTriangleInitializer(PopulationInitializer[TriangleIndividual]):
    """Genera una población inicial de individuos con triángulos aleatorios."""

    def __init__(self, triangles_per_individual: int, width: int, height: int) -> None:
        self._triangles_per_individual = triangles_per_individual
        self._width = width
        self._height = height

    def create_initial_population(
        self, population_size: int, context: EvolutionContext
    ) -> Collection[TriangleIndividual]:
        population = []
        for _ in range(population_size):
            genes = []
            for _ in range(self._triangles_per_individual):
                genes.append(self._generate_random_triangle(context))
            population.append(TriangleIndividual(tuple(genes)))
        return tuple(population)

    def _generate_random_triangle(self, context: EvolutionContext) -> TriangleGene:
        random_generator = context.random_generator
        angle_a = random_generator.uniform(1.0, 178.0)
        angle_b = random_generator.uniform(1.0, 179.0 - angle_a)

        return TriangleGene(
            center_x=random_generator.uniform(0, self._width),
            center_y=random_generator.uniform(0, self._height),
            size=random_generator.uniform(1, max(self._width, self._height)),
            angle_a=angle_a,
            angle_b=angle_b,
            rotation=random_generator.uniform(0, 360),
            r=random_generator.randint(0, 255),
            g=random_generator.randint(0, 255),
            b=random_generator.randint(0, 255),
            alpha=random_generator.uniform(0.0, 1.0),
        )
