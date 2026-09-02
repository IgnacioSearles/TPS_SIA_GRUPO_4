"""Mutador específico para TriangleGene."""

from __future__ import annotations

import random

from genetic_algorithm.application.contracts import GeneMutator
from genetic_algorithm.domain.contracts import EvolutionContext
from triangle_image.gene import TriangleGene


class TriangleGeneMutator(GeneMutator[TriangleGene]):
    """Aplica mutaciones aleatorias a los atributos de un TriangleGene."""

    def __init__(self, width: int, height: int, mutation_strength: float = 0.1) -> None:
        self._width = width
        self._height = height
        self._strength = mutation_strength

    def mutate_gene(self, gene: TriangleGene, context: EvolutionContext) -> TriangleGene:
        # Decidir aleatoriamente qué atributo mutar (pueden ser uno o varios)
        # Por simplicidad, mutaremos con una probabilidad a cada atributo
        
        # Helper para agregar ruido
        def add_noise(val: float, limit: float, strength: float) -> float:
            noise = (random.random() * 2 - 1) * limit * strength
            return val + noise

        # Coordenadas
        cx = add_noise(gene.center_x, self._width, self._strength)
        cy = add_noise(gene.center_y, self._height, self._strength)
        cx = max(0.0, min(float(self._width), cx))
        cy = max(0.0, min(float(self._height), cy))

        # Tamaño
        size = add_noise(gene.size, max(self._width, self._height), self._strength)
        size = max(1.0, min(float(max(self._width, self._height)), size))

        # Ángulos
        # Necesitamos garantizar que angle_a + angle_b < 180
        # Primero agregamos ruido independientemente
        aa = add_noise(gene.angle_a, 180.0, self._strength)
        ab = add_noise(gene.angle_b, 180.0, self._strength)
        
        # Luego los ajustamos
        aa = max(1.0, min(178.0, aa))
        if aa + ab >= 180.0:
            ab = 179.0 - aa
        ab = max(1.0, ab)

        # Rotación
        rot = add_noise(gene.rotation, 360.0, self._strength) % 360.0

        # Colores (enteros)
        def add_color_noise(val: int) -> int:
            noise = int((random.random() * 2 - 1) * 255 * self._strength)
            return max(0, min(255, val + noise))

        r = add_color_noise(gene.r)
        g = add_color_noise(gene.g)
        b = add_color_noise(gene.b)

        # Alpha
        alpha = add_noise(gene.alpha, 1.0, self._strength)
        alpha = max(0.0, min(1.0, alpha))

        return TriangleGene(
            center_x=cx,
            center_y=cy,
            size=size,
            angle_a=aa,
            angle_b=ab,
            rotation=rot,
            r=r,
            g=g,
            b=b,
            alpha=alpha,
        )
