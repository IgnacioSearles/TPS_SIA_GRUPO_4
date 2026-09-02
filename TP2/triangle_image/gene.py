from __future__ import annotations

import math
from dataclasses import dataclass

from genetic_algorithm.domain.contracts import Individual

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class TriangleGene:
    center_x: float
    center_y: float
    size: float
    angle_a: float
    angle_b: float
    rotation: float
    r: int
    g: int
    b: int
    alpha: float

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        if self.angle_a <= 0 or self.angle_b <= 0:
            raise ValueError("angle_a and angle_b must be positive")
        if self.angle_a + self.angle_b >= 180:
            raise ValueError("angle_a + angle_b must be less than 180 degrees")
        if not 0 <= self.r <= 255 or not 0 <= self.g <= 255 or not 0 <= self.b <= 255:
            raise ValueError("r, g and b must be between 0 and 255")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

    @property
    def angle_c(self) -> float:
        return 180.0 - self.angle_a - self.angle_b

    @property
    def vertices(self) -> tuple[Point, Point, Point]:
        theta_1 = 0.0
        theta_2 = theta_1 + 2 * self.angle_c
        theta_3 = theta_2 + 2 * self.angle_a
        return (
            self._vertex_at(theta_1),
            self._vertex_at(theta_2),
            self._vertex_at(theta_3),
        )

    def _vertex_at(self, theta_degrees: float) -> Point:
        angle = math.radians(theta_degrees + self.rotation)
        return (
            self.center_x + self.size * math.cos(angle),
            self.center_y + self.size * math.sin(angle),
        )


class TriangleIndividual(Individual[TriangleGene]):
    def __init__(self, triangles: tuple[TriangleGene, ...]) -> None:
        self._triangles = triangles

    @property
    def genome(self) -> tuple[TriangleGene, ...]:
        return self._triangles
