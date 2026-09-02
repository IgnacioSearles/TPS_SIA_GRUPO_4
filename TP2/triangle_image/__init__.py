"""Implementación concreta del TP2: aproximación de imágenes con triángulos."""

from .gene import TriangleGene, TriangleIndividual
from .codec import TriangleCodec
from .initializer import RandomTriangleInitializer

__all__ = ["RandomTriangleInitializer", "TriangleCodec", "TriangleGene", "TriangleIndividual"]

