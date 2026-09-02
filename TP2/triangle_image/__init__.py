"""Implementación concreta del TP2: aproximación de imágenes con triángulos."""

from .gene import TriangleGene, TriangleIndividual
from .codec import TriangleCodec
from .initializer import RandomTriangleInitializer
from .fitness import MSEComparator, MSEEvaluator, MSEFitness, TriangleImageTarget
from .mutator import TriangleGeneMutator
from .problem import TriangleConfiguration, TriangleContext, TriangleProblem

__all__ = [
    "MSEComparator", "MSEEvaluator", "MSEFitness", "RandomTriangleInitializer",
    "TriangleCodec", "TriangleConfiguration", "TriangleContext", "TriangleGene", 
    "TriangleGeneMutator", "TriangleImageTarget", "TriangleIndividual", "TriangleProblem"
]

