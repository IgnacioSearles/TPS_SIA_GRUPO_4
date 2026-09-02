"""Implementación concreta del TP2: aproximación de imágenes con triángulos."""

from .gene import TriangleGene, TriangleIndividual
from .codec import TriangleCodec
from .initializer import RandomTriangleInitializer
from .fitness import (BlurredMSEEvaluator, ColorHistogramEvaluator, CompositeEvaluator,
                       MSEComparator, MSEEvaluator, MSEFitness, MultiScaleMSEEvaluator,
                       NormalizedEvaluator, RegionalMSEEvaluator, SCALES, SSIMEvaluator,
                       TriangleImageTarget)
from .mutator import TriangleGeneMutator
from .problem import TriangleConfiguration, TriangleContext, TriangleProblem

__all__ = [
    "BlurredMSEEvaluator", "ColorHistogramEvaluator", "CompositeEvaluator", "MSEComparator",
    "MSEEvaluator", "MSEFitness", "MultiScaleMSEEvaluator", "NormalizedEvaluator",
    "RandomTriangleInitializer", "RegionalMSEEvaluator", "SCALES", "SSIMEvaluator",
    "TriangleCodec", "TriangleConfiguration", "TriangleContext", "TriangleGene",
    "TriangleGeneMutator", "TriangleImageTarget", "TriangleIndividual", "TriangleProblem"
]

