"""Implementación concreta del TP2: aproximación de imágenes con triángulos."""

from .gene import TriangleGene, TriangleIndividual
from .codec import TriangleCodec
from .initializer import RandomTriangleInitializer
from .fitness import (BlurredMSEEvaluator, ChamferEdgeEvaluator, ColorHistogramEvaluator,
                       CompositeEvaluator, EdgeMSEEvaluator, GradientOrientationEvaluator,
                       MSEComparator, MSEEvaluator, MSEFitness, MultiScaleMSEEvaluator,
                       NormalizedEvaluator, RegionalMSEEvaluator, SCALES, SSIMEvaluator,
                       SaliencyMSEEvaluator,
                       TriangleImageTarget)
from .mutator import (MixedTriangleGeneMutator, ScheduledTriangleGeneMutator, TriangleColorMutator, TriangleGeneMutator,
                      TriangleOrientationMutator, TrianglePositionMutator,
                      TriangleReplacementMutator, TriangleShapeMutator)
from .mutation_schedule import (AdaptiveReheatMutationSchedule, ConstantMutationSchedule,
                                ExponentialMutationSchedule, LinearMutationSchedule,
                                MutationParameters, MutationScheduleObserver,
                                ScheduledGenePositionSelector, TriangleMutationSchedule)
from .problem import TriangleConfiguration, TriangleContext, TriangleProblem

__all__ = [
    "BlurredMSEEvaluator", "ChamferEdgeEvaluator", "ColorHistogramEvaluator", "CompositeEvaluator",
    "EdgeMSEEvaluator", "GradientOrientationEvaluator", "MSEComparator",
    "MSEEvaluator", "MSEFitness", "MultiScaleMSEEvaluator", "NormalizedEvaluator",
    "RandomTriangleInitializer", "RegionalMSEEvaluator", "SCALES", "SSIMEvaluator", "SaliencyMSEEvaluator",
    "TriangleCodec", "TriangleConfiguration", "TriangleContext", "TriangleGene",
    "AdaptiveReheatMutationSchedule", "ConstantMutationSchedule", "ExponentialMutationSchedule",
    "LinearMutationSchedule", "MixedTriangleGeneMutator", "MutationParameters",
    "MutationScheduleObserver", "ScheduledGenePositionSelector", "ScheduledTriangleGeneMutator",
    "TriangleColorMutator", "TriangleGeneMutator",
    "TriangleImageTarget", "TriangleIndividual", "TriangleOrientationMutator",
    "TrianglePositionMutator", "TriangleProblem", "TriangleReplacementMutator",
    "TriangleShapeMutator", "TriangleMutationSchedule"
]
