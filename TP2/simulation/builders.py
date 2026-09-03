"""Traduce la configuración declarativa en los operadores concretos del motor."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from genetic_algorithm.application import (
    AdditiveSurvival,
    AnnularCrossover,
    CrossoverStrategy,
    EliteSelection,
    ExclusiveSurvival,
    OnePointCrossover,
    ProbabilisticTournamentSelection,
    RandomCutPointSelector,
    RandomRingCutPointSelector,
    RandomTwoCutPointSelector,
    SelectionStrategy,
    SurvivalStrategy,
    TwoPointCrossover,
    UniformCrossover,
)
from genetic_algorithm.domain import FitnessEvaluator
from triangle_image import (
    AdaptiveReheatMutationSchedule,
    BlurredMSEEvaluator,
    ChamferEdgeEvaluator,
    ColorHistogramEvaluator,
    CompositeEvaluator,
    ConstantMutationSchedule,
    EdgeMSEEvaluator,
    ExponentialMutationSchedule,
    GradientOrientationEvaluator,
    LinearMutationSchedule,
    MSEComparator,
    MSEEvaluator,
    MSEFitness,
    MultiScaleMSEEvaluator,
    RegionalMSEEvaluator,
    SSIMEvaluator,
    SaliencyMSEEvaluator,
    TriangleCodec,
    TriangleImageTarget,
    TriangleIndividual,
    TriangleMutationSchedule,
)

from simulation.config import (
    CrossoverConfig,
    FitnessConfig,
    MutationConfig,
    PopulationConfig,
    SelectionConfig,
)

type TriangleEvaluator = FitnessEvaluator[
    TriangleIndividual, TriangleImageTarget, MSEFitness
]

_METRIC_FACTORIES: Mapping[str, Callable[[FitnessConfig], TriangleEvaluator]] = {
    "mse": lambda fitness: MSEEvaluator(),
    "regional": lambda fitness: RegionalMSEEvaluator(
        fitness.regional.grid_rows, fitness.regional.grid_cols, fitness.regional.detail_weight
    ),
    "ssim": lambda fitness: SSIMEvaluator(fitness.ssim.window_size, fitness.ssim.mse_weight),
    "blur": lambda fitness: BlurredMSEEvaluator(fitness.blur.sigma),
    "multiscale": lambda fitness: MultiScaleMSEEvaluator(fitness.multiscale.scales),
    "histogram": lambda fitness: ColorHistogramEvaluator(
        fitness.histogram.bins, fitness.histogram.mse_weight
    ),
    "edge": lambda fitness: EdgeMSEEvaluator(fitness.edge.sigma),
    "gradient": lambda fitness: GradientOrientationEvaluator(
        fitness.gradient.sigma, fitness.gradient.orientation_weight
    ),
    "chamfer": lambda fitness: ChamferEdgeEvaluator(
        fitness.chamfer.sigma, fitness.chamfer.threshold
    ),
    "saliency": lambda fitness: SaliencyMSEEvaluator(
        fitness.saliency.weight, fitness.saliency.sigma
    ),
}


def build_fitness_evaluator(fitness: FitnessConfig) -> TriangleEvaluator:
    """Construye el evaluador de la métrica elegida, sea simple o combinada."""
    if fitness.metric == "combo":
        return _build_combo_evaluator(fitness)
    return _METRIC_FACTORIES[fitness.metric](fitness)


def _build_combo_evaluator(fitness: FitnessConfig) -> CompositeEvaluator:
    """Combina métricas ya normalizadas al rango [0, 1]."""
    combo = fitness.combo or {}
    components = [
        (_METRIC_FACTORIES[name](fitness), weight)
        for name, weight in combo.items()
    ]
    return CompositeEvaluator(components)


def build_selection(
    selection: SelectionConfig, comparator: MSEComparator
) -> SelectionStrategy:
    """Construye la estrategia de selección de padres."""
    if selection.strategy == "elite":
        return EliteSelection(comparator)
    return ProbabilisticTournamentSelection(
        comparator, selection.tournament_size, selection.win_probability
    )


def build_survival(
    population: PopulationConfig, comparator: MSEComparator
) -> SurvivalStrategy:
    """Construye la estrategia que arma cada nueva generación."""
    if population.survival == "additive":
        return AdditiveSurvival(comparator)
    return ExclusiveSurvival(comparator)


def build_crossover(crossover: CrossoverConfig, codec: TriangleCodec) -> CrossoverStrategy:
    """Construye la estrategia de cruza para genomas de triángulos."""
    if crossover.strategy == "one-point":
        return OnePointCrossover(codec, RandomCutPointSelector())
    if crossover.strategy == "two-point":
        return TwoPointCrossover(codec, RandomTwoCutPointSelector())
    if crossover.strategy == "uniform":
        return UniformCrossover(codec, crossover.uniform_swap_probability)
    return AnnularCrossover(codec, RandomRingCutPointSelector())


def build_mutation_schedule(mutation: MutationConfig) -> TriangleMutationSchedule:
    """Construye la política de mutación; `constant` conserva el comportamiento base."""
    if mutation.schedule == "constant":
        return ConstantMutationSchedule(mutation.initial)

    initial, final = mutation.initial, mutation.final_or_initial
    if mutation.schedule == "linear":
        return LinearMutationSchedule(initial, final, mutation.decay_generations)

    exponential = ExponentialMutationSchedule(initial, final, mutation.decay_generations)
    if mutation.schedule == "exponential":
        return exponential

    reheat = mutation.reheat
    return AdaptiveReheatMutationSchedule(
        exponential,
        reheat.stagnation_generations,
        reheat.duration_generations,
        probability_multiplier=reheat.probability_multiplier,
        strength_multiplier=reheat.strength_multiplier,
        replacement_multiplier=reheat.replacement_multiplier,
        improvement_delta=reheat.improvement_delta or 0.0,
        improvement_percent=(
            0.0 if reheat.improvement_delta is not None else reheat.improvement_percent
        ),
    )
