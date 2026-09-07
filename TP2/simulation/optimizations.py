"""Optimizaciones opcionales para la corrida de aproximacion de imagenes."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from genetic_algorithm.application import MutationStrategy, PopulationInitializer
from genetic_algorithm.domain import EvolutionContext, FitnessComparator, FitnessEvaluator
from triangle_image import (
    ConstantMutationSchedule,
    MixedTriangleGeneMutator,
    MutationParameters,
    RandomTriangleInitializer,
    TriangleCodec,
    TriangleImageTarget,
    TriangleIndividual,
    TriangleMutationSchedule,
)

from simulation.builders import build_mutation
from simulation.config import (
    FitnessAdaptiveMutationOptimizationConfig,
    LocalSearchOptimizationConfig,
    MutationConfig,
)


def scale_between_targets(
    individual: TriangleIndividual,
    source: TriangleImageTarget,
    destination: TriangleImageTarget,
) -> TriangleIndividual:
    """Escala un individuo desde la resolucion de trabajo de un target a otro."""
    if source.scale_factor == destination.scale_factor:
        return individual
    return individual.scale(destination.scale_factor / source.scale_factor)


class SeededTriangleInitializer(PopulationInitializer[TriangleIndividual]):
    """Incluye individuos semilla y completa el resto de la poblacion al azar."""

    def __init__(
        self,
        seeds: Collection[TriangleIndividual],
        triangles_per_individual: int,
        width: int,
        height: int,
    ) -> None:
        self._seeds = tuple(seeds)
        self._random = RandomTriangleInitializer(triangles_per_individual, width, height)

    def create_initial_population(
        self, population_size: int, context: EvolutionContext
    ) -> Collection[TriangleIndividual]:
        random_count = max(0, population_size - len(self._seeds))
        random_population = self._random.create_initial_population(random_count, context)
        return (*self._seeds[:population_size], *random_population)


class FitnessAdaptiveMutation(MutationStrategy[TriangleIndividual]):
    """Aumenta o reduce la mutacion de cada hijo segun su fitness previo."""

    def __init__(
        self,
        mutation: MutationConfig,
        width: int,
        height: int,
        codec: TriangleCodec,
        schedule: TriangleMutationSchedule,
        target: TriangleImageTarget,
        evaluator: FitnessEvaluator,
        parameters: FitnessAdaptiveMutationOptimizationConfig,
        initial_triangles: int,
    ) -> None:
        self._mutation = mutation
        self._width = width
        self._height = height
        self._codec = codec
        self._schedule = schedule
        self._target = target
        self._evaluator = evaluator
        self._parameters = parameters
        self._initial_triangles = initial_triangles

    def mutate(
        self, individual: TriangleIndividual, context: EvolutionContext
    ) -> TriangleIndividual:
        fitness = _evaluate_value(self._evaluator, individual, self._target, context)
        multiplier = _fitness_multiplier(
            fitness,
            self._parameters.min_multiplier,
            self._parameters.max_multiplier,
        )
        base = self._schedule.parameters_at(getattr(context, "generation", 0))
        scaled = _scale_parameters(base, multiplier)
        temporary_schedule = ConstantMutationSchedule(scaled)
        return build_mutation(
            self._mutation,
            self._width,
            self._height,
            self._codec,
            temporary_schedule,
            self._initial_triangles,
        ).mutate(individual, context)


class LocalSearchMutation(MutationStrategy[TriangleIndividual]):
    """Aplica la mutacion base y luego acepta pequenas mejoras sobre un triangulo."""

    def __init__(
        self,
        base_mutation: MutationStrategy[TriangleIndividual],
        width: int,
        height: int,
        codec: TriangleCodec,
        schedule: TriangleMutationSchedule,
        target: TriangleImageTarget,
        evaluator: FitnessEvaluator,
        comparator: FitnessComparator,
        parameters: LocalSearchOptimizationConfig,
    ) -> None:
        self._base_mutation = base_mutation
        self._width = width
        self._height = height
        self._codec = codec
        self._schedule = schedule
        self._target = target
        self._evaluator = evaluator
        self._comparator = comparator
        self._parameters = parameters

    def mutate(
        self, individual: TriangleIndividual, context: EvolutionContext
    ) -> TriangleIndividual:
        candidate = self._base_mutation.mutate(individual, context)
        generation = getattr(context, "generation", 0)
        if generation % self._parameters.every != 0:
            return candidate

        best = candidate
        best_fitness = self._evaluator.evaluate(best, self._target, context)
        strength = (
            self._schedule.parameters_at(generation).strength
            * self._parameters.strength_multiplier
        )
        mutator = MixedTriangleGeneMutator(
            self._width,
            self._height,
            mutation_strength=strength,
            replacement_probability=0.0,
        )
        for _ in range(self._parameters.attempts):
            genes = list(self._codec.extract_genes(best))
            if not genes:
                break
            position = context.random_generator.randrange(len(genes))
            genes[position] = mutator.mutate_gene(genes[position], context)
            refined = self._codec.build_individual(genes)
            refined_fitness = self._evaluator.evaluate(refined, self._target, context)
            if self._comparator.is_better(refined_fitness, best_fitness):
                best = refined
                best_fitness = refined_fitness
        return best


def _evaluate_value(
    evaluator: FitnessEvaluator,
    individual: TriangleIndividual,
    target: TriangleImageTarget,
    context: EvolutionContext,
) -> float:
    return float(evaluator.evaluate(individual, target, context).value)


def _fitness_multiplier(fitness: float, minimum: float, maximum: float) -> float:
    bounded = min(1.0, max(0.0, fitness))
    return maximum - bounded * (maximum - minimum)


def _scale_parameters(parameters: MutationParameters, multiplier: float) -> MutationParameters:
    return MutationParameters(
        probability=min(1.0, parameters.probability * multiplier),
        strength=parameters.strength * multiplier,
        replacement_probability=min(1.0, parameters.replacement_probability * multiplier),
    )


@dataclass(slots=True)
class IslandRuntime:
    """Estado y operadores de una isla."""

    state: object
    context: EvolutionContext
    selection: object
    pairing: object
    crossover: object
    mutation: object
    survival: object
    schedule: TriangleMutationSchedule
