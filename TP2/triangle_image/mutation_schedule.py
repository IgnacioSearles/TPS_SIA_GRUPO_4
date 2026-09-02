"""Políticas para variar la intensidad de mutación durante la evolución."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

from genetic_algorithm.application.contracts import EvolutionObserver, GenePositionSelector
from genetic_algorithm.domain.contracts import EvolutionContext, EvolutionState
from triangle_image.fitness import MSEFitness
from triangle_image.gene import TriangleIndividual


@dataclass(frozen=True, slots=True)
class MutationParameters:
    probability: float
    strength: float
    replacement_probability: float


class TriangleMutationSchedule(ABC):
    """Estrategia que decide parámetros de mutación según la generación."""

    @abstractmethod
    def parameters_at(self, generation: int) -> MutationParameters:
        """Devuelve los parámetros a usar durante la generación indicada."""

    def observe_best(self, generation: int, fitness: float) -> None:
        """Recibe el mejor fitness; solo las políticas adaptativas lo necesitan."""


class ConstantMutationSchedule(TriangleMutationSchedule):
    """Mantiene los mismos parámetros de mutación durante toda la ejecución."""

    def __init__(self, parameters: MutationParameters) -> None:
        self._parameters = parameters

    def parameters_at(self, generation: int) -> MutationParameters:
        return self._parameters


class LinearMutationSchedule(TriangleMutationSchedule):
    """Interpola linealmente entre parámetros iniciales y finales."""

    def __init__(self, initial: MutationParameters, final: MutationParameters, duration: int) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        self._initial, self._final, self._duration = initial, final, duration

    def parameters_at(self, generation: int) -> MutationParameters:
        progress = min(max(generation, 0) / self._duration, 1.0)
        return MutationParameters(*(
            start + progress * (end - start)
            for start, end in zip(
                (self._initial.probability, self._initial.strength, self._initial.replacement_probability),
                (self._final.probability, self._final.strength, self._final.replacement_probability),
            )
        ))


class ExponentialMutationSchedule(TriangleMutationSchedule):
    """Se enfría rápido al inicio y se aproxima suavemente a los valores finales."""

    def __init__(self, initial: MutationParameters, final: MutationParameters, time_constant: int) -> None:
        if time_constant <= 0:
            raise ValueError("time_constant must be positive")
        self._initial, self._final, self._time_constant = initial, final, time_constant

    def parameters_at(self, generation: int) -> MutationParameters:
        remaining = math.exp(-max(generation, 0) / self._time_constant)
        return MutationParameters(*(
            end + (start - end) * remaining
            for start, end in zip(
                (self._initial.probability, self._initial.strength, self._initial.replacement_probability),
                (self._final.probability, self._final.strength, self._final.replacement_probability),
            )
        ))


class AdaptiveReheatMutationSchedule(TriangleMutationSchedule):
    """Envuelve un enfriamiento y multiplica mutaciones temporalmente ante estancamiento."""

    def __init__(
        self,
        base: TriangleMutationSchedule,
        stagnation_generations: int = 100,
        reheat_generations: int = 40,
        probability_multiplier: float = 2.0,
        strength_multiplier: float = 2.0,
        replacement_multiplier: float = 3.0,
    ) -> None:
        if stagnation_generations <= 0 or reheat_generations <= 0:
            raise ValueError("stagnation and reheat durations must be positive")
        self._base = base
        self._stagnation_generations = stagnation_generations
        self._reheat_generations = reheat_generations
        self._multipliers = probability_multiplier, strength_multiplier, replacement_multiplier
        self._best_fitness: float | None = None
        self._last_improvement_generation = 0
        self._reheat_until = -1

    def observe_best(self, generation: int, fitness: float) -> None:
        if self._best_fitness is None or fitness < self._best_fitness:
            self._best_fitness = fitness
            self._last_improvement_generation = generation
        elif generation >= self._reheat_until and generation - self._last_improvement_generation >= self._stagnation_generations:
            self._reheat_until = generation + self._reheat_generations
            self._last_improvement_generation = generation

    def parameters_at(self, generation: int) -> MutationParameters:
        base = self._base.parameters_at(generation)
        if generation >= self._reheat_until:
            return base
        probability, strength, replacement = (
            value * multiplier for value, multiplier in zip(
                (base.probability, base.strength, base.replacement_probability),
                self._multipliers,
            )
        )
        return MutationParameters(min(1.0, probability), strength, min(1.0, replacement))


class ScheduledGenePositionSelector(GenePositionSelector):
    """Selecciona genes usando la probabilidad indicada por una política concreta."""

    def __init__(self, schedule: TriangleMutationSchedule) -> None:
        self._schedule = schedule

    def select_positions(self, genome_size: int, context: EvolutionContext) -> tuple[int, ...]:
        generation = getattr(context, "generation", 0)
        probability = self._schedule.parameters_at(generation).probability
        return tuple(
            index for index in range(genome_size)
            if context.random_generator.random() < probability
        )


class MutationScheduleObserver(EvolutionObserver[TriangleIndividual, MSEFitness]):
    """Conecta el mejor fitness de cada estado con una política adaptativa."""

    def __init__(self, schedule: TriangleMutationSchedule) -> None:
        self._schedule = schedule

    def on_generation(
        self, state: EvolutionState[TriangleIndividual, MSEFitness], context: EvolutionContext
    ) -> None:
        best = next(iter(state.population), None)
        if best is not None:
            self._schedule.observe_best(state.generation, best.fitness.value)
