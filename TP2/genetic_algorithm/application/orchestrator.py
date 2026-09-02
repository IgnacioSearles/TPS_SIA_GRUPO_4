"""Implementación genérica del ciclo de un algoritmo genético."""

from __future__ import annotations

from collections.abc import Collection
from functools import cmp_to_key
from typing import Any

from genetic_algorithm.application.contracts import (
    CrossoverStrategy,
    EvolutionConfiguration,
    EvolutionObserver,
    GeneticAlgorithm,
    MutationStrategy,
    ParentPairingStrategy,
    PopulationInitializer,
    SelectionStrategy,
    SurvivalStrategy,
    TerminationCondition,
)
from genetic_algorithm.domain.contracts import (
    EvolutionContext,
    EvolutionResult,
    EvolutionState,
    Fitness,
    GeneticProblem,
    ImageTarget,
    Individual,
    ScoredIndividual,
)


class DefaultScoredIndividual[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    ScoredIndividual[IndividualT, FitnessT]
):
    """Contenedor mínimo para relacionar un candidato con su fitness."""

    def __init__(self, individual: IndividualT, fitness: FitnessT) -> None:
        self._individual = individual
        self._fitness = fitness

    @property
    def individual(self) -> IndividualT:
        return self._individual

    @property
    def fitness(self) -> FitnessT:
        return self._fitness


class DefaultEvolutionState[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    EvolutionState[IndividualT, FitnessT]
):
    """Estado inmutable de una generación evaluada y ordenada por fitness."""

    def __init__(self, generation: int,
                 population: Collection[ScoredIndividual[IndividualT, FitnessT]]) -> None:
        self._generation = generation
        self._population = tuple(population)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def population(self) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        return self._population


class DefaultEvolutionResult[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    EvolutionResult[IndividualT, FitnessT]
):
    """Resultado mínimo que expone el último estado del ciclo."""

    def __init__(self, final_state: EvolutionState[IndividualT, FitnessT]) -> None:
        self._final_state = final_state

    @property
    def final_state(self) -> EvolutionState[IndividualT, FitnessT]:
        return self._final_state


class CompositeEvolutionObserver[IndividualT: Individual[Any], FitnessT: Fitness[Any]](
    EvolutionObserver[IndividualT, FitnessT]
):
    """Reenvía cada estado a varios observadores independientes."""

    def __init__(self, observers: Collection[EvolutionObserver[IndividualT, FitnessT]]) -> None:
        self._observers = tuple(observers)

    def on_generation(
        self, state: EvolutionState[IndividualT, FitnessT], context: EvolutionContext
    ) -> None:
        for observer in self._observers:
            observer.on_generation(state, context)


class OrchestratedGeneticAlgorithm[
    IndividualT: Individual[Any],
    TargetT: ImageTarget[Any],
    FitnessT: Fitness[Any],
](GeneticAlgorithm[
    GeneticProblem[IndividualT, TargetT, FitnessT],
    IndividualT,
    FitnessT,
    EvolutionConfiguration,
]):
    """Coordina operadores inyectados sin implementar ninguna estrategia."""

    def __init__(
        self,
        initializer: PopulationInitializer[IndividualT],
        selection: SelectionStrategy[IndividualT, FitnessT],
        pairing: ParentPairingStrategy[IndividualT, FitnessT],
        crossover: CrossoverStrategy[IndividualT],
        mutation: MutationStrategy[IndividualT],
        survival: SurvivalStrategy[IndividualT, FitnessT],
        termination: TerminationCondition[IndividualT, FitnessT],
        context: EvolutionContext,
        observer: EvolutionObserver[IndividualT, FitnessT] | None = None,
    ) -> None:
        self._initializer = initializer
        self._selection = selection
        self._pairing = pairing
        self._crossover = crossover
        self._mutation = mutation
        self._survival = survival
        self._termination = termination
        self._context = context
        self._observer = observer

    def run(
        self,
        problem: GeneticProblem[IndividualT, TargetT, FitnessT],
        configuration: EvolutionConfiguration,
    ) -> EvolutionResult[IndividualT, FitnessT]:
        """Ejecuta evaluación, reproducción y supervivencia hasta terminar."""
        initial_population = self._initializer.create_initial_population(
            configuration.population_size, self._context
        )
        state = self._create_state(problem, 0, initial_population)
        self._set_context_generation(state.generation)
        self._notify_observer(state)

        while not self._termination.should_stop(state, self._context):
            selected = self._selection.select(
                state.population, configuration.selected_parent_count, self._context
            )
            offspring = []
            for pair in self._pairing.pair(selected, self._context):
                crossed_individuals = self._crossover.cross(
                    pair.first_parent, pair.second_parent, self._context
                )
                offspring.extend(
                    self._mutation.mutate(individual, self._context)
                    for individual in crossed_individuals
                )

            evaluated_offspring = self._evaluate(problem, offspring)
            next_population = self._survival.build_next_generation(
                state.population,
                evaluated_offspring,
                configuration.population_size,
                self._context,
            )
            state = DefaultEvolutionState(
                state.generation + 1,
                self._order_by_fitness(next_population, problem),
            )
            self._set_context_generation(state.generation)
            self._notify_observer(state)

        return DefaultEvolutionResult(state)

    def _notify_observer(self, state: EvolutionState[IndividualT, FitnessT]) -> None:
        if self._observer is not None:
            self._observer.on_generation(state, self._context)

    def _set_context_generation(self, generation: int) -> None:
        """Actualiza contextos que optan por adaptar operadores según la generación."""
        set_generation = getattr(self._context, "set_generation", None)
        if callable(set_generation):
            set_generation(generation)

    def _create_state(
        self,
        problem: GeneticProblem[IndividualT, TargetT, FitnessT],
        generation: int,
        population: Collection[IndividualT],
    ) -> EvolutionState[IndividualT, FitnessT]:
        return DefaultEvolutionState(
            generation,
            self._order_by_fitness(self._evaluate(problem, population), problem),
        )

    def _order_by_fitness(
        self,
        population: Collection[ScoredIndividual[IndividualT, FitnessT]],
        problem: GeneticProblem[IndividualT, TargetT, FitnessT],
    ) -> tuple[ScoredIndividual[IndividualT, FitnessT], ...]:
        """Normaliza el orden observable del estado sin imponerlo a las estrategias."""
        comparator = problem.fitness_comparator

        def compare(
            left: ScoredIndividual[IndividualT, FitnessT],
            right: ScoredIndividual[IndividualT, FitnessT],
        ) -> int:
            if comparator.is_better(left.fitness, right.fitness):
                return -1
            if comparator.is_better(right.fitness, left.fitness):
                return 1
            return 0

        return tuple(sorted(population, key=cmp_to_key(compare)))

    def _evaluate(
        self,
        problem: GeneticProblem[IndividualT, TargetT, FitnessT],
        individuals: Collection[IndividualT],
    ) -> Collection[ScoredIndividual[IndividualT, FitnessT]]:
        return tuple(
            DefaultScoredIndividual(
                individual,
                problem.fitness_evaluator.evaluate(
                    individual, problem.target, self._context
                ),
            )
            for individual in individuals
        )
