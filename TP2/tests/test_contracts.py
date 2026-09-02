"""Pruebas estructurales de las abstracciones públicas."""

from __future__ import annotations

import unittest
from random import Random

from genetic_algorithm.application import (AnnularCrossover, CrossoverStrategy, GeneticAlgorithm,
    AdditiveSurvival, CutPointSelector, DefaultScoredIndividual,
    EvolutionConfiguration, EvolutionObserver, GeneMutator, GenePositionSelector, GenomeCodec,
    MutationStrategy, OnePointCrossover, OrchestratedGeneticAlgorithm, ParentPair,
    ParentPairingStrategy, PopulationInitializer, SelectionStrategy, SurvivalStrategy,
    TerminationCondition, EliteSelection, ExclusiveSurvival, MultiGeneMutation,
    RandomPairingStrategy, RingCutPointSelector, TwoCutPointSelector, TwoPointCrossover,
    UniformCrossover)
from genetic_algorithm.domain import (AlgorithmConfiguration, EvolutionContext,
    EvolutionResult, EvolutionState, Fitness, FitnessComparator, FitnessEvaluator,
    GeneticProblem, ImageTarget, Individual, ScoredIndividual)


class ContractAbstractnessTests(unittest.TestCase):
    def test_domain_contracts_cannot_be_instantiated(self) -> None:
        for contract in (Individual, Fitness, ImageTarget, EvolutionContext,
                         ScoredIndividual, FitnessEvaluator, FitnessComparator,
                         GeneticProblem, AlgorithmConfiguration, EvolutionState,
                         EvolutionResult):
            with self.subTest(contract=contract.__name__):
                with self.assertRaises(TypeError):
                    contract()

    def test_application_contracts_cannot_be_instantiated(self) -> None:
        for contract in (SelectionStrategy, ParentPair, ParentPairingStrategy,
                         PopulationInitializer, SurvivalStrategy, CrossoverStrategy,
                         MutationStrategy, TerminationCondition, GeneticAlgorithm,
                         EvolutionObserver, TwoCutPointSelector, RingCutPointSelector):
            with self.subTest(contract=contract.__name__):
                with self.assertRaises(TypeError):
                    contract()


class ConcreteIndividual(Individual[str]):
    @property
    def genome(self) -> object:
        return "any genome representation"


class ConcreteFitness(Fitness[float]):
    @property
    def value(self) -> float:
        return 0.0


class ConcreteTarget(ImageTarget[bytes]):
    @property
    def image(self) -> bytes:
        return b"any image format"


class ConcreteContext(EvolutionContext):
    def __init__(self, seed: int = 0) -> None:
        self._random_generator = Random(seed)

    @property
    def random_generator(self) -> Random:
        return self._random_generator

    @property
    def data(self) -> object:
        return {"generation": 0}


class ConcreteEvaluator(FitnessEvaluator[ConcreteIndividual, ConcreteTarget, ConcreteFitness]):
    def evaluate(self, individual: ConcreteIndividual, target: ConcreteTarget,
                 context: EvolutionContext) -> ConcreteFitness:
        return ConcreteFitness()


class ConcreteComparator(FitnessComparator[ConcreteFitness]):
    def is_better(self, left: ConcreteFitness, right: ConcreteFitness) -> bool:
        return left.value > right.value


class ContractSpecializationTests(unittest.TestCase):
    def test_domain_contracts_accept_concrete_type_parameters(self) -> None:
        individual = ConcreteIndividual()
        target = ConcreteTarget()
        fitness = ConcreteEvaluator().evaluate(individual, target, ConcreteContext())

        self.assertEqual(individual.genome, "any genome representation")
        self.assertEqual(target.image, b"any image format")
        self.assertEqual(fitness.value, 0.0)
        self.assertFalse(ConcreteComparator().is_better(fitness, fitness))


class ConcreteProblem(GeneticProblem[ConcreteIndividual, ConcreteTarget, ConcreteFitness]):
    @property
    def target(self) -> ConcreteTarget:
        return ConcreteTarget()

    @property
    def fitness_evaluator(self) -> ConcreteEvaluator:
        return ConcreteEvaluator()

    @property
    def fitness_comparator(self) -> ConcreteComparator:
        return ConcreteComparator()


class ConcreteConfiguration(EvolutionConfiguration):
    @property
    def data(self) -> object:
        return {}

    @property
    def population_size(self) -> int:
        return 1

    @property
    def selected_parent_count(self) -> int:
        return 1


class ConcreteInitializer(PopulationInitializer[ConcreteIndividual]):
    def create_initial_population(self, population_size: int,
                                  context: EvolutionContext) -> list[ConcreteIndividual]:
        return [ConcreteIndividual()]


class ConcretePair(ParentPair[ConcreteIndividual]):
    @property
    def first_parent(self) -> ConcreteIndividual:
        return ConcreteIndividual()

    @property
    def second_parent(self) -> ConcreteIndividual:
        return ConcreteIndividual()


class ConcreteSelection(SelectionStrategy[ConcreteIndividual, ConcreteFitness]):
    def select(self, population, amount: int, context: EvolutionContext):
        return population


class ConcretePairing(ParentPairingStrategy[ConcreteIndividual, ConcreteFitness]):
    def pair(self, selected, context: EvolutionContext) -> list[ConcretePair]:
        return [ConcretePair()]


class ConcreteCrossover(CrossoverStrategy[ConcreteIndividual]):
    def cross(self, parent_a: ConcreteIndividual, parent_b: ConcreteIndividual,
              context: EvolutionContext) -> list[ConcreteIndividual]:
        return [parent_a]


class ConcreteMutation(MutationStrategy[ConcreteIndividual]):
    def mutate(self, individual: ConcreteIndividual,
               context: EvolutionContext) -> ConcreteIndividual:
        return individual


class ConcreteSurvival(SurvivalStrategy[ConcreteIndividual, ConcreteFitness]):
    def build_next_generation(self, current_population, offspring, population_size: int,
                              context: EvolutionContext):
        return offspring


class StopAtSecondGeneration(TerminationCondition[ConcreteIndividual, ConcreteFitness]):
    def should_stop(self, state, context: EvolutionContext) -> bool:
        return state.generation >= 2


class OrchestratorTests(unittest.TestCase):
    def test_orchestrator_composes_injected_generic_strategies(self) -> None:
        algorithm = OrchestratedGeneticAlgorithm(
            initializer=ConcreteInitializer(),
            selection=ConcreteSelection(),
            pairing=ConcretePairing(),
            crossover=ConcreteCrossover(),
            mutation=ConcreteMutation(),
            survival=ConcreteSurvival(),
            termination=StopAtSecondGeneration(),
            context=ConcreteContext(),
        )

        result = algorithm.run(ConcreteProblem(), ConcreteConfiguration())

        self.assertEqual(result.final_state.generation, 2)
        self.assertEqual(len(result.final_state.population), 1)


class GeneIndividual(Individual[str]):
    def __init__(self, genes: tuple[str, ...]) -> None:
        self.genes = genes

    @property
    def genome(self) -> object:
        return self.genes


class TupleGenomeCodec(GenomeCodec[GeneIndividual, str]):
    def extract_genes(self, individual: GeneIndividual) -> tuple[str, ...]:
        return individual.genes

    def build_individual(self, genes) -> GeneIndividual:
        return GeneIndividual(tuple(genes))


class FixedCutPoint(CutPointSelector):
    def select_cut_point(self, genome_size: int, context: EvolutionContext) -> int:
        return 2


class FixedTwoCutPoints(TwoCutPointSelector):
    def select_cut_points(self, genome_size: int, context: EvolutionContext) -> tuple[int, int]:
        return 1, 4


class FixedRingCutPoints(RingCutPointSelector):
    def select_ring_cut_points(self, genome_size: int, context: EvolutionContext) -> tuple[int, int]:
        return 3, 1


class FixedPositions(GenePositionSelector):
    def select_positions(self, genome_size: int, context: EvolutionContext) -> tuple[int, ...]:
        return (0, 2)


class MarkGeneMutator(GeneMutator[str]):
    def mutate_gene(self, gene: str, context: EvolutionContext) -> str:
        return f"{gene}*"


class RankedFitness(Fitness[int]):
    def __init__(self, value: int) -> None:
        self._value = value

    @property
    def value(self) -> int:
        return self._value


class RankedComparator(FitnessComparator[RankedFitness]):
    def is_better(self, left: RankedFitness, right: RankedFitness) -> bool:
        return left.value > right.value


class StrategyImplementationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ConcreteContext()

    def test_one_point_crossover_uses_injected_codec_and_cut_point(self) -> None:
        crossover = OnePointCrossover(TupleGenomeCodec(), FixedCutPoint())
        children = crossover.cross(
            GeneIndividual(("a", "b", "c", "d")),
            GeneIndividual(("1", "2", "3", "4")),
            self.context,
        )
        self.assertEqual([child.genes for child in children], [
            ("a", "b", "3", "4"), ("1", "2", "c", "d"),
        ])

    def test_two_point_crossover_swaps_only_the_inner_segment(self) -> None:
        crossover = TwoPointCrossover(TupleGenomeCodec(), FixedTwoCutPoints())
        children = crossover.cross(
            GeneIndividual(("a", "b", "c", "d", "e")),
            GeneIndividual(("1", "2", "3", "4", "5")),
            self.context,
        )
        self.assertEqual([child.genes for child in children], [
            ("a", "2", "3", "4", "e"), ("1", "b", "c", "d", "5"),
        ])

    def test_uniform_crossover_can_swap_every_gene(self) -> None:
        crossover = UniformCrossover(TupleGenomeCodec(), swap_probability=1.0)
        children = crossover.cross(
            GeneIndividual(("a", "b", "c")),
            GeneIndividual(("1", "2", "3")),
            self.context,
        )
        self.assertEqual([child.genes for child in children], [
            ("1", "2", "3"), ("a", "b", "c"),
        ])

    def test_annular_crossover_swaps_a_segment_that_wraps_the_genome_end(self) -> None:
        crossover = AnnularCrossover(TupleGenomeCodec(), FixedRingCutPoints())
        children = crossover.cross(
            GeneIndividual(("a", "b", "c", "d", "e")),
            GeneIndividual(("1", "2", "3", "4", "5")),
            self.context,
        )
        self.assertEqual([child.genes for child in children], [
            ("1", "b", "c", "4", "5"), ("a", "2", "3", "d", "e"),
        ])

    def test_multi_gene_mutation_uses_injected_positions_and_gene_mutator(self) -> None:
        mutation = MultiGeneMutation(TupleGenomeCodec(), FixedPositions(), MarkGeneMutator())
        mutated = mutation.mutate(GeneIndividual(("a", "b", "c")), self.context)
        self.assertEqual(mutated.genes, ("a*", "b", "c*"))

    def test_elite_and_additive_survival_use_the_injected_fitness_comparator(self) -> None:
        population = tuple(
            DefaultScoredIndividual(GeneIndividual((str(score),)), RankedFitness(score))
            for score in (1, 3, 2)
        )
        comparator = RankedComparator()
        elite = EliteSelection(comparator).select(population, 2, self.context)
        survivors = AdditiveSurvival(comparator).build_next_generation(
            population[:1], population[1:], 2, self.context
        )

        self.assertEqual([candidate.fitness.value for candidate in elite], [3, 2])
        self.assertEqual([candidate.fitness.value for candidate in survivors], [3, 2])

    def test_exclusive_survival_uses_only_ranked_offspring(self) -> None:
        population = tuple(
            DefaultScoredIndividual(GeneIndividual((str(score),)), RankedFitness(score))
            for score in (3, 2)
        )
        offspring = tuple(
            DefaultScoredIndividual(GeneIndividual((str(score),)), RankedFitness(score))
            for score in (1, 4, 2)
        )

        survivors = ExclusiveSurvival(RankedComparator()).build_next_generation(
            population, offspring, 2, self.context
        )

        self.assertEqual([candidate.fitness.value for candidate in survivors], [4, 2])

    def test_random_pairing_is_reproducible_with_the_context_seed(self) -> None:
        population = tuple(
            DefaultScoredIndividual(GeneIndividual((str(score),)), RankedFitness(score))
            for score in range(6)
        )
        pairing = RandomPairingStrategy[GeneIndividual, RankedFitness]()

        first = pairing.pair(population, ConcreteContext(seed=42))
        second = pairing.pair(population, ConcreteContext(seed=42))

        first_pairs = [(pair.first_parent.genes, pair.second_parent.genes) for pair in first]
        second_pairs = [(pair.first_parent.genes, pair.second_parent.genes) for pair in second]
        self.assertEqual(first_pairs, second_pairs)
