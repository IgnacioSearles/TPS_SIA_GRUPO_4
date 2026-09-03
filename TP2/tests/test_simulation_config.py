"""Pruebas de la carga, validación y construcción de simulaciones desde configuración."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from simulation import ConfigurationError, SimulationConfig, load_simulation_config
from simulation.builders import (
    build_crossover,
    build_fitness_evaluator,
    build_mutation_schedule,
    build_selection,
    build_survival,
)
from triangle_image import (
    AdaptiveReheatMutationSchedule,
    ColorHistogramEvaluator,
    CompositeEvaluator,
    ConstantMutationSchedule,
    MSEComparator,
    TriangleCodec,
)
from genetic_algorithm.application import (
    AdditiveSurvival,
    AnnularCrossover,
    BoltzmannSelection,
    DeterministicTournamentSelection,
    EliteSelection,
    ExclusiveSurvival,
    ProbabilisticTournamentSelection,
    RankingSelection,
    RouletteSelection,
    UniversalSelection,
    UniformCrossover,
)


def build_config(**overrides: Any) -> SimulationConfig:
    """Configuración mínima válida, con las claves indicadas agregadas o reemplazadas."""
    return SimulationConfig.from_mapping({"image": "target.png", **overrides})


class OptionalKeysTests(unittest.TestCase):
    def test_only_the_image_is_required(self) -> None:
        config = build_config()
        self.assertEqual(config.image, Path("target.png"))
        self.assertEqual(config.output, Path("output.png"))
        self.assertEqual(config.population.size, 100)
        self.assertEqual(config.fitness.metric, "mse")
        self.assertEqual(config.mutation.schedule, "constant")

    def test_missing_image_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            SimulationConfig.from_mapping({})
        self.assertIn("image", str(error.exception))

    def test_seed_is_optional(self) -> None:
        self.assertIsNone(build_config().seed)
        self.assertEqual(build_config(seed=7).seed, 7)

    def test_explicit_null_means_default(self) -> None:
        self.assertIsNone(build_config(seed=None).seed)
        self.assertIsNone(build_config(preview=None).preview)
        self.assertEqual(build_config(triangles=None).triangles, 50)

    def test_previews_are_disabled_unless_declared(self) -> None:
        self.assertIsNone(build_config().preview)

    def test_preview_section_enables_previews(self) -> None:
        config = build_config(preview={"directory": "previews/run", "every": 25})
        self.assertIsNotNone(config.preview)
        self.assertEqual(config.preview.directory, Path("previews/run"))
        self.assertEqual(config.preview.every, 25)
        self.assertFalse(config.preview.full_resolution)

    def test_preview_directory_is_required_when_previews_are_declared(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(preview={"every": 10})
        self.assertIn("preview.directory", str(error.exception))

    def test_partial_sections_keep_their_defaults(self) -> None:
        config = build_config(population={"parents": 40})
        self.assertEqual(config.population.parents, 40)
        self.assertEqual(config.population.size, 100)
        self.assertEqual(config.population.generations, 1000)


class UnknownKeyTests(unittest.TestCase):
    def test_unknown_top_level_key_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(trianlges=100)
        self.assertIn("trianlges", str(error.exception))

    def test_unknown_nested_key_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(population={"sixe": 10})
        self.assertIn("population", str(error.exception))
        self.assertIn("sixe", str(error.exception))

    def test_unknown_key_in_optional_section_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(preview={"directory": "previews", "evry": 5})
        self.assertIn("evry", str(error.exception))


class TypeAndRangeTests(unittest.TestCase):
    def test_wrong_type_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(triangles="muchos")
        self.assertIn("triangles", str(error.exception))

    def test_boolean_is_not_accepted_as_a_number(self) -> None:
        with self.assertRaises(ConfigurationError):
            build_config(triangles=True)

    def test_out_of_range_values_are_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            build_config(triangles=0)
        with self.assertRaises(ConfigurationError):
            build_config(mutation={"probability": 1.5})
        with self.assertRaises(ConfigurationError):
            build_config(preview={"directory": "previews", "every": 0})

    def test_invalid_choice_lists_the_valid_ones(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(crossover={"strategy": "three-point"})
        self.assertIn("one-point", str(error.exception))


class SemanticValidationTests(unittest.TestCase):
    def test_decaying_schedules_require_a_duration(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(mutation={"schedule": "linear"})
        self.assertIn("decay_generations", str(error.exception))

    def test_final_parameters_default_to_the_initial_ones(self) -> None:
        config = build_config(
            mutation={"schedule": "linear", "decay_generations": 100, "strength": 0.4}
        )
        self.assertEqual(config.mutation.final_or_initial, config.mutation.initial)
        self.assertEqual(config.mutation.final_or_initial.strength, 0.4)

    def test_final_section_inherits_unspecified_parameters(self) -> None:
        config = build_config(
            mutation={
                "schedule": "linear",
                "decay_generations": 100,
                "probability": 0.3,
                "strength": 0.4,
                "final": {"strength": 0.05},
            }
        )
        self.assertEqual(config.mutation.final_or_initial.strength, 0.05)
        self.assertEqual(config.mutation.final_or_initial.probability, 0.3)

    def test_exclusive_survival_requires_enough_offspring(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(population={"size": 100, "parents": 50, "survival": "exclusive"})
        self.assertIn("exclusive", str(error.exception))

    def test_combo_requires_weights(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(fitness={"metric": "combo"})
        self.assertIn("combo", str(error.exception))

    def test_combo_rejects_unknown_metrics(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(fitness={"metric": "combo", "combo": {"mse": 0.5, "sharpness": 0.5}})
        self.assertIn("sharpness", str(error.exception))

    def test_combo_weights_are_read(self) -> None:
        config = build_config(fitness={"metric": "combo", "combo": {"mse": 0.4, "edge": 0.6}})
        self.assertEqual(config.fitness.combo, {"mse": 0.4, "edge": 0.6})

    def test_boltzmann_temperature_must_be_positive(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            build_config(selection={"strategy": "boltzmann", "boltzmann_temperature": 0})
        self.assertIn("boltzmann_temperature", str(error.exception))


class FileLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)

    def write_config(self, data: Any) -> Path:
        path = self.root / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_loads_a_file(self) -> None:
        path = self.write_config({"image": "target.png", "triangles": 12})
        self.assertEqual(load_simulation_config(path).triangles, 12)

    def test_missing_file_reports_the_path(self) -> None:
        with self.assertRaises(ConfigurationError) as error:
            load_simulation_config(self.root / "nope.json")
        self.assertIn("nope.json", str(error.exception))

    def test_malformed_json_is_reported(self) -> None:
        path = self.root / "broken.json"
        path.write_text("{ not json", encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            load_simulation_config(path)

    def test_errors_mention_the_config_file(self) -> None:
        path = self.write_config({"image": "target.png", "triangles": -1})
        with self.assertRaises(ConfigurationError) as error:
            load_simulation_config(path)
        self.assertIn("config.json", str(error.exception))

    def test_overrides_replace_file_values(self) -> None:
        path = self.write_config({"image": "a.png", "seed": 1})
        config = load_simulation_config(path, {"image": "b.png", "seed": 99})
        self.assertEqual(config.image, Path("b.png"))
        self.assertEqual(config.seed, 99)

    def test_override_can_disable_previews(self) -> None:
        path = self.write_config({"image": "a.png", "preview": {"directory": "previews"}})
        self.assertIsNotNone(load_simulation_config(path).preview)
        self.assertIsNone(load_simulation_config(path, {"preview": None}).preview)

    def test_nested_overrides_keep_sibling_values(self) -> None:
        path = self.write_config({
            "image": "a.png",
            "mutation": {
                "strategy": "multigen",
                "schedule": "linear",
                "decay_generations": 250,
                "probability": 0.3,
                "strength": 0.2,
                "replacement_probability": 0.1,
            },
            "fitness": {"metric": "edge", "edge": {"sigma": 0.8}},
        })
        config = load_simulation_config(path, {
            "mutation": {"strategy": "gen"},
            "fitness": {"metric": "mse"},
        })
        self.assertEqual(config.mutation.strategy, "gen")
        self.assertEqual(config.mutation.schedule, "linear")
        self.assertEqual(config.mutation.decay_generations, 250)
        self.assertEqual(config.mutation.initial.strength, 0.2)
        self.assertEqual(config.fitness.metric, "mse")
        self.assertEqual(config.fitness.edge.sigma, 0.8)

    def test_bundled_configs_are_valid(self) -> None:
        for path in sorted(Path("configs").glob("*.json")):
            with self.subTest(config=path.name):
                load_simulation_config(path)


class BuilderTests(unittest.TestCase):
    def test_single_metric_builds_its_evaluator(self) -> None:
        config = build_config(fitness={"metric": "histogram", "histogram": {"bins": 8}})
        self.assertIsInstance(build_fitness_evaluator(config.fitness), ColorHistogramEvaluator)

    def test_combo_builds_a_composite_evaluator(self) -> None:
        config = build_config(fitness={"metric": "combo", "combo": {"mse": 0.5, "ssim": 0.5}})
        self.assertIsInstance(build_fitness_evaluator(config.fitness), CompositeEvaluator)

    def test_mutation_schedules_are_built_from_their_name(self) -> None:
        constant = build_mutation_schedule(build_config().mutation)
        self.assertIsInstance(constant, ConstantMutationSchedule)

        adaptive = build_mutation_schedule(
            build_config(
                mutation={
                    "schedule": "adaptive-reheat",
                    "decay_generations": 500,
                    "reheat": {"stagnation_generations": 10, "duration_generations": 5},
                }
            ).mutation
        )
        self.assertIsInstance(adaptive, AdaptiveReheatMutationSchedule)

    def test_selection_and_survival_are_built_from_their_name(self) -> None:
        comparator = MSEComparator()
        self.assertIsInstance(
            build_selection(build_config().selection, comparator), EliteSelection
        )
        self.assertIsInstance(
            build_selection(
                build_config(selection={"strategy": "tournament"}).selection, comparator
            ),
            ProbabilisticTournamentSelection,
        )
        self.assertIsInstance(
            build_selection(
                build_config(selection={"strategy": "probabilistic-tournament"}).selection,
                comparator,
            ),
            ProbabilisticTournamentSelection,
        )
        self.assertIsInstance(
            build_selection(
                build_config(selection={"strategy": "deterministic-tournament"}).selection,
                comparator,
            ),
            DeterministicTournamentSelection,
        )
        self.assertIsInstance(
            build_selection(build_config(selection={"strategy": "roulette"}).selection, comparator),
            RouletteSelection,
        )
        self.assertIsInstance(
            build_selection(build_config(selection={"strategy": "universal"}).selection, comparator),
            UniversalSelection,
        )
        self.assertIsInstance(
            build_selection(build_config(selection={"strategy": "boltzmann"}).selection, comparator),
            BoltzmannSelection,
        )
        self.assertIsInstance(
            build_selection(build_config(selection={"strategy": "ranking"}).selection, comparator),
            RankingSelection,
        )
        self.assertIsInstance(
            build_survival(build_config().population, comparator), AdditiveSurvival
        )
        self.assertIsInstance(
            build_survival(
                build_config(
                    population={"size": 10, "parents": 20, "survival": "exclusive"}
                ).population,
                comparator,
            ),
            ExclusiveSurvival,
        )

    def test_crossover_strategies_are_built_from_their_name(self) -> None:
        codec = TriangleCodec()
        self.assertIsInstance(
            build_crossover(build_config(crossover={"strategy": "uniform"}).crossover, codec),
            UniformCrossover,
        )
        self.assertIsInstance(
            build_crossover(build_config(crossover={"strategy": "annular"}).crossover, codec),
            AnnularCrossover,
        )


if __name__ == "__main__":
    unittest.main()
