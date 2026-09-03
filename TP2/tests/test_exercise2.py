"""Pruebas de las integraciones agregadas para el Ejercicio 2."""

import unittest

from genetic_algorithm.application import (
    GenMutation, MultiGenMutation, NonUniformMutation, StagnationTermination,
    UniformMutation,
)
from simulation import SimulationConfig, expand_matrix, load_experiment_spec, load_simulation_config
from simulation.builders import build_mutation
from triangle_image import MSEComparator, TriangleCodec


class Exercise2IntegrationTests(unittest.TestCase):
    def test_all_four_mutation_variants_are_configurable(self) -> None:
        expected = {
            "gen": GenMutation,
            "multigen": MultiGenMutation,
            "uniform": UniformMutation,
            "non-uniform": NonUniformMutation,
        }
        for name, operator_type in expected.items():
            with self.subTest(strategy=name):
                config = SimulationConfig.from_mapping({"image": "target.png", "mutation": {"strategy": name}})
                self.assertIsInstance(build_mutation(config.mutation, 20, 20, TriangleCodec()), operator_type)

    def test_stagnation_termination_stops_after_configured_generations(self) -> None:
        self.assertEqual(StagnationTermination(2)._generations, 2)

    def test_experiment_matrix_is_cartesian_product(self) -> None:
        self.assertEqual(len(expand_matrix({"triangles": [1, 2], "seed": [3, 4]})), 4)

    def test_exhaustive_mutation_fitness_spec_has_expected_design(self) -> None:
        config, matrix, _ = load_experiment_spec("experiments/mutation_fitness_exhaustive.json")
        self.assertEqual(len(expand_matrix(matrix)), 480)
        self.assertEqual(
            [(item["path"], item["generations"]) for item in matrix["image"]],
            [
                ("imagenes/italia.png", 2000),
                ("imagenes/Firefox_logo,_2017.png", 2500),
                ("imagenes/monalisa.jpg", 3000),
            ],
        )
        self.assertEqual(matrix["mutation.strategy"], ["gen", "multigen", "uniform", "non-uniform"])
        self.assertEqual(len(matrix["mutation.schedule"]), 4)
        self.assertEqual(len(matrix["fitness"]), 10)
        self.assertEqual(sum(item["metric"] == "combo" for item in matrix["fitness"]), 7)
        self.assertIn(
            {"metric": "combo", "combo": {
                "mse": 0.35, "blur": 0.20, "edge": 0.20, "saliency": 0.25
            }},
            matrix["fitness"],
        )
        base = load_simulation_config(config)
        self.assertEqual(base.max_size, 96)
        self.assertEqual(base.triangles, 80)
        self.assertEqual(base.selection.strategy, "tournament")


if __name__ == "__main__":
    unittest.main()
