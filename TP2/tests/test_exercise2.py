"""Pruebas de las integraciones agregadas para el Ejercicio 2."""

import unittest

from genetic_algorithm.application import (
    GenMutation, MultiGenMutation, NonUniformMutation, StagnationTermination,
    UniformMutation,
)
from simulation import SimulationConfig, expand_matrix
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


if __name__ == "__main__":
    unittest.main()
