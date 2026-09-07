"""Pruebas de humo de las optimizaciones opcionales de simulación."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from simulation import SimulationConfig, run_simulation


class OptimizationSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.root = Path(self._directory.name)
        self.target = self.root / "target.png"
        Image.new("RGB", (8, 8), (230, 230, 230)).save(self.target)

    def config(self, strategy: str, **optimization: object) -> SimulationConfig:
        return SimulationConfig.from_mapping(
            {
                "image": str(self.target),
                "output": str(self.root / f"{strategy}.png"),
                "max_size": 8,
                "triangles": 2,
                "seed": 1,
                "progress_every": 0,
                "population": {"size": 4, "parents": 4, "generations": 1},
                "mutation": {"probability": 0.5, "strength": 0.05},
                "optimization": {"strategy": strategy, **optimization},
            }
        )

    def test_fitness_adaptive_mutation_runs(self) -> None:
        outcome = run_simulation(
            self.config(
                "fitness-adaptive-mutation",
                fitness_adaptive_mutation={
                    "min_multiplier": 0.5,
                    "max_multiplier": 1.5,
                },
            )
        )
        self.assertEqual(outcome.generations, 1)
        self.assertTrue(outcome.output_path.exists())

    def test_local_search_runs(self) -> None:
        outcome = run_simulation(
            self.config(
                "local-search",
                local_search={"attempts": 1, "every": 1, "strength_multiplier": 0.2},
            )
        )
        self.assertEqual(outcome.generations, 1)
        self.assertTrue(outcome.output_path.exists())

    def test_progressive_resolution_runs(self) -> None:
        outcome = run_simulation(
            self.config(
                "progressive-resolution",
                progressive_resolution={
                    "stages": [
                        {"max_size": 4, "generations": 1},
                        {"max_size": 8, "generations": 1},
                    ]
                },
            )
        )
        self.assertEqual(outcome.generations, 2)
        self.assertTrue(outcome.output_path.exists())

    def test_islands_run_and_keep_one_final_image(self) -> None:
        outcome = run_simulation(
            self.config(
                "islands",
                islands={"count": 2, "migration_every": 1, "migration_count": 1},
            )
        )
        self.assertEqual(outcome.generations, 1)
        self.assertTrue(outcome.output_path.exists())

    def test_parallel_islands_run_and_keep_one_final_image(self) -> None:
        outcome = run_simulation(
            self.config(
                "islands",
                islands={
                    "count": 2,
                    "migration_every": 1,
                    "migration_count": 1,
                    "parallel": True,
                    "workers": 2,
                },
            )
        )
        self.assertEqual(outcome.generations, 1)
        self.assertTrue(outcome.output_path.exists())

    def test_fitness_adaptive_mutation_can_wrap_spatial_guided_mutation(self) -> None:
        config = SimulationConfig.from_mapping(
            {
                "image": str(self.target),
                "output": str(self.root / "spatial-guided-adaptive.png"),
                "max_size": 8,
                "triangles": 2,
                "seed": 1,
                "progress_every": 0,
                "population": {"size": 4, "parents": 4, "generations": 1},
                "mutation": {
                    "strategy": "spatial-guided",
                    "probability": 0.5,
                    "strength": 0.05,
                },
                "optimization": {
                    "strategy": "fitness-adaptive-mutation",
                    "fitness_adaptive_mutation": {
                        "min_multiplier": 0.5,
                        "max_multiplier": 1.5,
                    },
                },
            },
        )
        outcome = run_simulation(config)
        self.assertEqual(outcome.generations, 1)
        self.assertTrue(outcome.output_path.exists())


if __name__ == "__main__":
    unittest.main()
