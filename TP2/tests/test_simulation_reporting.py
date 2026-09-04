"""Pruebas de los observadores que escriben imágenes de una corrida."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageSequence

from simulation.config import GifConfig
from simulation.reporting import GifWriter
from triangle_image import (
    RandomTriangleInitializer,
    TriangleContext,
    TriangleImageTarget,
    TriangleIndividual,
)


@dataclass(frozen=True)
class _ScoredStub:
    individual: TriangleIndividual


@dataclass(frozen=True)
class _StateStub:
    """Instantánea mínima: el observador solo mira la generación y el mejor individuo."""

    generation: int
    population: tuple[_ScoredStub, ...]


def _individuals(count: int, width: int, height: int) -> tuple[TriangleIndividual, ...]:
    initializer = RandomTriangleInitializer(3, width, height)
    return tuple(initializer.create_initial_population(count, TriangleContext(1)))


class GifWriterTests(unittest.TestCase):
    """La animación es opcional, se muestrea por generación y sale en alta resolución."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.path = Path(self._directory.name) / "animaciones" / "evolucion.gif"
        self.target = TriangleImageTarget(Image.new("RGB", (64, 32)), max_size=16)
        self.individuals = _individuals(6, self.target.width, self.target.height)

    def _write(self, config: GifConfig, sampled_generations: range) -> Image.Image:
        writer = GifWriter(self.target, config)
        for generation in sampled_generations:
            writer.on_generation(
                _StateStub(generation, (_ScoredStub(self.individuals[generation % 6]),)), None
            )
        writer.finalize(self.individuals[0])
        gif = Image.open(self.path)
        self.addCleanup(gif.close)
        return gif

    def test_keeps_one_frame_per_sampled_generation_plus_the_final_one(self) -> None:
        gif = self._write(GifConfig(path=self.path, every=2), range(0, 6))
        self.assertEqual(len(tuple(ImageSequence.Iterator(gif))), 4)

    def test_frames_use_the_original_resolution_by_default(self) -> None:
        self.assertNotEqual(self.target.scale_factor, 1.0)
        gif = self._write(GifConfig(path=self.path), range(0, 1))
        self.assertEqual(gif.size, (self.target.orig_width, self.target.orig_height))

    def test_working_resolution_is_available_for_lighter_animations(self) -> None:
        gif = self._write(GifConfig(path=self.path, full_resolution=False), range(0, 1))
        self.assertEqual(gif.size, (self.target.width, self.target.height))

    def test_creates_the_parent_directory_of_the_animation(self) -> None:
        GifWriter(self.target, GifConfig(path=self.path))
        self.assertTrue(self.path.parent.is_dir())

    def test_writes_the_animation_even_without_sampled_generations(self) -> None:
        gif = self._write(GifConfig(path=self.path, every=100), range(0))
        self.assertEqual(len(tuple(ImageSequence.Iterator(gif))), 1)


if __name__ == "__main__":
    unittest.main()
