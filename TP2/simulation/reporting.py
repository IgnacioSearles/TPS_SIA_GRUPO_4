"""Observadores que informan el avance de una corrida y guardan previews."""

from __future__ import annotations

from time import perf_counter

from genetic_algorithm.application import EvolutionObserver
from genetic_algorithm.domain import EvolutionContext, EvolutionState
from triangle_image import MSEFitness, TriangleImageTarget, TriangleIndividual
from triangle_image.rendering import render

from simulation.config import PreviewConfig


class PreviewWriter(EvolutionObserver[TriangleIndividual, MSEFitness]):
    """Guarda periódicamente una imagen del mejor individuo de la generación."""

    def __init__(self, target: TriangleImageTarget, config: PreviewConfig) -> None:
        self._target = target
        self._config = config
        config.directory.mkdir(parents=True, exist_ok=True)

    def on_generation(
        self,
        state: EvolutionState[TriangleIndividual, MSEFitness],
        context: EvolutionContext,
    ) -> None:
        if state.generation % self._config.every != 0:
            return
        best = next(iter(state.population), None)
        if best is None:
            return
        image = render(*self._scaled_for_output(best.individual))
        image.save(self._config.directory / f"generation_{state.generation:05d}.png")

    def _scaled_for_output(
        self, individual: TriangleIndividual
    ) -> tuple[TriangleIndividual, int, int]:
        """Devuelve el individuo y el lienzo en la resolución pedida para el preview."""
        if not self._config.full_resolution or self._target.scale_factor == 1.0:
            return individual, self._target.width, self._target.height
        return (
            individual.scale(1.0 / self._target.scale_factor),
            self._target.orig_width,
            self._target.orig_height,
        )


class ProgressReporter(EvolutionObserver[TriangleIndividual, MSEFitness]):
    """Imprime el mejor fitness y los tiempos cada `report_every` generaciones."""

    def __init__(self, report_every: int) -> None:
        self._report_every = report_every
        self._started_at = perf_counter()
        self._last_report_at = self._started_at

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self._started_at

    def on_generation(
        self,
        state: EvolutionState[TriangleIndividual, MSEFitness],
        context: EvolutionContext,
    ) -> None:
        if not self._report_every or state.generation % self._report_every != 0:
            return
        best = next(iter(state.population), None)
        if best is None:
            return
        now = perf_counter()
        print(
            f"Generación {state.generation:>5} | mejor fitness: {best.fitness.value:.6f} "
            f"| transcurrido: {now - self._started_at:.1f}s "
            f"| últimas {self._report_every}: {now - self._last_report_at:.1f}s"
        )
        self._last_report_at = now
