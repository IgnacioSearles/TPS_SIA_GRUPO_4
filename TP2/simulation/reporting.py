"""Observadores que informan el avance de una corrida y guardan sus imágenes."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from time import perf_counter, process_time
from pathlib import Path

from genetic_algorithm.application import EvolutionObserver
from genetic_algorithm.domain import EvolutionContext, EvolutionState
from triangle_image import MSEFitness, TriangleImageTarget, TriangleIndividual
from triangle_image.rendering import render

from PIL import Image

from simulation.config import GifConfig, PreviewConfig
from triangle_image.gene import TriangleGene


def render_individual(
    individual: TriangleIndividual, target: TriangleImageTarget, full_resolution: bool
) -> Image.Image:
    """Dibuja al individuo en la resolución de trabajo o en la original del objetivo."""
    if not full_resolution or target.scale_factor == 1.0:
        return render(individual, target.width, target.height)
    return render(
        individual.scale(1.0 / target.scale_factor), target.orig_width, target.orig_height
    )


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
        image = render_individual(best.individual, self._target, self._config.full_resolution)
        image.save(self._config.directory / f"generation_{state.generation:05d}.png")


class GifWriter(EvolutionObserver[TriangleIndividual, MSEFitness]):
    """Acumula cuadros del mejor individuo y arma un GIF animado al terminar.

    Los cuadros se guardan ya cuantizados a paleta: el GIF los necesita así y de
    paso la corrida retiene un tercio de la memoria que ocuparían en RGB.
    """

    def __init__(self, target: TriangleImageTarget, config: GifConfig) -> None:
        self._target = target
        self._config = config
        self._frames: list[Image.Image] = []
        config.path.parent.mkdir(parents=True, exist_ok=True)

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
        self._frames.append(self._frame(best.individual))

    def finalize(self, individual: TriangleIndividual) -> None:
        """Cierra la animación con el mejor individuo de la corrida y escribe el archivo."""
        frames = [*self._frames, self._frame(individual)]
        first, *rest = frames
        durations = [self._config.frame_duration_ms] * len(self._frames)
        durations.append(self._config.final_frame_duration_ms)
        options: dict[str, object] = {}
        if self._config.loop:
            options["loop"] = 0  # 0 significa repetición infinita en el formato GIF.
        first.save(
            self._config.path,
            save_all=True,
            append_images=rest,
            duration=durations,
            disposal=1,
            **options,
        )
        print(f"GIF guardado en: {self._config.path} ({len(frames)} cuadro(s))")

    def _frame(self, individual: TriangleIndividual) -> Image.Image:
        image = render_individual(individual, self._target, self._config.full_resolution)
        return image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)


class ProgressReporter(EvolutionObserver[TriangleIndividual, MSEFitness]):
    """Imprime el mejor fitness y los tiempos cada `report_every` generaciones."""

    def __init__(self, report_every: int) -> None:
        self._report_every = report_every
        self._started_at = perf_counter()
        self._last_report_at = self._started_at
        self._started_cpu_at = process_time()

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self._started_at

    @property
    def cpu_seconds(self) -> float:
        """Tiempo de CPU consumido por esta corrida."""
        return process_time() - self._started_cpu_at

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


class RunArtifactWriter(EvolutionObserver[TriangleIndividual, MSEFitness]):
    """Escribe historial por generación y los artefactos serializables de una corrida."""

    def __init__(self, directory: Path, config: object) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._history = self.directory / "history.csv"
        self._history.write_text("generation,best_fitness,population_size\n", encoding="utf-8")
        self._write_json("config.json", _jsonable(config))

    def on_generation(self, state: EvolutionState[TriangleIndividual, MSEFitness], context: EvolutionContext) -> None:
        best = next(iter(state.population), None)
        if best is None:
            return
        with self._history.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow((state.generation, best.fitness.value, len(state.population)))

    def finalize(self, individual: TriangleIndividual, fitness: float, reason: str,
                 metadata: dict[str, object] | None = None) -> None:
        self._write_json("triangles.json", {"triangles": [_triangle_json(gene) for gene in individual.genome]})
        summary = {
            "best_fitness": fitness,
            "termination_reason": reason,
            "history": "history.csv",
            "triangles": "triangles.json",
        }
        if metadata:
            summary.update(metadata)
        self._write_json("summary.json", summary)

    def _write_json(self, name: str, value: object) -> None:
        (self.directory / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _triangle_json(gene: TriangleGene) -> dict[str, object]:
    return {"center_x": gene.center_x, "center_y": gene.center_y, "size": gene.size,
            "angle_a": gene.angle_a, "angle_b": gene.angle_b, "angle_c": gene.angle_c,
            "rotation": gene.rotation, "r": gene.r, "g": gene.g, "b": gene.b,
            "alpha": gene.alpha, "vertices": [list(point) for point in gene.vertices]}


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
