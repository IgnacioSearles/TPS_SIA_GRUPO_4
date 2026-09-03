"""Ejecuta una simulación completa a partir de su configuración."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import SystemRandom

from PIL import Image

from genetic_algorithm.application import (
    CompositeEvolutionObserver,
    MaxGenerationsTermination,
    MultiGeneMutation,
    OrchestratedGeneticAlgorithm,
    RandomPairingStrategy,
)
from triangle_image import (
    MSEComparator,
    MutationScheduleObserver,
    RandomTriangleInitializer,
    ScheduledGenePositionSelector,
    ScheduledTriangleGeneMutator,
    TriangleCodec,
    TriangleConfiguration,
    TriangleContext,
    TriangleImageTarget,
    TriangleIndividual,
    TriangleProblem,
)
from triangle_image.rendering import render

from simulation.builders import (
    build_crossover,
    build_fitness_evaluator,
    build_mutation_schedule,
    build_selection,
    build_survival,
)
from simulation.config import SimulationConfig
from simulation.reporting import PreviewWriter, ProgressReporter

_SEED_UPPER_BOUND = 2 ** 32


@dataclass(frozen=True, slots=True)
class SimulationOutcome:
    """Resumen de una corrida terminada."""

    seed: int
    generations: int
    best_fitness: float
    elapsed_seconds: float
    output_path: Path


def run_simulation(config: SimulationConfig) -> SimulationOutcome:
    """Corre el algoritmo genético descrito por `config` y guarda la mejor imagen."""
    seed = _resolve_seed(config.seed)
    target = _load_target(config)

    comparator = MSEComparator()
    problem = TriangleProblem(target, build_fitness_evaluator(config.fitness), comparator)
    mutation_schedule = build_mutation_schedule(config.mutation)
    progress = ProgressReporter(config.progress_every)
    codec = TriangleCodec()

    engine = OrchestratedGeneticAlgorithm(
        initializer=RandomTriangleInitializer(
            config.triangles, target.width, target.height
        ),
        selection=build_selection(config.selection, comparator),
        pairing=RandomPairingStrategy(),
        crossover=build_crossover(config.crossover, codec),
        mutation=MultiGeneMutation(
            codec,
            ScheduledGenePositionSelector(mutation_schedule),
            ScheduledTriangleGeneMutator(target.width, target.height, mutation_schedule),
        ),
        survival=build_survival(config.population, comparator),
        termination=MaxGenerationsTermination(config.population.generations),
        context=TriangleContext(seed),
        observer=CompositeEvolutionObserver(
            (progress, MutationScheduleObserver(mutation_schedule), *_preview_observers(config, target))
        ),
    )

    print("Iniciando evolución...")
    result = engine.run(problem, TriangleConfiguration(config.population.size, config.population.parents))

    best = result.final_state.population[0]
    outcome = SimulationOutcome(
        seed=seed,
        generations=result.final_state.generation,
        best_fitness=best.fitness.value,
        elapsed_seconds=progress.elapsed_seconds,
        output_path=config.output,
    )
    print(
        f"Evolución terminada. Generación: {outcome.generations}, "
        f"mejor fitness ({config.fitness.metric}): {outcome.best_fitness:.4f}, "
        f"tiempo total: {outcome.elapsed_seconds:.1f}s"
    )
    _save_best(best.individual, target, config.output)
    return outcome


def _resolve_seed(seed: int | None) -> int:
    """Usa la semilla configurada o sortea una y la informa para poder repetir la corrida."""
    if seed is not None:
        return seed
    generated = SystemRandom().randrange(_SEED_UPPER_BOUND)
    print(f'Semilla generada: {generated} (agregá "seed": {generated} al config para repetirla)')
    return generated


def _load_target(config: SimulationConfig) -> TriangleImageTarget:
    print(f"Cargando imagen: {config.image}")
    target = TriangleImageTarget(Image.open(config.image), max_size=config.max_size)
    if target.scale_factor != 1.0:
        print(f"Redimensionando a {target.width}x{target.height} para evaluación...")
    return target


def _preview_observers(
    config: SimulationConfig, target: TriangleImageTarget
) -> tuple[PreviewWriter, ...]:
    """Los previews son opcionales: sin sección `preview` no se engancha ningún observador."""
    if config.preview is None:
        return ()
    print(f"Guardando previews en: {config.preview.directory}")
    return (PreviewWriter(target, config.preview),)


def _save_best(
    individual: TriangleIndividual, target: TriangleImageTarget, output: Path
) -> None:
    if target.scale_factor != 1.0:
        print(f"Escalando triángulos a {target.orig_width}x{target.orig_height}...")
        individual = individual.scale(1.0 / target.scale_factor)
    output.parent.mkdir(parents=True, exist_ok=True)
    render(individual, target.orig_width, target.orig_height).save(output)
    print(f"Imagen guardada en: {output}")
