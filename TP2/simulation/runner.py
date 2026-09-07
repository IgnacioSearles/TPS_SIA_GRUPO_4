"""Ejecuta una simulación completa a partir de su configuración."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections.abc import Collection
from dataclasses import dataclass, replace
from functools import cmp_to_key
from pathlib import Path
from random import SystemRandom
from time import process_time

from PIL import Image

from genetic_algorithm.application import (
    CompositeEvolutionObserver,
    DefaultEvolutionState,
    DefaultScoredIndividual,
    OrchestratedGeneticAlgorithm,
    RandomPairingStrategy,
)
from genetic_algorithm.domain import EvolutionState, ScoredIndividual
from triangle_image import (
    MSEComparator,
    MSEFitness,
    MutationScheduleObserver,
    RandomTriangleInitializer,
    SpatialErrorGuidanceObserver,
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
    build_mutation,
    build_termination,
    build_selection,
    build_survival,
)
from simulation.config import SimulationConfig
from simulation.reporting import (
    GifWriter,
    PreviewWriter,
    ProgressReporter,
    RunArtifactWriter,
)
from simulation.optimizations import (
    FitnessAdaptiveMutation,
    IslandRuntime,
    LocalSearchMutation,
    SeededTriangleInitializer,
    scale_between_targets,
)

_SEED_UPPER_BOUND = 2 ** 32


@dataclass(frozen=True, slots=True)
class SimulationOutcome:
    """Resumen de una corrida terminada."""

    seed: int
    generations: int
    best_fitness: float
    elapsed_seconds: float
    cpu_seconds: float
    output_path: Path
    run_directory: Path = Path("run")
    termination_reason: str = "max-generations"


def run_simulation(config: SimulationConfig) -> SimulationOutcome:
    """Corre el algoritmo genético descrito por `config` y guarda la mejor imagen."""
    seed = _resolve_seed(config.seed)
    if config.optimization.strategy == "progressive-resolution":
        return _run_progressive_resolution(config, seed)
    if config.optimization.strategy == "islands":
        return _run_islands(config, seed)
    outcome, _, _ = _run_single(config, seed)
    return outcome


def _run_single(
    config: SimulationConfig,
    seed: int,
    initial_individual: TriangleIndividual | None = None,
    target: TriangleImageTarget | None = None,
    run_directory: Path | None = None,
    save_final: bool = True,
) -> tuple[SimulationOutcome, TriangleIndividual, TriangleImageTarget]:
    target = target or _load_target(config)

    comparator = MSEComparator()
    evaluator = build_fitness_evaluator(config.fitness)
    problem = TriangleProblem(target, evaluator, comparator)
    mutation_schedule = build_mutation_schedule(config.mutation)
    progress = ProgressReporter(config.progress_every)
    codec = TriangleCodec()
    run_directory = run_directory or config.output.parent / "run"
    artifacts = RunArtifactWriter(run_directory, config)
    gif = _gif_writer(config, target)
    initializer = (
        RandomTriangleInitializer(config.triangles, target.width, target.height)
        if initial_individual is None
        else SeededTriangleInitializer(
            (initial_individual,), config.triangles, target.width, target.height
        )
    )
    mutation = build_mutation(
        config.mutation,
        target.width,
        target.height,
        codec,
        mutation_schedule,
        config.triangles,
    )
    if config.optimization.strategy == "fitness-adaptive-mutation":
        mutation = FitnessAdaptiveMutation(
            config.mutation,
            target.width,
            target.height,
            codec,
            mutation_schedule,
            target,
            evaluator,
            config.optimization.fitness_adaptive_mutation,
            config.triangles,
        )
    elif config.optimization.strategy == "local-search":
        mutation = LocalSearchMutation(
            mutation,
            target.width,
            target.height,
            codec,
            mutation_schedule,
            target,
            evaluator,
            comparator,
            config.optimization.local_search,
        )

    context = TriangleContext(seed)
    spatial_guidance = (
        (SpatialErrorGuidanceObserver(target),)
        if config.mutation.strategy == "spatial-guided" else ()
    )
    engine = OrchestratedGeneticAlgorithm(
        initializer=initializer,
        selection=build_selection(config.selection, comparator),
        pairing=RandomPairingStrategy(),
        crossover=build_crossover(
            config.crossover,
            codec,
            config.mutation.strategy == "spatial-guided",
        ),
        mutation=mutation,
        survival=build_survival(config.population, comparator),
        termination=build_termination(
            config.termination, config.population.generations, comparator
        ),
        context=context,
        observer=CompositeEvolutionObserver(
            (
                progress,
                artifacts,
                MutationScheduleObserver(mutation_schedule),
                *spatial_guidance,
                *_preview_observers(config, target),
                *(() if gif is None else (gif,)),
            )
        ),
    )

    print("Iniciando evolución...")
    result = engine.run(
        problem, TriangleConfiguration(config.population.size, config.population.parents)
    )

    best = result.final_state.population[0]
    outcome = SimulationOutcome(
        seed=seed,
        generations=result.final_state.generation,
        best_fitness=best.fitness.value,
        elapsed_seconds=progress.elapsed_seconds,
        cpu_seconds=progress.cpu_seconds,
        output_path=config.output,
        run_directory=run_directory,
        termination_reason=_termination_reason(config, result.final_state.generation),
    )
    print(
        f"Evolución terminada. Generación: {outcome.generations}, "
        f"mejor fitness ({config.fitness.metric}): {outcome.best_fitness:.4f}, "
        f"tiempo pared: {outcome.elapsed_seconds:.1f}s, CPU: {outcome.cpu_seconds:.1f}s"
    )
    if gif is not None:
        gif.finalize(best.individual)
    if save_final:
        _save_best(best.individual, target, config.output)
        _save_best(best.individual, target, run_directory / "best.png")
    artifacts.finalize(
        best.individual,
        best.fitness.value,
        outcome.termination_reason,
        {
            "seed": outcome.seed,
            "generations": outcome.generations,
            "elapsed_seconds": outcome.elapsed_seconds,
            "cpu_seconds": outcome.cpu_seconds,
            "triangle_count": len(best.individual.genome),
        },
    )
    return outcome, best.individual, target


def _run_progressive_resolution(config: SimulationConfig, seed: int) -> SimulationOutcome:
    assert config.optimization.progressive_resolution is not None
    best_individual: TriangleIndividual | None = None
    previous_target: TriangleImageTarget | None = None
    final_outcome: SimulationOutcome | None = None
    elapsed_seconds = 0.0
    cpu_seconds = 0.0
    total_generations = 0
    stages = config.optimization.progressive_resolution.stages
    root_run_directory = config.output.parent / "run"

    for index, stage in enumerate(stages, start=1):
        print(
            f"Etapa progresiva {index}/{len(stages)}: "
            f"max_size={stage.max_size}, generaciones={stage.generations}"
        )
        stage_config = _config_for_progressive_stage(config, index, stage.max_size, stage.generations)
        target = _load_target(stage_config)
        seed_individual = (
            None if best_individual is None or previous_target is None
            else scale_between_targets(best_individual, previous_target, target)
        )
        is_last = index == len(stages)
        run_directory = root_run_directory if is_last else root_run_directory / f"stage_{index:02d}"
        outcome, best_individual, previous_target = _run_single(
            stage_config,
            seed + index - 1,
            initial_individual=seed_individual,
            target=target,
            run_directory=run_directory,
            save_final=is_last,
        )
        final_outcome = outcome
        elapsed_seconds += outcome.elapsed_seconds
        cpu_seconds += outcome.cpu_seconds
        total_generations += outcome.generations

    assert final_outcome is not None
    return replace(
        final_outcome,
        generations=total_generations,
        elapsed_seconds=elapsed_seconds,
        cpu_seconds=cpu_seconds,
        run_directory=root_run_directory,
    )


def _config_for_progressive_stage(
    config: SimulationConfig, index: int, max_size: int, generations: int
) -> SimulationConfig:
    preview = None
    if config.preview is not None:
        preview = replace(
            config.preview,
            directory=config.preview.directory / f"stage_{index:02d}",
        )
    gif = None
    if config.gif is not None:
        gif = replace(
            config.gif,
            path=config.gif.path.with_name(
                f"{config.gif.path.stem}_stage_{index:02d}{config.gif.path.suffix}"
            ),
        )
    return replace(
        config,
        max_size=max_size,
        population=replace(config.population, generations=generations),
        preview=preview,
        gif=gif,
    )


def _run_islands(config: SimulationConfig, seed: int) -> SimulationOutcome:
    if config.optimization.islands.parallel:
        return _run_parallel_islands(config, seed)

    target = _load_target(config)
    comparator = MSEComparator()
    evaluator = build_fitness_evaluator(config.fitness)
    problem = TriangleProblem(target, evaluator, comparator)
    progress = ProgressReporter(config.progress_every)
    run_directory = config.output.parent / "run"
    artifacts = RunArtifactWriter(run_directory, config)
    gif = _gif_writer(config, target)
    observer = CompositeEvolutionObserver(
        (
            progress,
            artifacts,
            *_preview_observers(config, target),
            *(() if gif is None else (gif,)),
        )
    )
    termination = build_termination(
        config.termination, config.population.generations, comparator
    )
    islands = tuple(
        _create_island(config, seed + index, target, problem, comparator)
        for index in range(config.optimization.islands.count)
    )
    global_context = TriangleContext(seed)
    generation = 0
    state = _global_state(generation, islands, comparator)
    observer.on_generation(state, global_context)

    print("Iniciando evolución por islas...")
    while not termination.should_stop(state, global_context):
        generation += 1
        for island in islands:
            island.context.set_generation(generation - 1)
            _update_spatial_guidance(config, target, island.state, island.context)
            island.state = _evolve_island(config, island, problem, comparator)
            island.schedule.observe_best(
                island.state.generation, island.state.population[0].fitness.value
            )
        if generation % config.optimization.islands.migration_every == 0:
            _migrate_ring(
                islands,
                config.optimization.islands.migration_count,
                generation,
                comparator,
            )
        global_context.set_generation(generation)
        state = _global_state(generation, islands, comparator)
        observer.on_generation(state, global_context)

    best = state.population[0]
    outcome = SimulationOutcome(
        seed=seed,
        generations=generation,
        best_fitness=best.fitness.value,
        elapsed_seconds=progress.elapsed_seconds,
        cpu_seconds=progress.cpu_seconds,
        output_path=config.output,
        run_directory=run_directory,
        termination_reason=_termination_reason(config, generation),
    )
    print(
        f"Evolución por islas terminada. Generación: {outcome.generations}, "
        f"mejor fitness ({config.fitness.metric}): {outcome.best_fitness:.4f}, "
        f"tiempo pared: {outcome.elapsed_seconds:.1f}s, CPU: {outcome.cpu_seconds:.1f}s"
    )
    if gif is not None:
        gif.finalize(best.individual)
    _save_best(best.individual, target, config.output)
    _save_best(best.individual, target, run_directory / "best.png")
    artifacts.finalize(
        best.individual,
        best.fitness.value,
        outcome.termination_reason,
        {
            "seed": outcome.seed,
            "generations": outcome.generations,
            "elapsed_seconds": outcome.elapsed_seconds,
            "cpu_seconds": outcome.cpu_seconds,
            "islands": config.optimization.islands.count,
            "triangle_count": len(best.individual.genome),
        },
    )
    return outcome


def _run_parallel_islands(config: SimulationConfig, seed: int) -> SimulationOutcome:
    target = _load_target(config)
    comparator = MSEComparator()
    evaluator = build_fitness_evaluator(config.fitness)
    problem = TriangleProblem(target, evaluator, comparator)
    progress = ProgressReporter(config.progress_every)
    run_directory = config.output.parent / "run"
    artifacts = RunArtifactWriter(run_directory, config)
    gif = _gif_writer(config, target)
    observer = CompositeEvolutionObserver(
        (
            progress,
            artifacts,
            *_preview_observers(config, target),
            *(() if gif is None else (gif,)),
        )
    )
    termination = build_termination(
        config.termination, config.population.generations, comparator
    )
    islands = tuple(
        _create_island(config, seed + index, target, problem, comparator)
        for index in range(config.optimization.islands.count)
    )
    global_context = TriangleContext(seed)
    generation = 0
    worker_cpu_seconds = 0.0
    state = _global_state(generation, islands, comparator)
    observer.on_generation(state, global_context)

    workers = config.optimization.islands.workers or config.optimization.islands.count
    workers = min(workers, config.optimization.islands.count)
    print(f"Iniciando evolución por islas en paralelo ({workers} worker(s))...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while not termination.should_stop(state, global_context):
            steps = _parallel_island_steps(config, generation)
            results = tuple(
                executor.map(
                    _evolve_island_block,
                    ((config, island, steps) for island in islands),
                )
            )
            islands = tuple(result.island for result in results)
            worker_cpu_seconds += sum(result.cpu_seconds for result in results)
            generation = islands[0].state.generation

            if generation % config.optimization.islands.migration_every == 0:
                _migrate_ring(
                    islands,
                    config.optimization.islands.migration_count,
                    generation,
                    comparator,
                )
            global_context.set_generation(generation)
            state = _global_state(generation, islands, comparator)
            observer.on_generation(state, global_context)

    best = state.population[0]
    outcome = SimulationOutcome(
        seed=seed,
        generations=generation,
        best_fitness=best.fitness.value,
        elapsed_seconds=progress.elapsed_seconds,
        cpu_seconds=progress.cpu_seconds + worker_cpu_seconds,
        output_path=config.output,
        run_directory=run_directory,
        termination_reason=_termination_reason(config, generation),
    )
    print(
        f"Evolución por islas terminada. Generación: {outcome.generations}, "
        f"mejor fitness ({config.fitness.metric}): {outcome.best_fitness:.4f}, "
        f"tiempo pared: {outcome.elapsed_seconds:.1f}s, CPU: {outcome.cpu_seconds:.1f}s"
    )
    if gif is not None:
        gif.finalize(best.individual)
    _save_best(best.individual, target, config.output)
    _save_best(best.individual, target, run_directory / "best.png")
    artifacts.finalize(
        best.individual,
        best.fitness.value,
        outcome.termination_reason,
        {
            "seed": outcome.seed,
            "generations": outcome.generations,
            "elapsed_seconds": outcome.elapsed_seconds,
            "cpu_seconds": outcome.cpu_seconds,
            "islands": config.optimization.islands.count,
            "parallel": True,
            "workers": workers,
            "triangle_count": len(best.individual.genome),
        },
    )
    return outcome


@dataclass(frozen=True, slots=True)
class _IslandBlockResult:
    island: IslandRuntime
    cpu_seconds: float


def _evolve_island_block(arguments: tuple[SimulationConfig, IslandRuntime, int]) -> _IslandBlockResult:
    config, island, steps = arguments
    started_cpu = process_time()
    target = _load_target_quiet(config)
    comparator = MSEComparator()
    evaluator = build_fitness_evaluator(config.fitness)
    problem = TriangleProblem(target, evaluator, comparator)
    for _ in range(steps):
        island.context.end_render_scope()
        island.context.set_generation(island.state.generation)
        _update_spatial_guidance(config, target, island.state, island.context)
        island.state = _evolve_island(config, island, problem, comparator)
        island.schedule.observe_best(
            island.state.generation, island.state.population[0].fitness.value
        )
    island.context.end_render_scope()
    return _IslandBlockResult(island, process_time() - started_cpu)


def _parallel_island_steps(config: SimulationConfig, generation: int) -> int:
    if config.termination.strategy != "max-generations":
        return 1

    sync_every = config.optimization.islands.migration_every
    if config.progress_every:
        sync_every = min(sync_every, config.progress_every)
    remaining = config.population.generations - generation
    return max(1, min(sync_every, remaining))


def _create_island(
    config: SimulationConfig,
    seed: int,
    target: TriangleImageTarget,
    problem: TriangleProblem,
    comparator: MSEComparator,
) -> IslandRuntime:
    context = TriangleContext(seed)
    codec = TriangleCodec()
    schedule = build_mutation_schedule(config.mutation)
    initializer = RandomTriangleInitializer(config.triangles, target.width, target.height)
    initial_population = initializer.create_initial_population(config.population.size, context)
    state = DefaultEvolutionState(
        0, _order_by_fitness(_evaluate(problem, initial_population, context), comparator)
    )
    schedule.observe_best(0, state.population[0].fitness.value)
    _update_spatial_guidance(config, target, state, context)
    return IslandRuntime(
        state=state,
        context=context,
        selection=build_selection(config.selection, comparator),
        pairing=RandomPairingStrategy(),
        crossover=build_crossover(
            config.crossover,
            codec,
            config.mutation.strategy == "spatial-guided",
        ),
        mutation=build_mutation(
            config.mutation,
            target.width,
            target.height,
            codec,
            schedule,
            config.triangles,
        ),
        survival=build_survival(config.population, comparator),
        schedule=schedule,
    )


def _update_spatial_guidance(
    config: SimulationConfig,
    target: TriangleImageTarget,
    state: EvolutionState[TriangleIndividual, MSEFitness],
    context: TriangleContext,
) -> None:
    if config.mutation.strategy == "spatial-guided":
        SpatialErrorGuidanceObserver(target).on_generation(state, context)


def _evolve_island(
    config: SimulationConfig,
    island: IslandRuntime,
    problem: TriangleProblem,
    comparator: MSEComparator,
) -> EvolutionState[TriangleIndividual, MSEFitness]:
    selected = island.selection.select(
        island.state.population, config.population.parents, island.context
    )
    offspring = []
    for pair in island.pairing.pair(selected, island.context):
        crossed_individuals = island.crossover.cross(
            pair.first_parent, pair.second_parent, island.context
        )
        offspring.extend(
            island.mutation.mutate(individual, island.context)
            for individual in crossed_individuals
        )

    evaluated_offspring = _evaluate(problem, offspring, island.context)
    next_population = island.survival.build_next_generation(
        island.state.population,
        evaluated_offspring,
        config.population.size,
        island.context,
    )
    return DefaultEvolutionState(
        island.state.generation + 1,
        _order_by_fitness(next_population, comparator),
    )


def _evaluate(
    problem: TriangleProblem,
    individuals: Collection[TriangleIndividual],
    context: TriangleContext,
) -> tuple[ScoredIndividual[TriangleIndividual, MSEFitness], ...]:
    return tuple(
        DefaultScoredIndividual(
            individual,
            problem.fitness_evaluator.evaluate(individual, problem.target, context),
        )
        for individual in individuals
    )


def _order_by_fitness(
    population: Collection[ScoredIndividual[TriangleIndividual, MSEFitness]],
    comparator: MSEComparator,
) -> tuple[ScoredIndividual[TriangleIndividual, MSEFitness], ...]:
    def compare(
        left: ScoredIndividual[TriangleIndividual, MSEFitness],
        right: ScoredIndividual[TriangleIndividual, MSEFitness],
    ) -> int:
        if comparator.is_better(left.fitness, right.fitness):
            return -1
        if comparator.is_better(right.fitness, left.fitness):
            return 1
        return 0

    return tuple(sorted(population, key=cmp_to_key(compare)))


def _global_state(
    generation: int,
    islands: Collection[IslandRuntime],
    comparator: MSEComparator,
) -> EvolutionState[TriangleIndividual, MSEFitness]:
    population = tuple(
        candidate
        for island in islands
        for candidate in island.state.population
    )
    return DefaultEvolutionState(generation, _order_by_fitness(population, comparator))


def _migrate_ring(
    islands: tuple[IslandRuntime, ...],
    migration_count: int,
    generation: int,
    comparator: MSEComparator,
) -> None:
    migrants_by_source = tuple(
        tuple(island.state.population[:migration_count]) for island in islands
    )
    for source_index, migrants in enumerate(migrants_by_source):
        destination = islands[(source_index + 1) % len(islands)]
        retained = tuple(destination.state.population[:-migration_count])
        destination.state = DefaultEvolutionState(
            generation,
            _order_by_fitness((*retained, *migrants), comparator),
        )


def _termination_reason(config: SimulationConfig, generation: int) -> str:
    if config.termination.strategy == "max-generations":
        return "max-generations"
    if config.termination.strategy == "target-fitness":
        return "target-fitness"
    return "stagnation"


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


def _load_target_quiet(config: SimulationConfig) -> TriangleImageTarget:
    return TriangleImageTarget(Image.open(config.image), max_size=config.max_size)


def _preview_observers(
    config: SimulationConfig, target: TriangleImageTarget
) -> tuple[PreviewWriter, ...]:
    """Los previews son opcionales: sin sección `preview` no se engancha ningún observador."""
    if config.preview is None:
        return ()
    print(f"Guardando previews en: {config.preview.directory}")
    return (PreviewWriter(target, config.preview),)


def _gif_writer(config: SimulationConfig, target: TriangleImageTarget) -> GifWriter | None:
    """El GIF es opcional: sin sección `gif` no se acumula ningún cuadro en memoria."""
    if config.gif is None:
        return None
    print(f"Acumulando cuadros para el GIF: {config.gif.path}")
    return GifWriter(target, config.gif)


def _save_best(
    individual: TriangleIndividual, target: TriangleImageTarget, output: Path
) -> None:
    if target.scale_factor != 1.0:
        print(f"Escalando triángulos a {target.orig_width}x{target.orig_height}...")
        individual = individual.scale(1.0 / target.scale_factor)
    output.parent.mkdir(parents=True, exist_ok=True)
    render(individual, target.orig_width, target.orig_height).save(output)
    print(f"Imagen guardada en: {output}")
