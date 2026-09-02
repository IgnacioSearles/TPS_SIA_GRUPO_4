"""Punto de entrada reservado para una implementación concreta del TP."""

import argparse
from pathlib import Path
from time import perf_counter

from PIL import Image

from genetic_algorithm.application import (
    AdditiveSurvival,
    AnnularCrossover,
    CompositeEvolutionObserver,
    EliteSelection,
    EvolutionObserver,
    ExclusiveSurvival,
    MaxGenerationsTermination,
    MultiGeneMutation,
    OnePointCrossover,
    OrchestratedGeneticAlgorithm,
    RandomCutPointSelector,
    RandomPairingStrategy,
    RandomRingCutPointSelector,
    RandomTwoCutPointSelector,
    TwoPointCrossover,
    UniformCrossover,
)
from genetic_algorithm.domain import EvolutionContext, EvolutionState
from triangle_image import (
    BlurredMSEEvaluator,
    ChamferEdgeEvaluator,
    ColorHistogramEvaluator,
    CompositeEvaluator,
    EdgeMSEEvaluator,
    GradientOrientationEvaluator,
    MSEComparator,
    MSEEvaluator,
    MSEFitness,
    AdaptiveReheatMutationSchedule,
    ConstantMutationSchedule,
    ExponentialMutationSchedule,
    LinearMutationSchedule,
    MutationParameters,
    MutationScheduleObserver,
    MultiScaleMSEEvaluator,
    NormalizedEvaluator,
    RandomTriangleInitializer,
    RegionalMSEEvaluator,
    SCALES,
    SSIMEvaluator,
    SaliencyMSEEvaluator,
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

FITNESS_NAMES = [
    "mse", "regional", "ssim", "blur", "multiscale", "histogram", "edge",
    "gradient", "chamfer", "saliency",
]


def build_evaluator(name: str, args: argparse.Namespace):
    """Construye el evaluador crudo (sin normalizar) correspondiente a `name`."""
    if name == "regional":
        return RegionalMSEEvaluator(args.grid_rows, args.grid_cols, args.detail_weight)
    if name == "ssim":
        return SSIMEvaluator(args.ssim_win_size, args.ssim_mse_weight)
    if name == "blur":
        return BlurredMSEEvaluator(args.blur_sigma)
    if name == "multiscale":
        scales = tuple(float(s) for s in args.ms_scales.split(","))
        return MultiScaleMSEEvaluator(scales)
    if name == "histogram":
        return ColorHistogramEvaluator(args.hist_bins, args.hist_mse_weight)
    if name == "edge":
        return EdgeMSEEvaluator(args.edge_sigma)
    if name == "gradient":
        return GradientOrientationEvaluator(args.edge_sigma, args.gradient_orientation_weight)
    if name == "chamfer":
        return ChamferEdgeEvaluator(args.edge_sigma, args.chamfer_threshold)
    if name == "saliency":
        return SaliencyMSEEvaluator(args.saliency_weight, args.saliency_sigma)
    if name == "mse":
        return MSEEvaluator()
    raise ValueError(f"Fitness desconocido: {name}")


def build_combo_evaluator(spec: str, args: argparse.Namespace) -> CompositeEvaluator:
    """Parsea una combinación tipo 'mse:0.4,ssim:0.3,histogram:0.3' en un CompositeEvaluator.

    Cada componente se normaliza a un rango comparable (~[0, 1]) antes de ponderarse,
    usando el error máximo teórico de `SCALES`.
    """
    components = []
    for chunk in spec.split(","):
        name, _, weight_str = chunk.strip().partition(":")
        name = name.strip()
        if name not in SCALES:
            raise ValueError(f"Fitness desconocido en combo: '{name}'. Opciones: {sorted(SCALES)}")
        try:
            weight = float(weight_str) if weight_str else 1.0
        except ValueError as error:
            raise ValueError(f"Peso inválido para '{name}': '{weight_str}'") from error
        raw_evaluator = build_evaluator(name, args)
        components.append((NormalizedEvaluator(raw_evaluator, SCALES[name]), weight))
    return CompositeEvaluator(components)


def build_crossover(name: str, codec: TriangleCodec, args: argparse.Namespace):
    """Construye la estrategia de cruza elegida para genomas de triángulos."""
    if name == "one-point":
        return OnePointCrossover(codec, RandomCutPointSelector())
    if name == "two-point":
        return TwoPointCrossover(codec, RandomTwoCutPointSelector())
    if name == "uniform":
        return UniformCrossover(codec, args.uniform_swap_prob)
    if name == "annular":
        return AnnularCrossover(codec, RandomRingCutPointSelector())
    raise ValueError(f"Cruza desconocida: {name}")


def build_mutation_schedule(args: argparse.Namespace):
    """Construye una política explícita; constant conserva el comportamiento base."""
    initial = MutationParameters(
        args.mutation_prob, args.mutation_strength, args.replacement_prob
    )
    if args.mutation_schedule == "constant":
        return ConstantMutationSchedule(initial)

    if args.mutation_decay_generations <= 0:
        raise ValueError("las políticas de decaimiento requieren --mutation-decay-generations positivo")
    final = MutationParameters(
        args.mutation_prob if args.mutation_prob_final is None else args.mutation_prob_final,
        args.mutation_strength if args.mutation_strength_final is None else args.mutation_strength_final,
        args.replacement_prob if args.replacement_prob_final is None else args.replacement_prob_final,
    )
    if args.mutation_schedule == "linear":
        return LinearMutationSchedule(initial, final, args.mutation_decay_generations)
    exponential = ExponentialMutationSchedule(initial, final, args.mutation_decay_generations)
    if args.mutation_schedule == "exponential":
        return exponential
    return AdaptiveReheatMutationSchedule(
        exponential,
        args.mutation_stagnation_generations,
        args.mutation_reheat_generations,
    )


class EvolutionProgressReporter(EvolutionObserver[TriangleIndividual, MSEFitness]):
    """Informa progreso y, opcionalmente, guarda previews del mejor individuo."""

    def __init__(
        self,
        target: TriangleImageTarget,
        progress_every: int,
        preview_dir: Path | None,
        preview_every: int,
        preview_full_resolution: bool,
    ) -> None:
        self._target = target
        self._progress_every = progress_every
        self._preview_dir = preview_dir
        self._preview_every = preview_every
        self._preview_full_resolution = preview_full_resolution
        self._started_at = perf_counter()
        self._last_report_at = self._started_at
        if preview_dir is not None:
            preview_dir.mkdir(parents=True, exist_ok=True)

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self._started_at

    def on_generation(
        self,
        state: EvolutionState[TriangleIndividual, MSEFitness],
        context: EvolutionContext,
    ) -> None:
        now = perf_counter()
        population = iter(state.population)
        best = next(population, None)
        if best is None:
            return

        if self._preview_dir is not None and state.generation % self._preview_every == 0:
            individual = best.individual
            width, height = self._target.width, self._target.height
            if self._preview_full_resolution and self._target.scale_factor != 1.0:
                individual = individual.scale(1.0 / self._target.scale_factor)
                width, height = self._target.orig_width, self._target.orig_height
            image = render(individual, width, height)
            image.save(self._preview_dir / f"generation_{state.generation:05d}.png")

        if self._progress_every and state.generation % self._progress_every == 0:
            elapsed = now - self._started_at
            since_last = now - self._last_report_at
            print(
                f"Generación {state.generation:>5} | mejor fitness: {best.fitness.value:.6f} "
                f"| transcurrido: {elapsed:.1f}s | últimas {self._progress_every}: {since_last:.1f}s"
            )
            self._last_report_at = now


def main() -> None:
    """Configura y ejecuta el motor del algoritmo genético para aproximar una imagen."""
    parser = argparse.ArgumentParser(description="Aproximación de imágenes con triángulos.")
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen objetivo.")
    parser.add_argument("--output", type=str, default="output.png", help="Ruta de guardado.")
    parser.add_argument("--triangles", type=int, default=50, help="Triángulos por individuo.")
    parser.add_argument("--pop-size", type=int, default=100, help="Tamaño de la población.")
    parser.add_argument("--parents", type=int, default=50, help="Cantidad de padres a seleccionar.")
    parser.add_argument(
        "--survival", choices=["additive", "exclusive"], default="additive",
        help="Estrategia de supervivencia para construir cada nueva generación.",
    )
    parser.add_argument(
        "--crossover", choices=["one-point", "two-point", "uniform", "annular"],
        default="one-point", help="Estrategia de cruza aplicada a cada pareja de padres.",
    )
    parser.add_argument(
        "--uniform-swap-prob", type=float, default=0.5,
        help="Probabilidad de intercambiar cada gen en cruza uniforme.",
    )
    parser.add_argument("--generations", type=int, default=1000, help="Máximo de generaciones.")
    parser.add_argument("--mutation-prob", type=float, default=0.1, help="Prob. de mutar cada gen.")
    parser.add_argument(
        "--mutation-schedule", choices=["constant", "linear", "exponential", "adaptive-reheat"],
        default="constant", help="Política de mutación; constant mantiene el comportamiento base.",
    )
    parser.add_argument(
        "--mutation-prob-final", type=float, default=None,
        help="Probabilidad de mutación al finalizar el decaimiento; por defecto no cambia.",
    )
    parser.add_argument("--mutation-strength", type=float, default=0.1, help="Fuerza de la mutación.")
    parser.add_argument(
        "--mutation-strength-final", type=float, default=None,
        help="Fuerza de mutación al finalizar el decaimiento; por defecto no cambia.",
    )
    parser.add_argument(
        "--replacement-prob", type=float, default=0.02,
        help="Probabilidad de reemplazar por completo un gen seleccionado para mutar.",
    )
    parser.add_argument(
        "--replacement-prob-final", type=float, default=None,
        help="Probabilidad de reemplazo al finalizar el decaimiento; por defecto no cambia.",
    )
    parser.add_argument(
        "--mutation-decay-generations", type=int, default=0,
        help="Duración lineal o constante de tiempo exponencial; no aplica a constant.",
    )
    parser.add_argument(
        "--mutation-stagnation-generations", type=int, default=100,
        help="Generaciones sin mejora antes de recalentar adaptive-reheat.",
    )
    parser.add_argument(
        "--mutation-reheat-generations", type=int, default=40,
        help="Duración del pulso agresivo de adaptive-reheat.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Semilla para reproducir la ejecución.")
    parser.add_argument(
        "--progress-every", type=int, default=10,
        help="Informa progreso cada N generaciones; 0 lo desactiva.",
    )
    parser.add_argument(
        "--preview-dir", type=Path, default=None,
        help="Directorio opcional donde guardar previews del mejor candidato.",
    )
    parser.add_argument(
        "--preview-every", type=int, default=1,
        help="Guarda un preview cada N generaciones cuando se usa --preview-dir.",
    )
    parser.add_argument(
        "--preview-full-resolution", action="store_true",
        help="Guarda previews a resolución final; por defecto usa la resolución de trabajo.",
    )
    parser.add_argument("--max-size", type=int, default=128, help="Resolución máxima de trabajo para acelerar el cálculo.")
    parser.add_argument(
        "--fitness", type=str, choices=["global", *FITNESS_NAMES, "combo"], default="global",
        help="Tipo de fitness. 'global' = MSE clásico (alias de 'mse'). 'combo' combina varios "
             "normalizados, ver --combo.",
    )
    parser.add_argument(
        "--combo", type=str, default=None,
        help="Combinación de fitness normalizados y ponderados, solo con --fitness combo. "
             "Formato 'nombre:peso,nombre:peso,...', ej. 'mse:0.4,ssim:0.3,histogram:0.3'. "
             f"Nombres válidos: {', '.join(FITNESS_NAMES)}.",
    )
    parser.add_argument("--grid-rows", type=int, default=8, help="Filas de la grilla (solo --fitness regional).")
    parser.add_argument("--grid-cols", type=int, default=8, help="Columnas de la grilla (solo --fitness regional).")
    parser.add_argument(
        "--detail-weight", type=float, default=1.0,
        help="Cuánto pesan las regiones de mayor detalle/contraste (solo --fitness regional). 0 = todas las regiones pesan igual.",
    )
    parser.add_argument(
        "--ssim-win-size", type=int, default=7,
        help="Tamaño (impar) de la ventana deslizante para SSIM (solo --fitness ssim).",
    )
    parser.add_argument(
        "--ssim-mse-weight", type=float, default=0.5,
        help="Peso del término RMSE mezclado con SSIM, 0.0-1.0 (solo --fitness ssim). "
             "Evita que el algoritmo colapse a un color plano explotando puntos débiles de SSIM puro.",
    )
    parser.add_argument(
        "--blur-sigma", type=float, default=1.5,
        help="Desvío estándar del desenfoque gaussiano aplicado antes de comparar (solo --fitness blur).",
    )
    parser.add_argument(
        "--ms-scales", type=str, default="1.0,0.5,0.25",
        help="Escalas separadas por coma para el MSE multi-escala (solo --fitness multiscale).",
    )
    parser.add_argument(
        "--hist-bins", type=int, default=32,
        help="Cantidad de bins por canal para el histograma de color (solo --fitness histogram).",
    )
    parser.add_argument(
        "--hist-mse-weight", type=float, default=0.5,
        help="Peso del término RMSE mezclado con el histograma, 0.0-1.0 (solo --fitness histogram). "
             "El histograma es ciego a la posición: sin este término el algoritmo puede reproducir "
             "el reparto de colores correcto sin parecerse estructuralmente al objetivo.",
    )
    parser.add_argument(
        "--edge-sigma", type=float, default=1.0,
        help="Desvío estándar del suavizado previo a Sobel (solo --fitness edge). 0 desactiva el suavizado.",
    )
    parser.add_argument(
        "--gradient-orientation-weight", type=float, default=0.5,
        help="Peso de orientación frente a magnitud, 0.0-1.0 (solo --fitness gradient).",
    )
    parser.add_argument(
        "--chamfer-threshold", type=float, default=20.0,
        help="Umbral 0-255 para binarizar bordes Sobel (solo --fitness chamfer).",
    )
    parser.add_argument(
        "--saliency-weight", type=float, default=3.0,
        help="Refuerzo para zonas salientes del objetivo (solo --fitness saliency).",
    )
    parser.add_argument(
        "--saliency-sigma", type=float, default=2.0,
        help="Suavizado de la máscara de saliencia (solo --fitness saliency).",
    )
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every debe ser no negativo")
    if args.preview_every <= 0:
        parser.error("--preview-every debe ser positivo")
    if args.mutation_decay_generations < 0:
        parser.error("--mutation-decay-generations debe ser no negativo")
    offspring_count = 2 * (args.parents // 2)
    if args.survival == "exclusive" and offspring_count < args.pop_size:
        parser.error(
            "--survival exclusive requiere suficientes hijos; usá una cantidad par de "
            "--parents al menos igual a --pop-size"
        )

    # Cargar objetivo
    print(f"Cargando imagen: {args.image}")
    original = Image.open(args.image)
    target = TriangleImageTarget(original, max_size=args.max_size)

    if target.scale_factor != 1.0:
        print(f"Redimensionando a {target.width}x{target.height} para evaluación...")

    # Configuración de problema
    comparator = MSEComparator()
    if args.fitness == "combo":
        if not args.combo:
            parser.error("--fitness combo requiere --combo 'nombre:peso,...'")
        evaluator = build_combo_evaluator(args.combo, args)
    elif args.fitness == "global":
        evaluator = build_evaluator("mse", args)
    else:
        evaluator = build_evaluator(args.fitness, args)
    problem = TriangleProblem(target, evaluator, comparator)
    config = TriangleConfiguration(args.pop_size, args.parents)
    context = TriangleContext(args.seed)
    reporter = EvolutionProgressReporter(
        target, args.progress_every, args.preview_dir, args.preview_every,
        args.preview_full_resolution,
    )
    codec = TriangleCodec()

    # Operadores
    initializer = RandomTriangleInitializer(args.triangles, target.width, target.height)
    selection = EliteSelection(comparator)
    pairing = RandomPairingStrategy()
    crossover = build_crossover(args.crossover, codec, args)
    
    try:
        mutation_schedule = build_mutation_schedule(args)
    except ValueError as error:
        parser.error(str(error))
    gene_mutator = ScheduledTriangleGeneMutator(target.width, target.height, mutation_schedule)
    mutation = MultiGeneMutation(
        codec,
        ScheduledGenePositionSelector(mutation_schedule),
        gene_mutator,
    )
    
    survival = (
        AdditiveSurvival(comparator)
        if args.survival == "additive"
        else ExclusiveSurvival(comparator)
    )
    termination = MaxGenerationsTermination(args.generations)

    # Orquestador
    engine = OrchestratedGeneticAlgorithm(
        initializer=initializer,
        selection=selection,
        pairing=pairing,
        crossover=crossover,
        mutation=mutation,
        survival=survival,
        termination=termination,
        context=context,
        observer=CompositeEvolutionObserver((reporter, MutationScheduleObserver(mutation_schedule))),
    )

    print("Iniciando evolución...")
    result = engine.run(problem, config)
    
    best = result.final_state.population[0]
    print(
        f"Evolución terminada. Generación: {result.final_state.generation}, "
        f"Mejor fitness ({args.fitness}): {best.fitness.value:.4f}, "
        f"Tiempo total: {reporter.elapsed_seconds:.1f}s"
    )

    # Guardar el mejor
    best_ind = best.individual
    if target.scale_factor != 1.0:
        print(f"Escalando triángulos de vuelta a {target.orig_width}x{target.orig_height}...")
        best_ind = best_ind.scale(1.0 / target.scale_factor)

    final_image = render(best_ind, target.orig_width, target.orig_height)
    final_image.save(args.output)
    print(f"Imagen guardada en: {args.output}")


if __name__ == "__main__":
    main()
