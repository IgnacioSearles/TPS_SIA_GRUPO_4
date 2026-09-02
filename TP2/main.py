"""Punto de entrada reservado para una implementación concreta del TP."""

import argparse
from PIL import Image

from genetic_algorithm.application import (
    AdditiveSurvival,
    EliteSelection,
    MaxGenerationsTermination,
    MultiGeneMutation,
    OnePointCrossover,
    OrchestratedGeneticAlgorithm,
    RandomCutPointSelector,
    RandomGenePositionSelector,
    RandomPairingStrategy,
)
from triangle_image import (
    BlurredMSEEvaluator,
    ColorHistogramEvaluator,
    CompositeEvaluator,
    MSEComparator,
    MSEEvaluator,
    MultiScaleMSEEvaluator,
    NormalizedEvaluator,
    RandomTriangleInitializer,
    RegionalMSEEvaluator,
    SCALES,
    SSIMEvaluator,
    TriangleCodec,
    TriangleConfiguration,
    TriangleContext,
    TriangleGeneMutator,
    TriangleImageTarget,
    TriangleProblem,
)
from triangle_image.rendering import render

FITNESS_NAMES = ["mse", "regional", "ssim", "blur", "multiscale", "histogram"]


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
        weight = float(weight_str) if weight_str else 1.0
        raw_evaluator = build_evaluator(name, args)
        components.append((NormalizedEvaluator(raw_evaluator, SCALES[name]), weight))
    return CompositeEvaluator(components)


def main() -> None:
    """Configura y ejecuta el motor del algoritmo genético para aproximar una imagen."""
    parser = argparse.ArgumentParser(description="Aproximación de imágenes con triángulos.")
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen objetivo.")
    parser.add_argument("--output", type=str, default="output.png", help="Ruta de guardado.")
    parser.add_argument("--triangles", type=int, default=50, help="Triángulos por individuo.")
    parser.add_argument("--pop-size", type=int, default=100, help="Tamaño de la población.")
    parser.add_argument("--parents", type=int, default=50, help="Cantidad de padres a seleccionar.")
    parser.add_argument("--generations", type=int, default=1000, help="Máximo de generaciones.")
    parser.add_argument("--mutation-prob", type=float, default=0.1, help="Prob. de mutar cada gen.")
    parser.add_argument("--mutation-strength", type=float, default=0.1, help="Fuerza de la mutación.")
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
    args = parser.parse_args()

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
    context = TriangleContext()
    codec = TriangleCodec()

    # Operadores
    initializer = RandomTriangleInitializer(args.triangles, target.width, target.height)
    selection = EliteSelection(comparator)
    pairing = RandomPairingStrategy()
    crossover = OnePointCrossover(codec, RandomCutPointSelector())
    
    gene_mutator = TriangleGeneMutator(target.width, target.height, args.mutation_strength)
    mutation = MultiGeneMutation(codec, RandomGenePositionSelector(args.mutation_prob), gene_mutator)
    
    survival = AdditiveSurvival(comparator)
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
    )

    print("Iniciando evolución...")
    result = engine.run(problem, config)
    
    best = result.final_state.population[0]
    print(f"Evolución terminada. Generación: {result.final_state.generation}, Mejor fitness ({args.fitness}): {best.fitness.value:.4f}")

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
