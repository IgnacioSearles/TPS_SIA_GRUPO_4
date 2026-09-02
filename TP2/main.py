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
    MSEComparator,
    MSEEvaluator,
    RandomTriangleInitializer,
    TriangleCodec,
    TriangleConfiguration,
    TriangleContext,
    TriangleGeneMutator,
    TriangleImageTarget,
    TriangleProblem,
)
from triangle_image.rendering import render


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
    args = parser.parse_args()

    # Cargar objetivo
    print(f"Cargando imagen: {args.image}")
    original = Image.open(args.image)
    
    orig_width, orig_height = original.size
    scale_factor = 1.0
    if max(orig_width, orig_height) > args.max_size:
        scale_factor = args.max_size / float(max(orig_width, orig_height))
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        print(f"Redimensionando a {new_width}x{new_height} para evaluación...")
        # Image.Resampling.LANCZOS or Image.LANCZOS depending on PIL version, we can just use Image.LANCZOS if it exists, or just original.resize directly
        # For max compatibility:
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        work_image = original.resize((new_width, new_height), resample_filter)
    else:
        work_image = original

    target = TriangleImageTarget(work_image)

    # Configuración de problema
    comparator = MSEComparator()
    problem = TriangleProblem(target, MSEEvaluator(), comparator)
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
    print(f"Evolución terminada. Generación: {result.final_state.generation}, Mejor MSE: {best.fitness.value:.2f}")

    # Guardar el mejor
    best_ind = best.individual
    if scale_factor != 1.0:
        inv_scale = 1.0 / scale_factor
        from triangle_image import TriangleGene, TriangleIndividual
        scaled_genes = []
        for g in best_ind.genome:
            scaled_genes.append(TriangleGene(
                center_x=g.center_x * inv_scale,
                center_y=g.center_y * inv_scale,
                size=g.size * inv_scale,
                angle_a=g.angle_a,
                angle_b=g.angle_b,
                rotation=g.rotation,
                r=g.r,
                g=g.g,
                b=g.b,
                alpha=g.alpha
            ))
        best_ind = TriangleIndividual(tuple(scaled_genes))
        print(f"Escalando triángulos de vuelta a {orig_width}x{orig_height}...")

    final_image = render(best_ind, orig_width, orig_height)
    final_image.save(args.output)
    print(f"Imagen guardada en: {args.output}")


if __name__ == "__main__":
    main()
