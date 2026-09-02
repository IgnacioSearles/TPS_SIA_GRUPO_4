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
    args = parser.parse_args()

    # Cargar objetivo
    print(f"Cargando imagen: {args.image}")
    original = Image.open(args.image)
    target = TriangleImageTarget(original)

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
    final_image = render(best.individual, target.width, target.height)
    final_image.save(args.output)
    print(f"Imagen guardada en: {args.output}")


if __name__ == "__main__":
    main()
