# TP2 — Aproximación de imágenes con algoritmos genéticos

El proyecto contiene un motor genérico de algoritmos genéticos y una
implementación concreta que aproxima una imagen mediante triángulos
semitransparentes.

## Capas

- `genetic_algorithm.domain`: contratos para individuo, imagen objetivo,
  fitness, evaluación/comparación, problema, configuración y estado/resultado.
- `genetic_algorithm.application`: contratos de inicialización de población,
  selección, emparejamiento, supervivencia, cruza, mutación, terminación y motor
  evolutivo. También incluye `OrchestratedGeneticAlgorithm`, que coordina estos
  contratos sin decidir cómo funciona ninguna estrategia.

Todos usan `abc.ABC` y la sintaxis de parámetros de tipo de Python 3.12. Esto
permite elegir posteriormente cualquier representación de genoma, fitness o imagen.

## Cómo extenderlo

1. Implementar `Individual`, `Fitness` e `ImageTarget` con el formato elegido.
2. Implementar `FitnessEvaluator` y `FitnessComparator`.
3. Crear las estrategias concretas necesarias como subclases de los contratos.
4. Implementar un `GeneticAlgorithm` que las componga.

Se incluyen cuatro implementaciones iniciales: `EliteSelection`,
`AdditiveSurvival`, `OnePointCrossover` y `MultiGeneMutation`. Las dos últimas
reciben por inyección un `GenomeCodec`; MultiGen además recibe quién elige las
posiciones (`GenePositionSelector`) y cómo mutar cada gen (`GeneMutator`). Por lo
tanto, no dependen de una representación específica del individuo ni de una
mutación concreta.

`OrchestratedGeneticAlgorithm` ya ofrece ese ciclo genérico. Para usarlo se le
inyectan las estrategias, un `EvolutionContext` y una configuración que implemente
`EvolutionConfiguration` (`population_size` y `selected_parent_count`).

## Fitness

Además del MSE global, se pueden usar métricas regionales, SSIM, blur,
multiescala, histograma y bordes Sobel (`edge`). También están disponibles
`gradient` (magnitud y orientación de contornos), `chamfer` (distancia tolerante
a pequeños desplazamientos entre bordes) y `saliency` (MSE que pondera más las
zonas de alto contraste del objetivo). `--edge-sigma` controla el suavizado
previo a Sobel.

Se pueden combinar métricas con pesos relativos mediante `--fitness combo`; cada
una se normaliza antes de ponderarse. Por ejemplo:

```bash
python main.py --image objetivo.png --fitness combo \\
  --combo "mse:0.25,gradient:0.25,chamfer:0.25,saliency:0.25" --seed 42
```

Los pesos deben ser finitos y no negativos; el programa los renormaliza para que
sumen 1. Con `--seed` se puede repetir exactamente la misma ejecución.

## Pruebas

```bash
python -m unittest discover -v
```
