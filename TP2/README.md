# TP2 — Motor de Algoritmos Genéticos

Esta entrega contiene solamente la arquitectura extensible del ejercicio 2. No
incluye estrategias concretas ni representa todavía triángulos, colores, imágenes
o valores de fitness.

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

## Pruebas

```bash
python -m unittest discover -v
```
