# Protocolo: vecindad uniforme en Kohonen

Fecha: 26/10/2026. Se repiten los experimentos recientes de finalistas, remuestreo y augmentation con dos variantes de vecindad uniforme. Los pesos solo viven en memoria. No se modifica ningún archivo de `presentation/`; el manifiesto conserva su inventario y hashes antes de comenzar y se vuelve a comprobar al terminar.

## Variantes fijadas antes de ver los resultados

| Nombre en resultados | Vecindad | Radio inicial/final | Resto del entrenamiento |
| --- | --- | --- | --- |
| `gaussian` | Gaussiana | 2 → 0,5 | Control existente |
| `uniform_matched` | Uniforme | 2 → 0,5 | Igual al control |
| `uniform_neighbors` | Uniforme | 2 → 1 | Igual al control |

Todas usan inicialización con muestras, tasa 0,3 → 0,01, decaimiento exponencial de tasa y radio durante las primeras 500 épocas y valores finales constantes durante las otras 500. No hay parada temprana. Se cambia la función de vecindad y, en la segunda variante, también el radio final. No se agregó la tasa 1/i sugerida como ejemplo por el apunte ni se afirma que todos sus detalles se repliquen literalmente.

La implementación existente de vecindad uniforme ya calcula `h = 1` si `distancia_grilla <= radio`, y cero fuera. Se usa la frontera inclusiva para incluir a las vecinas inmediatas con radio 1, como describe el texto del apunte; su fórmula con `<` se distingue explícitamente. No se simuló otra red recurrente ni se cambió la regla de elección de ganadora.

**Consecuencia importante:** en la uniforme de radio final 0,5 aprende solo la ganadora en esa etapa, porque las vecinas inmediatas están a distancia 1. Con radio final 1 también aprenden esas vecinas, todas con la misma tasa que la ganadora. El ancho gaussiano 0,5 tiene otro significado: no define un corte duro.

## Qué se repite

| Etapa | Diseño idéntico al previo | Modelos por variante uniforme | Nuevos entre ambas |
| --- | --- | ---: | ---: |
| Finalistas | 4×4 rectangular, 4×4 hexagonal y 2×8 hexagonal; semillas 100–129 | 90 | 180 |
| Submuestras | 100 grupos de 40/50 creadores; semillas 200/201/202 | 300 | 600 |
| Augmentation | 20 grupos de 40/50, semillas 400/401/402, intensidades 0/0,05/0,15/0,30 | 240 | 480 |
| Total | | **630** | **1.260** |

Se entrenan aparte dos referencias de semilla 0, una por variante uniforme, para describir ocupación, matriz U y perfiles regionales. Las pruebas de funcionamiento y réplicas de control se ejecutan separadamente y no entran en los 1.260 modelos.

Oja no se repite porque su aprendizaje no usa la vecindad de Kohonen. Tampoco se vuelven a ejecutar los 1.500 barridos históricos; el alcance son las tres pruebas recientes seleccionadas y corridas en esta conversación. No se cambia ningún resultado anterior.

## Emparejamiento y reutilización de la gaussiana

Se reutilizan **630 modelos gaussianos** de `confirmacion_20261022` y `augmentation_20261022/pilot`. Antes de reutilizarlos:

1. Se verifica el hash del catálogo y de cada fuente registrada en ambos manifiestos históricos.
2. Se exige correspondencia exacta entre tareas declaradas y salidas, sin fallos.
3. Se comprueba que las referencias gaussianas coincidan.
4. Se reproducen un finalista, una submuestra y una condición de augmentation gaussianos, comparando etiquetas y métricas con los archivos guardados, con tolerancia de 1e-12 para métricas.

Las tareas se toman de aquellos manifiestos: mismas semillas, índices de entrenamiento y niveles de perturbación, no otras particiones parecidas. La inicialización y el orden de ejemplos de cada par usan el mismo generador del entrenamiento. Solo cambia la vecindad/radio configurados.

## Evaluación

- Finalistas: QE y TE sobre los 50 perfiles, ocupación de neuronas y regiones, ARI entre semillas y contra la referencia.
- Submuestras: escalador ajustado solo con los 40 incluidos. Se evalúan los diez originales excluidos. QE, TE y diferencias contra la gaussiana se calculan sobre la misma partición y escala. Se promedian las tres semillas antes de contar victorias por submuestra.
- Augmentation: misma interpolación convexa hacia hasta tres vecinos del mismo tema elegidos solo entre los 40 incluidos; mitad de presentaciones intactas en esperanza. Las copias se regeneran por época y se conserva el mismo generador independiente del entrenamiento. Se evalúan originales excluidos, no copias.
- Presupuesto: 50.000 actualizaciones por modelo de catálogo completo; 40.000 por modelo de submuestra o augmentation. Mil épocas en todos los casos.
- Campañas: mismas reglas y referencia gaussianas. Se proyectan los 50 originales y se alinean etiquetas por máxima coincidencia de miembros. Es una comparación descriptiva de estabilidad bajo reglas congeladas, no una validación comercial independiente ni una garantía de que cada región conserve la misma semántica. No se reetiquetan objetivos para mejorar los resultados.

Los criterios de cribado de augmentation son los anteriores: mejora media de QE de excluidos al menos 1 %, mejora en al menos 70 % de particiones, aumento medio de TE no mayor a 0,02 y pérdida media de solapamiento de arte no mayor a 0,02. Se comparan contra el nivel cero de la **misma variante**. Cumplirlos identifica un candidato; no modifica automáticamente el modelo final ni demuestra confirmación en catálogos nuevos.

Los percentiles 5–95 describen diferencias entre semillas o entre medias de particiones, según la etapa. No son intervalos de confianza de una población nueva. Las particiones se superponen y se mantienen los mismos 50 creadores. El ARI mide reproducibilidad de agrupaciones, no exactitud comercial. TE usa la adyacencia propia de cada geometría, por lo que no sirve como único criterio entre topologías.

## Reproducción

Desde TP4, con sus dependencias y los controles anteriores disponibles:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m analysis.kohonen_uniform_experiments --output /tmp/uniforme-nuevo --workers 6
OPENBLAS_NUM_THREADS=1 python3 -m analysis.summarize_kohonen_uniform /tmp/uniforme-nuevo
# Si se interrumpió sin modificar fuentes, datos, controles o presentación:
OPENBLAS_NUM_THREADS=1 python3 -m analysis.kohonen_uniform_experiments --output /tmp/uniforme-nuevo --workers 6 --resume
python3 -m unittest tests.test_uniform_experiments
```

Las configuraciones de ejemplo nuevas `configs/kohonen_uniform_matched.json` y `configs/kohonen_uniform_neighbors.json` permiten ejecutar el 4×4 de referencia mediante la CLI existente. No se modificaron `MODEL`, los valores por defecto de Kohonen ni la implementación compartida.

El consolidado valida modelos, condiciones y presupuestos. Conserva métricas individuales, tablas agregadas, etiquetas por posición de fila, perfiles agregados y figuras. No guarda modelos, pesos, checkpoints ni copias sintéticas. Los controles gaussianos se copian como métricas para poder revisar los pares.
