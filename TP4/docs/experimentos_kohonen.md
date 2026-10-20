# Experimentos de Kohonen para el ejercicio 1

Objetivo: cumplir la comparación de hiperparámetros y justificar una elección final mediante evidencia, como pide la página 3 del PDF. La interpretación de regiones, matriz U y campañas se revisará después sobre las corridas elegidas.

Hay **2.600 corridas exploratorias y 8 de verificación rápida**, preparadas pero sin ejecución automática de la batería completa. El máximo es **1.829.080 épocas y 91.454.000 actualizaciones** para 50 creadores. Early stopping puede reducir ese presupuesto. No se estima un tiempo sin medirlo en la máquina usada; checkpoints por época y exportación de gráficos también cuestan tiempo. Las ejecuciones son secuenciales y reanudables.

## Etapas y preguntas

Todos los JSON están en `configs/kohonen_suite/`. Cada etapa tiene su propio directorio bajo `results/kohonen_suite/`.

| Etapa | Corridas | Qué compara |
| --- | ---: | --- |
| `00_smoke` | 8 | Dos semillas, ambas topologías y entrada de 27 variables o PCA de 10; 10 épocas. Solo verifica el circuito. |
| `01_geometry` | 300 | 15 formas de grilla × 2 topologías × 2 inicializaciones × 5 semillas; 500 épocas. |
| `02_learning` | 260 | 26 esquemas de aprendizaje × 2 topologías × 5 semillas; 500 épocas. |
| `03_neighborhood` | 580 | 29 esquemas de radio × 2 vecindades × 2 topologías × 5 semillas; 500 épocas. |
| `04_representation` | 360 | 12 representaciones × 3 tamaños × 2 topologías × 5 semillas; 500 épocas. |
| `05_duration` | 60 | 50, 100, 250, 500, 1.000 y 2.000 épocas × 2 topologías × 5 semillas. |
| `06_early_stopping` | 240 | 4 umbrales × 3 paciencias × 2 mínimos de épocas × 2 topologías × 5 semillas; máximo de 1.000 épocas. |
| `07_order_initialization` | 80 | Con/sin mezcla × 2 inicializaciones × 2 topologías × 10 semillas; 1.000 épocas. |
| `08_reference_stability` | 120 | Grillas 3×3, 4×4 y 5×5 × 2 topologías × 20 semillas; 1.000 épocas. Es estabilidad de referencias, no confirmación de un ganador. |
| `09_interactions` | 600 | 120 combinaciones conjuntas, muestreadas sin reemplazo con semilla de diseño 20261018, × 5 semillas de entrenamiento; 1.000 épocas. |

Se comparten las semillas 0–4 entre configuraciones exploratorias. Esto permite comparaciones pareadas; no se elige la mejor semilla como resultado final. La etapa 07 usa 0–9 y la 08 usa 0–19.

### Geometría

Se prueban 2×2, 3×3, 4×4, 5×5, 6×6, 7×7, 8×8, 2×6, 6×2, 3×4, 4×3, 2×8, 8×2, 4×6 y 6×4. Los pares 2×6/3×4 y 2×8/4×4 ayudan a distinguir cantidad de neuronas de forma. Las orientaciones intercambiadas también se prueban, especialmente porque la hexagonal usa filas desplazadas.

Con 50 creadores, 7×7 y 8×8 son controles deliberados de mapas muy grandes; 8×8 tiene más neuronas que observaciones. No se consideran preferibles solo por bajar el error: pueden separar cada creador y dejar muchas celdas vacías.

### Aprendizaje, radio y orden

- Aprendizaje inicial: 0,05; 0,1; 0,3; 0,6. Finales: 0,001; 0,01; 0,05. Se incluyen constantes y decaimientos lineales/exponenciales, excluyendo combinaciones sin descenso redundantes con las constantes.
- Radios iniciales: 0,5; 1; 2; 4; 8. Finales: 0,1; 0,5; 1. Se incluyen constantes y descensos lineales/exponenciales. Vecindad gaussiana o uniforme. Las distancias de grilla difieren entre topologías: igual radio numérico no implica idéntico vecindario.
- Inicialización desde muestras o uniforme dentro del rango de cada variable. Con/sin mezcla de muestras por época.
- Early stopping: `min_delta` de 1e-6, 1e-4, 1e-3 o 1e-2; paciencia de 10, 30 o 60; mínimo de 50 o 150 épocas. Comparar contra el control de 1.000 épocas sin parada anticipada de la etapa 05. Con parada activada se evalúan los mejores pesos restaurados, según `selected_epoch` del experimento.

Los barridos específicos mantienen la referencia 4×4, z-score de 27 variables y los demás parámetros en sus valores base. Sirven para interpretar factores alrededor de esa referencia; no prueban por sí solos el óptimo conjunto. La etapa 09 combina factores para estudiar interacciones. Se muestrean también representaciones, formas y orden de presentación.

### Representaciones

Se comparan 27 variables seleccionadas; las 31 originales; las 27 con logaritmos en magnitudes; PCA de 2, 6, 10, 13 o 25 componentes; logaritmos más PCA de 6, 10 o 13; y una selección de 24 variables que conserva alcance pero retira seguidores, impresiones y reproducciones además de las cuatro derivadas habituales.

PCA de 2 es un control de pérdida de información, no una elección recomendada. PCA de 25 conserva todas las dimensiones de rango no nulo de la representación seleccionada y sirve como control frente a las 27 variables, cuyas proporciones introducen dos dependencias lineales. La variante de 24 variables explora el peso del bloque de visibilidad; no significa que las variables retiradas carezcan de valor comercial.

## Cómo ejecutar

Desde TP4, primero validar y correr la prueba rápida:

```bash
python3 -m experiments.kohonen_suite
python3 main.py experiment --config configs/kohonen_suite/00_smoke.json --dry-run
python3 main.py experiment --config configs/kohonen_suite/00_smoke.json
```

Luego ejecutar una etapa por vez. Para empezar por geometría:

```bash
python3 main.py experiment --config configs/kohonen_suite/01_geometry.json
```

También conviene revisar representaciones temprano (`04_representation`). Después seguir con aprendizaje, vecindad, duración y parada. La batería completa queda disponible; no hace falta esperar a terminar todas las etapas para mirar las primeras curvas y detectar configuraciones poco útiles.

Ctrl+C termina la época actual y pausa. Continuar con la ruta que imprime el programa:

```bash
python3 main.py resume --experiment results/kohonen_suite/01_geometry/<ejecucion>
```

Repetir el comando `experiment` crea otra ejecución desde cero; para continuar la existente usar `resume`. No modificar el código, datos o configuración del experimento entre pausa y reanudación. Se conserva el único checkpoint por corrida y el historial de mediciones escalares.

Los JSON preparados se pueden regenerar sin entrenar:

```bash
python3 -m experiments.kohonen_suite --write
```

El soporte nuevo `cases` combina parámetros acoplados sin multiplicarlos entre sí:

```json
{
  "algorithm": "kohonen",
  "seeds": [0, 1, 2],
  "cases": [
    {"model.rows": 2, "model.cols": 6},
    {"model.rows": 3, "model.cols": 4}
  ],
  "grid": {"model.topology": ["rectangular", "hexagonal"]}
}
```

Este ejemplo son 12 corridas: casos × topologías × semillas. No se permite definir un mismo parámetro, o uno de sus padres/hijos, simultáneamente en `cases` y `grid`.

## Cómo comparar

Después de cualquier etapa, incluso pausada:

```bash
python3 -m analysis.kohonen_experiments
```

Lee los planes y artefactos guardados, sin volver a entrenar. Genera en `results/kohonen_comparison/`:

- `runs.csv`: parámetros efectivos, estado, errores y diagnósticos de cada corrida encontrada.
- `summary.csv`: medias y desvíos entre semillas por configuración; incluye cantidad de éxitos y semillas únicas.
- `stability.csv`: ARI entre cada par de semillas de una misma configuración y catálogo.
- `errors.csv`: artefactos inválidos o datos que ya no coinciden con los de entrenamiento.

Las configuraciones repetidas entre etapas no cuentan como semillas independientes: para las estadísticas se usa una sola corrida exitosa por configuración, catálogo y semilla. Los resultados de semillas diferentes pueden generar mapas rotados o reflejados; ARI compara quién queda junto a quién, sin requerir que las coordenadas de las neuronas coincidan. Es estabilidad de asignaciones a neuronas, no de regiones agrupadas de la matriz U.

### Criterios para interpretar

1. Comparar QE **dentro de una misma representación**; las escalas cambian al modificar variables, logaritmos o componentes. TE describe vecindad y se complementa con ocupación y perfiles.
2. Revisar la fracción de celdas vacías y creadores aislados. Un mapa donde cada creador ocupa una neurona puede tener excelente QE y ARI sin resumir perfiles útiles.
3. Usar curvas y épocas efectivas para decidir si alcanza el entrenamiento. El decaimiento se distribuye sobre el máximo de épocas, por lo que cambiar duración también cambia la velocidad del programa de aprendizaje.
4. Revisar estabilidad entre semillas. No elegir una corrida excepcionalmente buena ignorando la dispersión de la configuración.
5. Interpretar perfiles y fronteras del mapa en variables originales antes de recomendarlo para campañas.

El comparador añade `common_centroid_distance`: reproyecta los creadores de todas las variantes a la referencia común de 27 variables estandarizadas y calcula la distancia media al centro observado de los creadores que comparten neurona. **No es el QE de los prototipos entrenados ni un error de validación**. Permite comparar cohesión en el mismo espacio, pero favorece mapas finos y vale cero para creadores aislados; por eso se informa junto con ocupación y fracción de singletons. `occupancy_entropy` resume qué tan distribuida está la ocupación, sin determinar por sí sola si el mapa es bueno.

No se produce un ranking global con ponderaciones arbitrarias. Seleccionar unos pocos candidatos que equilibren error, topología, ocupación, estabilidad e interpretación. Una mejora de estas métricas no demuestra aumentos de ventas o adopciones.

## Confirmación después de explorar

Las primeras nueve etapas son exploratorias y controles; todavía no hay ganadores. Una vez elegidos candidatos, generar una comparación con **30 semillas nuevas (100–129)** usando sus `config.json` guardados:

```bash
python3 -m experiments.kohonen_suite --confirm-from \
  results/kohonen_suite/<etapa>/<ejecucion>/run_0000/config.json \
  results/kohonen_suite/<etapa>/<ejecucion>/run_0001/config.json --write
python3 main.py experiment --config configs/kohonen_suite/10_confirmation.json
```

La cantidad será 30 × candidatos. Conviene pasar configuraciones diferentes, no varias semillas de la misma configuración. Se copian representación, grilla e hiperparámetros efectivos; las semillas y la salida cambian. Esta confirmación evalúa robustez frente a inicialización y orden, no generalización a nuevos creadores ni campañas.

## Medir si alcanza la duración mientras corre la batería

```bash
python3 -m analysis.kohonen_convergence
```

El diagnóstico solo lee corridas terminadas con 500 épocas y deja un snapshot, tablas y curvas en `results/kohonen_convergence/`. Se puede repetir mientras siguen terminando corridas; no modifica modelos, planes ni checkpoints. Informa cuánto baja QE durante la segunda mitad, las últimas 100 y las últimas 50 épocas, en qué época fue mínimo y cómo cambia TE.

Como señal descriptiva de meseta exige menos de 0,5% de cambio absoluto entre medias de ventanas consecutivas de 50 épocas, menos de 1% de amplitud de QE en la última ventana y cambio de TE medio de hasta 0,02. Es una heurística explícita, no una garantía de convergencia ni el criterio de early stopping del modelo. Un mínimo al final y una caída sostenida sugieren que conviene comparar con presupuestos mayores. La cobertura de tamaños/topologías y las semillas disponibles deben acompañar cualquier conclusión.

## Continuar más allá del presupuesto original sin repetir épocas

`resume` conserva el presupuesto original. Para aumentar explícitamente el objetivo de corridas ya terminadas:

```bash
python3 -m experiments.extend_kohonen \
  --source results/kohonen_suite/01_geometry/<ejecucion> \
  --target-epochs 1000 --output results/kohonen_continuation/1000
```

Se copian los checkpoints de las corridas exitosas a otra ejecución, preservando el origen. Se validan configuración, datos transformados, estado y versiones de Python/NumPy. Las corridas incompletas no se incluyen. `continuation.json` guarda procedencia, época de origen y hash del checkpoint original. Esta operación explícita migra el objetivo del checkpoint; no altera la ejecución fuente.

El parámetro nuevo `model.schedule_epochs` fija cuántas épocas dura el programa de decaimiento, independientemente del nuevo máximo. Al continuar 500→1.000 se conserva `schedule_epochs=500`; las épocas 501–1.000 mantienen el aprendizaje/radio del extremo del programa original. No se reinicia el generador ni se repite el historial. Es una prueba de entrenamiento adicional sobre la trayectoria existente, distinta de entrenar 1.000 épocas desde cero con un decaimiento más lento.

La ejecución nueva admite la reanudación habitual con `python3 main.py resume --experiment <ruta>`. Para continuar luego hasta 2.000 se pasa como fuente la ejecución de 1.000; el programa original de 500 se conserva. No cambiar el código entre pausa y reanudación.

## Recuperar un plan bloqueado por cambios de código

No editar manualmente `source_digest` ni desactivar la validación de `resume`. Si hay una evolución compatible del formato, se puede preparar un plan nuevo mediante una migración explícita de Kohonen:

```bash
python3 -m experiments.recover_kohonen \
  --source results/kohonen_suite/01_geometry/<ejecucion-original> \
  --completed-from results/kohonen_continuation/1000/<ejecucion-completa> \
  --target-epochs 1000 --output results/kohonen_geometry_1000
python3 main.py resume --experiment results/kohonen_geometry_1000
```

La recuperación prepara el plan; no entrena. Comprueba versiones de Python/NumPy, contenido del catálogo, parámetros, historial y digest de los datos transformados de los checkpoints. Migra explícitamente el parámetro opcional `schedule_epochs` de checkpoints previos. No sirve como garantía de compatibilidad ante cambios arbitrarios del algoritmo: debe usarse después de revisar el cambio y sus pruebas de continuidad, como en la incorporación de `schedule_epochs`.

`--completed-from` importa corridas exitosas cuyo objetivo y parámetros coinciden exactamente con el plan nuevo; no se repiten. Del origen se recuperan los checkpoints restantes, incluso los parciales; donde no existe checkpoint el entrenamiento empieza desde cero. El programa de decaimiento original se conserva. El destino debe ser una carpeta nueva: no se sobrescriben planes ni resultados originales. `recovery.json` conserva procedencia, hashes y época recuperada. El plan reconstruido vuelve a aplicar el control normal de firma al reanudarse.

## Etapas posteriores al barrido completo de geometría

Los [patrones observados](../reports/ejercicio1/geometria_kohonen/README.md) se basan en 300 corridas de 1.000 épocas y programa de decaimiento de 500. Para mantener ese protocolo al estudiar los demás factores se prepararon variantes en `configs/kohonen_suite_1000/`: 02, 03, 04, 06, 07, 08 y 09. Los JSON originales de 500 épocas permanecen como definición histórica de la batería.

Comandos, cantidades y diferencias con duración se detallan en [la guía de las etapas de 1.000 épocas](../configs/kohonen_suite_1000/README.md). Estas variantes no se han ejecutado al prepararlas. El barrido de geometría completado no implica que las otras etapas estén terminadas.
