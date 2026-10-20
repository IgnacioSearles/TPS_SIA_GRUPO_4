# Etapas adicionales con protocolo de 1.000 épocas

Variantes de `configs/kohonen_suite/` para continuar el estudio después del barrido de geometría recuperado: `epochs=1000`, `schedule_epochs=500`. La tasa y el radio terminan su decaimiento en 500 y mantienen esos valores hasta el máximo. Early stopping puede cortar antes. Las salidas se separan bajo `results/kohonen_suite_1000/`.

Las combinaciones y semillas originales se conservan. No se modifican los planes históricos ni se lanzan entrenamientos al preparar estos archivos.

| Archivo | Corridas | Pregunta |
| --- | ---: | --- |
| `02_learning.json` | 260 | Efectos de tasas iniciales/finales y decaimiento |
| `03_neighborhood.json` | 580 | Efectos de radio, decaimiento de radio y tipo de vecindad |
| `04_representation.json` | 360 | Efectos de selección, logaritmos y dimensiones PCA |
| `06_early_stopping.json` | 240 | Cuánto entrenamiento se ahorra y qué calidad se pierde al detener antes |
| `07_order_initialization.json` | 80 | Influencia del orden de presentación y la inicialización, con diez semillas |
| `08_reference_stability.json` | 120 | Distribución de resultados de referencias con veinte semillas |
| `09_interactions.json` | 600 | Interacciones de factores mediante 120 combinaciones conjuntas y cinco semillas |

Ejecutar **una etapa por vez** desde TP4, por ejemplo:

```bash
python3 main.py experiment --config configs/kohonen_suite_1000/04_representation.json
```

Cambiar el nombre del JSON por la etapa deseada. Añadir `--dry-run` muestra el número de corridas sin entrenar. Un primer lanzamiento crea una ejecución nueva; para continuar una pausa usar el comando `resume` impreso por el programa, no volver a lanzar `experiment`.

Orden de análisis sugerido: representación, vecindad, aprendizaje, orden/inicialización, parada anticipada, estabilidad e interacciones. Los barridos por factor usan una referencia 4×4; las interacciones permiten contrastar factores en conjunto. Las etapas incluyen controles compartidos: el ejecutor los vuelve a ejecutar si se inicia otro experimento; el comparador no los trata como semillas independientes adicionales si configuración, catálogo y semilla coinciden.

Para comparar etapas después de ejecutarlas:

```bash
python3 -m analysis.kohonen_experiments --root results/kohonen_suite_1000 --output results/kohonen_followup_analysis
```

No comparar directamente QE entre representaciones distintas. Utilizar también cohesión en el espacio común, ocupación y estabilidad, con las limitaciones del [protocolo](../../docs/experimentos_kohonen.md).

`05_duration.json` se mantiene en la carpeta original: prueba máximos de 50–2.000 con decaimiento repartido sobre cada máximo, por lo que cambia simultáneamente duración y ritmo de aprendizaje. Es un control de otro protocolo; no equivale a continuar nuestros checkpoints ni integra esta carpeta. El comando es `python3 main.py experiment --config configs/kohonen_suite/05_duration.json`.
