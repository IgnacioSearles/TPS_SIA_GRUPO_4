# Ejercicio 2 — Resultado final

## Fase 6 — Entrenamiento final y evaluación independiente

**Protocolo aceptado y fijado antes de consultar test:** red 784→64→32→10, tanh en
capas ocultas, sigmoide en salida, Xavier, MSE, momentum clásico 0,9, η=0,1,
lotes de 128 y 76 épocas. Semilla de pesos 42 y generador separado de lotes con
semilla 42. Se entrenó desde cero con **todo digits.csv: 12.449 imágenes**.

La duración de 76 épocas se tomó de la mediana de las mejores épocas de desarrollo
para momentum 0,9; no se buscó la mejor época en test. Con el dataset completo se
realizaron 98 actualizaciones por época (7.448 en total). Se guardaron los pesos al
final de la época 76 y se verificó su recarga antes de abrir el test. No hubo parada
temprana, selección de checkpoint ni cambios de hiperparámetros basados en test.
La configuración y el protocolo se escribieron a disco antes del entrenamiento.

### Resultado principal: todas las imágenes de test

| Medida | Resultado |
|---|---:|
| Imágenes de test | 2.497 |
| Aciertos | 2.145 |
| Errores | 352 |
| **Accuracy de test** | **85,90 %** |
| F1 macro de test (las diez clases) | 0,8122 |
| Error cuadrático promedio de test | 0,102130 |
| Accuracy de entrenamiento final | 99,58 % |
| Recall del 5 en test | 186/223 = 83,41 % |
| Recall del 8 en test | 0/243 = 0 % |

La accuracy de entrenamiento se informa como diagnóstico de ajuste, no como
estimación de generalización. El resultado oficial de esta evaluación es **85,90 %**;
no se sustituye por el resultado sobre un subconjunto favorable.

### Resultados por dígito

| Dígito real | Cantidad en test | Aciertos | Recall | F1 |
|---|---:|---:|---:|---:|
| 0 | 245 | 243 | 99.18 % | 0.9419 |
| 1 | 283 | 280 | 98.94 % | 0.9573 |
| 2 | 258 | 239 | 92.64 % | 0.8707 |
| 3 | 252 | 252 | 100.00 % | 0.8456 |
| 4 | 245 | 238 | 97.14 % | 0.9426 |
| 5 | 223 | 186 | 83.41 % | 0.7915 |
| 6 | 239 | 230 | 96.23 % | 0.9237 |
| 7 | 257 | 243 | 94.55 % | 0.9492 |
| 8 | 243 | 0 | 0.00 % | 0.0000 |
| 9 | 252 | 234 | 92.86 % | 0.9000 |

### Interpretación de los errores y cobertura

A diferencia de la validación de desarrollo, **test sí contiene las diez clases**.
Hay 243 imágenes de 8 y no hubo ejemplos de 8 en entrenamiento. La red no predijo
8 para ninguna imagen del test. Los 243 ochos fueron errores, aproximadamente el
69,03 % de los 352 errores totales. Se confundieron principalmente con 3 (67 casos),
5 (56) y 2 (42). La matriz de confusión conserva esas imágenes y muestra el problema.

Esto afecta también la precisión de las clases que reciben ochos mal clasificados.
Por ejemplo, se reconocieron los 252 treses reales (recall 100 %), pero la precisión
del 3 fue 73,26 %: varias imágenes de otros dígitos fueron clasificadas como 3.

Como **análisis secundario de cobertura**, sobre las 2.254 imágenes cuyo dígito sí
estuvo presente en entrenamiento hubo 2.145 aciertos: accuracy 95,16 % y F1 macro
0,9502. Es una descripción posterior calculada con las mismas predicciones; no se
reentrenó ni se seleccionó un segundo modelo, y no reemplaza la métrica global.

Además, el 5 representa una proporción mayor en test (223/2.497 = 8,93 %) que en
entrenamiento (271/12.449 = 2,18 %), y su recall es menor al de otras clases conocidas.
Por ello, no se debe comparar directamente la accuracy global de desarrollo con la
de test suponiendo que ambos conjuntos tienen idéntica composición.

El F1 macro de desarrollo promediaba nueve clases con soporte y no incluía al 8.
En test sí hay soporte para el 8, cuyo F1 es cero: el promedio principal incluye
las diez clases. Se documenta esta diferencia para evitar una comparación engañosa.

Se comprobó que no hay imágenes exactamente idénticas de test dentro de entrenamiento
mediante hash de sus vectores de píxeles. Este control no descarta semejanzas visuales
ni duplicados aproximados. No se quitaron imágenes del test para mejorar las métricas.

### Implementación y controles

Se agregaron `experiments/digits/final_evaluation.py`,
`experiments/digits/configs/final.json` y pruebas de la separación de responsabilidades.
Las pruebas usan archivos sintéticos y verifican que el test se lea después de guardar
pesos, que se entrene con todas las filas y que se conserven las clases ausentes de
entrenamiento en las métricas principales. Pasaron **99 pruebas** del proyecto.

Se hizo una evaluación del modelo sobre el test real, después del entrenamiento.
Las tablas por clase y el análisis de cobertura derivan de esa misma matriz de salidas.
Se guardaron hashes de configuración, pesos, dataset de entrenamiento y dataset de test;
el archivo de pesos permaneció intacto después de evaluar. La recarga de pesos se
verificó comparando exactamente las salidas de las primeras 128 imágenes de entrenamiento.

Desde TP3:

```bash
python -m experiments.digits.final_evaluation experiments/digits/configs/final.json
```

La ejecución escribe por defecto en `results/digits_final`. Rechaza un directorio
con resultados existentes para no reemplazar esta evaluación. Reproducir el mismo
protocolo en otro directorio no constituye una nueva evaluación independiente ni
autoriza a ajustar el modelo a partir del test ya observado.

La corrida original se guardó en `outputs/ejercicio2_final` y fue copiada al proyecto.
Archivos: `final_weights.npz`, `config.json`, `protocol.json`, `history.csv`,
`summary.json`, `test_predictions.csv`, `training_curves.png`,
`test_confusion_matrix.png` y `environment.json`. Las columnas `output_0`…`output_9`
son salidas sigmoides, no probabilidades normalizadas que sumen uno.

**Cierre:** no se cambió la configuración después de observar test y no se usó
`more_digits.csv`. El ejercicio 2 queda con experimentos de desarrollo, decisión
justificada y evaluación final documentados. La ausencia de la clase 8 es una
limitación central del sistema entrenado, no un motivo para ocultar su error.
