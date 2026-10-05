# Ejercicio 3: qué aporta cambiar cómo aprende la red

Mismos datos en todas las filas (ambos archivos, balanceados y con imágenes desplazadas y rotadas) y **el mismo conjunto de validación** (4900 imágenes). Cada fila agrega un cambio a la anterior y se entrena con 3 semillas y 150 épocas.

Las filas que cambian la pérdida o el optimizador eligen primero su tasa de aprendizaje con una corrida corta (40 épocas, semilla 42) sobre la misma validación.

| Paso | Salida | Pérdida | Optimizador | η | Accuracy de validación | F1 macro |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Referencia: sigmoide + MSE + momentum | sigmoid | mse | momentum | 0.1 | 98,07 % ± 0,09 | 0,9751 |
| + softmax y entropía cruzada | softmax | cross_entropy | momentum | 0.01 | 98,37 % ± 0,03 | 0,9777 |
| + Adam | softmax | cross_entropy | adam | 0.001 | 98,29 % ± 0,03 | 0,9782 |

La referencia da 98,07 % y no el 98,13 % de `reports/digits_ablation`: la configuración es la misma, pero el balanceo de clases duplica imágenes al azar y aquí se sortean distinto (el script anterior balanceaba dos filas seguidas con el mismo generador). La diferencia, 0,05 puntos, es menor que el desvío entre semillas y las 3 filas de esta tabla comparten el mismo conjunto de entrenamiento.

## Búsqueda de tasa de aprendizaje

| Paso | η | Accuracy de validación |
| --- | ---: | ---: |
| + softmax y entropía cruzada | 0.001 | 93,96 % |
| + softmax y entropía cruzada | 0.003 | 96,69 % |
| + softmax y entropía cruzada | 0.01 | 97,86 % |
| + softmax y entropía cruzada | 0.03 | 97,73 % |
| + softmax y entropía cruzada | 0.1 | 97,49 % |
| + softmax y entropía cruzada | 0.3 | 96,35 % |
| + Adam | 0.0003 | 97,71 % |
| + Adam | 0.001 | 97,90 % |
| + Adam | 0.003 | 97,67 % |
| + Adam | 0.01 | 96,71 % |

## Evaluación sobre el test

Modelo de «+ softmax y entropía cruzada» (la mejor semilla según validación) sobre 2497 imágenes de `digits_test.csv`. Es la **segunda** consulta del test en el ejercicio 3: la primera fue la de `ablation.py`. Todas las decisiones de esta tabla se tomaron sobre validación.

- **Accuracy de test: 98,32 %**
- F1 macro: 0,9830
- Primera consulta («+ imágenes desplazadas y rotadas»): 97,56 %

| Dígito | Soporte | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0 | 245 | 99,18 % | 0,9918 |
| 1 | 283 | 99,65 % | 0,9877 |
| 2 | 258 | 98,06 % | 0,9844 |
| 3 | 252 | 99,60 % | 0,9843 |
| 4 | 245 | 99,18 % | 0,9939 |
| 5 | 223 | 96,41 % | 0,9751 |
| 6 | 239 | 98,74 % | 0,9833 |
| 7 | 257 | 99,22 % | 0,9770 |
| 8 | 243 | 95,47 % | 0,9727 |
| 9 | 252 | 97,22 % | 0,9800 |

## Reproducir

```bash
python -m experiments.digits.training_ablation
```
