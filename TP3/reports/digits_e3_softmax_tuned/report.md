# Ejercicio 3: qué aporta cambiar cómo aprende la red

Mismos datos en todas las filas (ambos archivos, balanceados y con imágenes desplazadas y rotadas) y **el mismo conjunto de validación** (4900 imágenes). Cada fila agrega un cambio a la anterior y se entrena con 5 semillas y 150 épocas.

Las variantes que cambian la pérdida o el optimizador seleccionan su tasa con una corrida corta (40 épocas) sobre validación.

| Paso | Salida | Pérdida | Optimizador | η | Accuracy validación | F1 macro | ECE top-score* | Score medio en errores |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Referencia: sigmoide + MSE + momentum | sigmoid | mse | momentum | 0.1 | 98,66 % ± 0,06 | 0,9828 | 0,0058 ± 0,0013 | 0,6927 |
| + softmax y entropía cruzada | softmax | cross_entropy | momentum | 0.03 | 98,78 % ± 0,04 | 0,9847 | 0,0043 ± 0,0006 | 0,7801 |

*ECE compara el score máximo con la frecuencia de acierto en 10 bins. Para softmax es una medida de calibración top-label; para salidas sigmoides independientes, el score máximo es solo un proxy diagnóstico, no una probabilidad categórica.

`mean_top_score_on_errors` resume cuán alto es el score asignado a la clase predicha cuando el modelo se equivoca. Ninguna de estas métricas se calcula sobre el test.

## Búsqueda de tasa de aprendizaje

| Paso | η | Accuracy de validación |
| --- | ---: | ---: |
| + softmax y entropía cruzada | 0.003 | 96,57 % |
| + softmax y entropía cruzada | 0.01 | 97,78 % |
| + softmax y entropía cruzada | 0.03 | 98,41 % |
| + softmax y entropía cruzada | 0.1 | 98,14 % |

## Reproducir

```bash
python -m experiments.digits.training_ablation
```
