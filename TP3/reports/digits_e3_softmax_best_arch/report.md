# Ejercicio 3: qué aporta cambiar cómo aprende la red

Mismos datos en todas las filas (ambos archivos, balanceados y con imágenes desplazadas y rotadas) y **el mismo conjunto de validación** (4900 imágenes). Cada fila agrega un cambio a la anterior y se entrena con 3 semillas y 150 épocas.

Las tasas de las variantes softmax/Adam se reutilizan del estudio previo con arquitectura [784,64,32,10]; no se vuelven a optimizar aquí.

| Paso | Salida | Pérdida | Optimizador | η | Accuracy validación | F1 macro | ECE top-score* | Score medio en errores |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Referencia: sigmoide + MSE + momentum | sigmoid | mse | momentum | 0.1 | 98,63 % ± 0,03 | 0,9822 | 0,0058 ± 0,0016 | 0,6836 |
| + softmax y entropía cruzada | softmax | cross_entropy | momentum | 0.01 | 98,71 % ± 0,10 | 0,9838 | 0,0032 ± 0,0007 | 0,7689 |
| + Adam | softmax | cross_entropy | adam | 0.001 | 98,72 % ± 0,03 | 0,9849 | 0,0051 ± 0,0006 | 0,7974 |

*ECE compara el score máximo con la frecuencia de acierto en 10 bins. Para softmax es una medida de calibración top-label; para salidas sigmoides independientes, el score máximo es solo un proxy diagnóstico, no una probabilidad categórica.

`mean_top_score_on_errors` resume cuán alto es el score asignado a la clase predicha cuando el modelo se equivoca. Ninguna de estas métricas se calcula sobre el test.

## Búsqueda de tasa de aprendizaje

| Paso | η | Accuracy de validación |
| --- | ---: | ---: |

## Reproducir

```bash
python -m experiments.digits.training_ablation
```
