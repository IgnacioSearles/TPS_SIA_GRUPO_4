# Ejercicio 3: qué aporta cada cambio

Cada fila cambia **una sola cosa** respecto de la anterior y se mide sobre **el mismo conjunto de validación** (4900 imágenes, con los diez dígitos). Cada fila se entrena con 3 semillas; se informa la media y el desvío. `digits_test.csv` no interviene en esta tabla.

| Paso | Filas de entrenamiento | Accuracy de validación | Recall del 8 | Recall del 5 | F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Solo digits.csv, sin técnicas | 10003 | 93,18 % ± 0,07 | 0,00 % | 82,59 % | 0,8392 |
| Solo digits.csv, balanceo y augmentación | 12321 | 95,49 % ± 0,07 | 0,00 % | 94,90 % | 0,8655 |
| Combinado, sin técnicas | 19601 | 96,17 % ± 0,19 | 85,75 % | 87,90 % | 0,9501 |
| Combinado, balanceo y augmentación | 25700 | 97,88 % ± 0,26 | 96,30 % | 97,66 % | 0,9733 |

## Lectura

**(a) Mejor resultado.** El mejor paso es «Combinado, balanceo y augmentación», con 97,88 % de accuracy de validación, 4,70 puntos por encima del punto de partida.

**(b) Qué aportó cada técnica.** La tabla es la respuesta: cada fila agrega un solo cambio. De esos 4,70 puntos, 2,31 vienen de la primera fila, es decir de los datos.

**(c) Qué cambió además de nuestras técnicas.** El salto de la primera a la segunda fila no es un modelo mejor: es la aparición del dígito 8, que `digits.csv` no contenía. El recall del 8 pasa de 0,00 % a 0,00 % sin tocar la red. Ver `reports/digits_datasets/report.md`.

La accuracy de entrenamiento no sirve para elegir entre estas filas: aumentar los datos y desplazar las imágenes **empeoran** el error de entrenamiento a propósito. Por eso todas las filas se comparan por validación.

## Reproducir

```bash
python -m experiments.digits.ablation --final-test
```

