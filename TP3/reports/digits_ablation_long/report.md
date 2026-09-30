# Ejercicio 3: qué aporta cada cambio

Cada fila cambia **una sola cosa** respecto de la anterior y se mide sobre **el mismo conjunto de validación** (4900 imágenes, con los diez dígitos). Cada fila se entrena con 3 semillas; se informa la media y el desvío. `digits_test.csv` no interviene en esta tabla.

| Paso | Filas de entrenamiento | Accuracy de validación | Recall del 8 | Recall del 5 | F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ejercicio 2 (solo digits.csv) | 9967 | 93,90 % ± 0,14 | 0,00 % | 82,17 % | 0,8465 |
| + more_digits.csv (sin repetidos) | 19601 | 96,77 % ± 0,18 | 88,03 % | 88,96 % | 0,9576 |
| + balanceo de clases | 25700 | 96,84 % ± 0,21 | 88,89 % | 91,30 % | 0,9587 |
| + imágenes desplazadas y rotadas | 25700 | 98,29 % ± 0,02 | 96,01 % | 96,39 % | 0,9778 |

## Lectura

**(a) Mejor resultado,** El mejor paso es «+ imágenes desplazadas y rotadas», con 98,29 % de accuracy de validación, 4,39 puntos por encima del punto de partida,

**(b) Qué aportó cada técnica,** La tabla es la respuesta: cada fila agrega un solo cambio, De esos 4,39 puntos, 2,87 vienen de la primera fila, es decir de los datos,

**(c) Qué cambió además de nuestras técnicas.** El salto de la primera a la segunda fila no es un modelo mejor: es la aparición del dígito 8, que `digits.csv` no contenía. El recall del 8 pasa de 0,00 % a 88,03 % sin tocar la red. Ver `reports/digits_datasets/report.md`.

La accuracy de entrenamiento no sirve para elegir entre estas filas: aumentar los datos y desplazar las imágenes **empeoran** el error de entrenamiento a propósito. Por eso todas las filas se comparan por validación.

## Reproducir

```bash
python -m experiments.digits.ablation --final-test
```

