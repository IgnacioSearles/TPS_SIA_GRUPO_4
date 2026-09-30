# Ejercicio 3: qué aporta cada cambio

Cada fila cambia **una sola cosa** respecto de la anterior y se mide sobre **el mismo conjunto de validación** (4900 imágenes, con los diez dígitos). Cada fila se entrena con 3 semillas; se informa la media y el desvío. `digits_test.csv` no interviene en esta tabla.

| Paso | Filas de entrenamiento | Accuracy de validación | Recall del 8 | Recall del 5 | F1 macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ejercicio 2 (solo digits.csv) | 9967 | 93,78 % ± 0,17 | 0,00 % | 80,04 % | 0,8447 |
| + more_digits.csv (sin repetidos) | 19601 | 96,73 % ± 0,15 | 87,46 % | 88,54 % | 0,9572 |
| + balanceo de clases | 25700 | 96,84 % ± 0,21 | 88,89 % | 91,30 % | 0,9587 |
| + imágenes desplazadas y rotadas | 25700 | 98,13 % ± 0,08 | 94,02 % | 96,39 % | 0,9758 |

## Lectura

**(a) Mejor resultado,** El mejor paso es «+ imágenes desplazadas y rotadas», con 98,13 % de accuracy de validación, 4,35 puntos por encima del punto de partida,

**(b) Qué aportó cada técnica,** La tabla es la respuesta: cada fila agrega un solo cambio, De esos 4,35 puntos, 2,96 vienen de la primera fila, es decir de los datos,

**(c) Qué cambió además de nuestras técnicas.** El salto de la primera a la segunda fila no es un modelo mejor: es la aparición del dígito 8, que `digits.csv` no contenía. El recall del 8 pasa de 0,00 % a 87,46 % sin tocar la red. Ver `reports/digits_datasets/report.md`.

La accuracy de entrenamiento no sirve para elegir entre estas filas: aumentar los datos y desplazar las imágenes **empeoran** el error de entrenamiento a propósito. Por eso todas las filas se comparan por validación.

## Evaluación final sobre el test

Una única medición, con el modelo de «+ imágenes desplazadas y rotadas», sobre 2497 imágenes de `digits_test.csv` nunca usadas:

- **Accuracy de test: 97,56 %**
- F1 macro: 0,9753
- Recall del 8: 95,06 %
- Recall del 5: 96,86 %

| Dígito | Soporte | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0 | 245 | 99,18 % | 0,9858 |
| 1 | 283 | 99,29 % | 0,9947 |
| 2 | 258 | 96,90 % | 0,9728 |
| 3 | 252 | 97,62 % | 0,9609 |
| 4 | 245 | 99,59 % | 0,9859 |
| 5 | 223 | 96,86 % | 0,9686 |
| 6 | 239 | 97,07 % | 0,9768 |
| 7 | 257 | 98,05 % | 0,9749 |
| 8 | 243 | 95,06 % | 0,9645 |
| 9 | 252 | 95,63 % | 0,9679 |

## Reproducir

```bash
python -m experiments.digits.ablation --final-test
```

