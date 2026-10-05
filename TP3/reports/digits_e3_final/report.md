# Ejercicio 3: evaluación final en test

Arquitectura 784-128-128-10 elegida por validación en `reports/digits_e3_final/entrenamiento`, con balanceo y augmentación. Se evalúan las 3 semillas sobre las 2497 imágenes de `digits_test.csv`. El test no se usó para elegir nada.

| Semilla | Accuracy validación | Accuracy test | F1 macro test | Errores |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 98,67 % | 98,56 % | 0,9855 | 36 |
| 7 | 98,76 % | 98,80 % | 0,9879 | 30 |
| 21 | 98,63 % | 98,52 % | 0,9851 | 37 |

**Media: 98,63 % ± 0,15 pp** (mín. 98,52 %, máx. 98,80 %).

## Por dígito (semilla 7, la mejor en validación)

| Dígito | Soporte | Recall | Precisión |
| ---: | ---: | ---: | ---: |
| 0 | 245 | 98,78 % | 99,59 % |
| 1 | 283 | 100,00 % | 98,95 % |
| 2 | 258 | 98,45 % | 99,61 % |
| 3 | 252 | 98,81 % | 99,20 % |
| 4 | 245 | 99,18 % | 100,00 % |
| 5 | 223 | 98,65 % | 98,21 % |
| 6 | 239 | 99,16 % | 99,16 % |
| 7 | 257 | 99,22 % | 96,96 % |
| 8 | 243 | 97,53 % | 98,34 % |
| 9 | 252 | 98,02 % | 98,02 % |

## Confusiones más frecuentes

- 9 leído como 7: 3
- 4 leído como 9: 2
- 3 leído como 9: 2
- 2 leído como 7: 2
- 0 leído como 7: 2
- 9 leído como 8: 1

Figuras: `test_confusion_matrix.png`, `test_errores.png`.
