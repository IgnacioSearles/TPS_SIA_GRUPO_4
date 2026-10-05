# Ejercicio 3: evaluación final en test

Arquitectura 784-128-128-10 (salida sigmoid) elegida por validación en `reports/digits_e3_softmax_tuned`, con balanceo y augmentación. Se evalúan las 5 semillas sobre las 2497 imágenes de `digits_test.csv`. El test no se usó para elegir nada.

**Consulta adicional al test.** El test ya se había usado para el resultado final de `reports/digits_e3_final` (sigmoide + MSE). Se evaluaron en una misma consulta las dos filas de `reports/digits_e3_softmax_tuned` (softmax y su referencia sigmoide, mismas 5 semillas y mismo protocolo) para compararlas de forma pareada. Nada se eligió mirando estos números.

| Semilla | Accuracy validación | Accuracy test | F1 macro test | Errores |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 98,67 % | 98,56 % | 0,9855 | 36 |
| 7 | 98,59 % | 98,64 % | 0,9863 | 34 |
| 21 | 98,61 % | 98,56 % | 0,9855 | 36 |
| 11 | 98,67 % | 98,72 % | 0,9869 | 32 |
| 2025 | 98,76 % | 98,36 % | 0,9834 | 41 |

**Media: 98,57 % ± 0,13 pp** (mín. 98,36 %, máx. 98,72 %).

## Por dígito (semilla 2025, la mejor en validación)

| Dígito | Soporte | Recall | Precisión |
| ---: | ---: | ---: | ---: |
| 0 | 245 | 99,59 % | 97,21 % |
| 1 | 283 | 100,00 % | 98,95 % |
| 2 | 258 | 97,67 % | 98,44 % |
| 3 | 252 | 99,21 % | 98,43 % |
| 4 | 245 | 98,37 % | 99,18 % |
| 5 | 223 | 97,76 % | 96,89 % |
| 6 | 239 | 98,33 % | 99,58 % |
| 7 | 257 | 99,22 % | 96,96 % |
| 8 | 243 | 96,71 % | 99,16 % |
| 9 | 252 | 96,43 % | 98,78 % |

## Confusiones más frecuentes

- 9 leído como 7: 3
- 9 leído como 4: 2
- 8 leído como 5: 2
- 8 leído como 3: 2
- 8 leído como 0: 2
- 7 leído como 2: 2

Figuras: `test_confusion_matrix.png`, `test_errores.png`.
