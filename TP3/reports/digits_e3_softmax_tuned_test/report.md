# Ejercicio 3: evaluación final en test

Arquitectura 784-128-128-10 (salida softmax) elegida por validación en `reports/digits_e3_softmax_tuned`, con balanceo y augmentación. Se evalúan las 5 semillas sobre las 2497 imágenes de `digits_test.csv`. El test no se usó para elegir nada.

**Consulta adicional al test.** El test ya se había usado para el resultado final de `reports/digits_e3_final` (sigmoide + MSE). Se evaluaron en una misma consulta las dos filas de `reports/digits_e3_softmax_tuned` (softmax y su referencia sigmoide, mismas 5 semillas y mismo protocolo) para compararlas de forma pareada. Nada se eligió mirando estos números.

| Semilla | Accuracy validación | Accuracy test | F1 macro test | Errores |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 98,78 % | 98,68 % | 0,9865 | 33 |
| 7 | 98,82 % | 98,52 % | 0,9850 | 37 |
| 21 | 98,80 % | 98,60 % | 0,9858 | 35 |
| 11 | 98,71 % | 98,72 % | 0,9870 | 32 |
| 2025 | 98,80 % | 98,32 % | 0,9829 | 42 |

**Media: 98,57 % ± 0,16 pp** (mín. 98,32 %, máx. 98,72 %).

## Por dígito (semilla 7, la mejor en validación)

| Dígito | Soporte | Recall | Precisión |
| ---: | ---: | ---: | ---: |
| 0 | 245 | 97,96 % | 97,96 % |
| 1 | 283 | 99,65 % | 100,00 % |
| 2 | 258 | 97,67 % | 98,82 % |
| 3 | 252 | 100,00 % | 96,92 % |
| 4 | 245 | 99,18 % | 99,59 % |
| 5 | 223 | 96,86 % | 99,08 % |
| 6 | 239 | 99,16 % | 97,53 % |
| 7 | 257 | 98,44 % | 97,31 % |
| 8 | 243 | 97,53 % | 99,16 % |
| 9 | 252 | 98,41 % | 98,80 % |

## Confusiones más frecuentes

- 5 leído como 3: 3
- 9 leído como 7: 2
- 8 leído como 3: 2
- 7 leído como 2: 2
- 5 leído como 0: 2
- 2 leído como 7: 2

Figuras: `test_confusion_matrix.png`, `test_errores.png`.
