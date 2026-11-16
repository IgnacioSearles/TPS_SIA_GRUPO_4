# Más ruido, mismos modelos

Evaluación sin reentrenar:224 notas, tres semillas y dos arquitecturas. Ruido de entrenamiento:σ0.1.

| σ | Entrada | Denso | CNN | Reducción CNN frente a entrada |
|---|---:|---:|---:|---:|
| 0.0 | 0.000000 | 0.016263 | 0.015674 | — |
| 0.05 | 0.001927 | 0.016156 | 0.015460 | -702.4% |
| 0.1 | 0.007413 | 0.016585 | 0.015362 | -107.2% |
| 0.2 | 0.027086 | 0.020476 | 0.016126 | 40.5% |
| 0.3 | 0.054999 | 0.030810 | 0.018666 | 66.1% |
| 0.4 | 0.086901 | 0.046607 | 0.023078 | 73.4% |
| 0.5 | 0.118569 | 0.062864 | 0.028396 | 76.1% |

La reducción negativa significa que empeora. Las magnitudes se recortan a[0,1]: a ruido fuerte también hay saturación. Se evalúa corrupción mel, no ruido acústico. Esta ampliación no modifica pesos ni los resultados guardados del estudio inicial.
