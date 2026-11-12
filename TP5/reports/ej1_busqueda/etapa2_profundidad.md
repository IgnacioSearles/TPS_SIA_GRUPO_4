# Ejercicio 1a, etapa 2: profundidad

Latente 2, 10000 épocas, controles cada 10 épocas, 10 semillas por fila.
Orden: cumple al final ↓, recaídas (redondeadas al 1 %) ↑, pasos mediana ↑, parámetros ↑.
Pasos = pasos de gradiente hasta cumplir por primera vez (batch 32: un paso por época).
⚠: alguna semilla cumplió recién en el último 10 % del entrenamiento (pueden haber faltado épocas).

| # | activación | inicializador | ocultas | optimizador | η | pérdida | batch | cumple al final | alguna vez | divergió | recaídas | pasos mediana (Q1–Q3) | parámetros |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | leaky_relu | he | 32-32 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 0.00 % | 235 (220–355) | 4581 |
| 2 | relu | he | 32-32 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 0.05 % | 315 (265–350) | 4581 |
| 3 | leaky_relu | he | 32-8 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 0.27 % | 330 (270–502) | 2901 |
| 4 | leaky_relu | he | 32-32-32 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 0.54 % | 295 (235–458) | 6693 |
| 5 | leaky_relu | he | 16-16 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 2.23 % | 770 (588–1900) | 1797 |
| 6 | leaky_relu | he | 16-16-16 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 4.24 % | 875 (535–1178) | 2341 |
| 7 | tanh | xavier | 32-8 | adam | 0.01 | bce | 32 | 9/10 | 10/10 | 0 | 3.54 % | 365 (345–412) | 2901 |
| 8 | relu | he | 16-16 | adam | 0.01 | bce | 32 | 9/10 | 9/10 | 0 | 4.28 % | 1000 (710–1845) | 1797 |
| 9 | tanh | xavier | 16-4 | adam | 0.01 | bce | 32 | 9/10 | 10/10 | 0 | 27.06 % | 1310 (1038–5755) ⚠ | 1341 |
| 10 | relu | he | 32-8 | adam | 0.01 | bce | 32 | 8/10 | 8/10 | 0 | 2.14 % | 440 (310–1608) | 2901 |
| 11 | relu | he | 32-32-32 | adam | 0.01 | bce | 32 | 8/10 | 10/10 | 0 | 8.41 % | 305 (235–872) | 6693 |
| 12 | tanh | xavier | 32-32 | adam | 0.01 | bce | 32 | 8/10 | 10/10 | 0 | 10.30 % | 325 (292–432) | 4581 |
| 13 | relu | he | 16-16-16 | adam | 0.01 | bce | 32 | 8/10 | 10/10 | 0 | 16.11 % | 2140 (590–2948) | 2341 |
| 14 | tanh | xavier | 32-32-32 | adam | 0.01 | bce | 32 | 7/10 | 9/10 | 0 | 8.89 % | 290 (190–335) | 6693 |
| 15 | tanh | xavier | 16-16 | adam | 0.01 | bce | 32 | 7/10 | 10/10 | 0 | 18.96 % | 1560 (918–2092) | 1797 |
| 16 | leaky_relu | he | 16-4 | adam | 0.01 | bce | 32 | 4/10 | 4/10 | 0 | 38.39 % | 6395 (2012–9330) ⚠ | 1341 |
| 17 | tanh | xavier | 16-16-16 | adam | 0.01 | bce | 32 | 3/10 | 8/10 | 0 | 51.65 % | 1070 (818–1915) | 2341 |
| 18 | relu | he | 16-4 | adam | 0.01 | bce | 32 | 0/10 | 1/10 | 0 | 49.69 % | 6750 | 1341 |
