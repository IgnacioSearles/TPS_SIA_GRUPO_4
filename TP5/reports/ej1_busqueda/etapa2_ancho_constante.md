# Ejercicio 1a, etapa 2: ancho_constante

Latente 2, 10000 épocas, controles cada 10 épocas, 10 semillas por fila.
Orden: cumple al final ↓, recaídas (redondeadas al 1 %) ↑, pasos mediana ↑, parámetros ↑.
Pasos = pasos de gradiente hasta cumplir por primera vez (batch 32: un paso por época).
⚠: alguna semilla cumplió recién en el último 10 % del entrenamiento (pueden haber faltado épocas).

| # | activación | inicializador | ocultas | optimizador | η | pérdida | batch | cumple al final | alguna vez | divergió | recaídas | pasos mediana (Q1–Q3) | parámetros |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | leaky_relu | he | 64-64-64 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 0.09 % | 160 (130–240) | 21541 |
| 2 | relu | he | 64-64-64 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 0.16 % | 200 (185–232) | 21541 |
| 3 | leaky_relu | he | 64-64 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 0.01 % | 205 (140–278) | 13221 |
| 4 | relu | he | 64-64 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 0.41 % | 205 (178–260) | 13221 |
| 5 | relu | he | 64-64 | adam | 0.003 | bce | 32 | 10/10 | 10/10 | 0 | 0.03 % | 290 (258–382) | 13221 |
| 6 | leaky_relu | he | 64-64 | adam | 0.003 | bce | 32 | 10/10 | 10/10 | 0 | 0.07 % | 330 (285–405) | 13221 |
| 7 | tanh | xavier | 64-64 | adam | 0.003 | bce | 32 | 10/10 | 10/10 | 0 | 0.02 % | 330 (298–372) | 13221 |
| 8 | leaky_relu | he | 64-64-64 | adam | 0.003 | bce | 32 | 10/10 | 10/10 | 0 | 0.08 % | 335 (235–412) | 21541 |
| 9 | tanh | xavier | 64-64-64 | adam | 0.003 | bce | 32 | 10/10 | 10/10 | 0 | 0.00 % | 350 (308–480) | 21541 |
| 10 | leaky_relu | he | 64-64-64 | adam | 0.001 | bce | 32 | 10/10 | 10/10 | 0 | 0.04 % | 460 (365–532) | 21541 |
| 11 | relu | he | 64-64-64 | adam | 0.001 | bce | 32 | 10/10 | 10/10 | 0 | 0.11 % | 470 (408–612) | 21541 |
| 12 | tanh | xavier | 64-64-64 | adam | 0.001 | bce | 32 | 10/10 | 10/10 | 0 | 0.01 % | 615 (582–698) | 21541 |
| 13 | leaky_relu | he | 64-64 | adam | 0.001 | bce | 32 | 10/10 | 10/10 | 0 | 0.12 % | 620 (560–685) | 13221 |
| 14 | relu | he | 64-64 | adam | 0.001 | bce | 32 | 10/10 | 10/10 | 0 | 0.01 % | 660 (580–690) | 13221 |
| 15 | tanh | xavier | 64-64-64 | adam | 0.01 | bce | 32 | 10/10 | 10/10 | 0 | 0.53 % | 170 (148–198) | 21541 |
| 16 | tanh | xavier | 64-64 | adam | 0.001 | bce | 32 | 10/10 | 10/10 | 0 | 1.00 % | 715 (662–745) | 13221 |
| 17 | relu | he | 64-64-64 | adam | 0.003 | bce | 32 | 9/10 | 10/10 | 0 | 0.31 % | 350 (318–385) | 21541 |
| 18 | tanh | xavier | 64-64 | adam | 0.01 | bce | 32 | 9/10 | 10/10 | 0 | 6.33 % | 170 (150–185) | 13221 |
