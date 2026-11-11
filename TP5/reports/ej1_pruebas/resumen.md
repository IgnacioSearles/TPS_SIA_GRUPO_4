# Ejercicio 1: pruebas extra para la presentación

## 1. Estabilidad del ≤ 1 píxel (`ej1a_stability`)

| ocultas | η | alguna vez cumple | cumple al final | época mediana en que cumple | controles fuera del objetivo después de cumplir |
|---|---|---|---|---|---|
| 32 | 0.01 | 10/10 | 9/10 | 400 | 3 % |
| 32 | 0.003 | 10/10 | 10/10 | 950 | 0 % |
| 32 | 0.001 | 10/10 | 10/10 | 2300 | 0 % |
| 32-16 | 0.01 | 10/10 | 9/10 | 350 | 17 % |
| 32-16 | 0.003 | 10/10 | 10/10 | 775 | 2 % |
| 32-16 | 0.001 | 10/10 | 10/10 | 1825 | 2 % |
| 64-32-16 | 0.01 | 10/10 | 10/10 | 200 | 0 % |
| 64-32-16 | 0.003 | 10/10 | 10/10 | 400 | 0 % |
| 64-32-16 | 0.001 | 10/10 | 10/10 | 925 | 0 % |

## 2. Por qué fallan las redes chicas (`ej1a_small_failures`)

| latente | ocultas | semillas que cumplen | letras mal (promedio) | píxeles mal (promedio) | letras que más fallan (semillas) |
|---|---|---|---|---|---|
| 1 | ninguna | 0/5 | 31.8 | 281.8 | 31 de 32 fallan en todas las semillas |
| 1 | 8 | 0/5 | 24.4 | 136.6 | 12 de 32 fallan en todas las semillas |
| 1 | 32-16 | 1/5 | 3.4 | 12.8 | o(2) \`(1) c(1) d(1) f(1) h(1) |
| 2 | ninguna | 0/5 | 29.8 | 201.2 | 29 de 32 fallan en todas las semillas |
| 2 | 8 | 0/5 | 15.6 | 63.2 | f(5) m(5) e(4) n(4) r(4) s(4) |
| 2 | 32-16 | 4/5 | 1.0 | 4.8 | n(1) r(1) u(1) v(1) w(1) |

## 3. Arquitectura del denoising (`ej1b_denoising_architectures`)

Píxeles mal por letra (de 35), promedio de 5 semillas y 50 repeticiones del font.

| ocultas | latente | letras limpias ≤ 1 píxel | salida @0,1 | salida @0,2 | salida @0,3 |
|---|---|---|---|---|---|
| 32-16 | 2 | 1/5 | 0.89 | 4.44 | 9.28 |
| 32-16 | 4 | 3/5 | 0.58 | 3.64 | 8.35 |
| 32-16 | 8 | 5/5 | 0.47 | 3.28 | 8.02 |
| 64-32 | 2 | 5/5 | 0.64 | 4.00 | 8.97 |
| 64-32 | 4 | 5/5 | 0.54 | 3.55 | 8.47 |
| 64-32 | 8 | 4/5 | 0.48 | 3.25 | 8.07 |
| (entrada sin red) | — | — | 3.48 | 6.99 | 10.50 |

## 4. Tipo de ruido (`ej1b_noise_types`)

Píxeles mal por letra a la salida. Nivel de cada ruido: invertir 0.1, apagar 0.2, sal y pimienta 0.2, gaussiano 0.4.

| entrenado con | probado con invertir | probado con apagar | probado con sal y pimienta | probado con gaussiano |
|---|---|---|---|---|
| invertir | 0.48 | 0.34 | 0.50 | 0.32 |
| apagar | 4.79 | 0.11 | 5.02 | 9.19 |
| sal y pimienta | 0.49 | 0.34 | 0.56 | 0.34 |
| gaussiano | 0.64 | 0.71 | 0.70 | 0.30 |
| (entrada sin red) | 3.48 | 2.33 | 3.52 | 3.65 |
