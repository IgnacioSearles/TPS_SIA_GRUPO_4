# Ejercicio 1: pruebas extra para la presentación

## 1. Estabilidad del ≤ 1 píxel (`ej1a_stability`)

| ocultas | η | alguna vez cumple | cumple al final | época mediana en que cumple | controles fuera del objetivo después de cumplir |
|---|---|---|---|---|---|
| 32 | 0.01 | 10/10 | 9/10 | 400 | 2 % |
| 32 | 0.003 | 10/10 | 10/10 | 925 | 1 % |
| 32 | 0.001 | 10/10 | 10/10 | 2300 | 0 % |
| 32-16 | 0.01 | 10/10 | 9/10 | 350 | 14 % |
| 32-16 | 0.003 | 10/10 | 9/10 | 800 | 3 % |
| 32-16 | 0.001 | 10/10 | 10/10 | 1850 | 2 % |
| 64-32-16 | 0.01 | 10/10 | 10/10 | 200 | 0 % |
| 64-32-16 | 0.003 | 10/10 | 10/10 | 400 | 0 % |
| 64-32-16 | 0.001 | 10/10 | 10/10 | 925 | 0 % |

## 2. Por qué fallan las redes chicas (`ej1a_small_failures`)

| latente | ocultas | semillas que cumplen | letras mal (promedio) | píxeles mal (promedio) | letras que más fallan (semillas) |
|---|---|---|---|---|---|
| 1 | ninguna | 0/5 | 31.8 | 280.8 | 31 de 32 fallan en todas las semillas |
| 1 | 8 | 0/5 | 25.2 | 135.8 | 11 de 32 fallan en todas las semillas |
| 1 | 32-16 | 0/5 | 4.6 | 24.6 | x(4) c(3) y(2) \`(1) d(1) g(1) |
| 2 | ninguna | 0/5 | 29.8 | 201.0 | 29 de 32 fallan en todas las semillas |
| 2 | 8 | 0/5 | 14.6 | 64.4 | e(5) f(5) m(5) h(4) j(4) n(4) |
| 2 | 32-16 | 4/5 | 1.8 | 7.0 | b(1) c(1) e(1) f(1) i(1) k(1) |

## 3. Arquitectura del denoising (`ej1b_denoising_architectures`)

Píxeles mal por letra (de 35), promedio de 5 semillas y 50 repeticiones del font.

| ocultas | latente | letras limpias ≤ 1 píxel | salida @0,1 | salida @0,2 | salida @0,3 |
|---|---|---|---|---|---|
| 32-16 | 2 | 1/5 | 0.80 | 4.18 | 9.02 |
| 32-16 | 4 | 4/5 | 0.53 | 3.62 | 8.28 |
| 32-16 | 8 | 4/5 | 0.49 | 3.33 | 8.04 |
| 64-32 | 2 | 4/5 | 0.63 | 4.08 | 9.05 |
| 64-32 | 4 | 5/5 | 0.48 | 3.43 | 8.35 |
| 64-32 | 8 | 5/5 | 0.48 | 3.38 | 8.03 |
| (entrada sin red) | — | — | 3.48 | 6.99 | 10.50 |

## 4. Tipo de ruido (`ej1b_noise_types`)

Píxeles mal por letra a la salida. Nivel de cada ruido: invertir 0.1, apagar 0.2, sal y pimienta 0.2, gaussiano 0.4.

| entrenado con | probado con invertir | probado con apagar | probado con sal y pimienta | probado con gaussiano |
|---|---|---|---|---|
| invertir | 0.48 | 0.41 | 0.52 | 0.30 |
| apagar | 4.53 | 0.11 | 4.71 | 8.48 |
| sal y pimienta | 0.57 | 0.33 | 0.59 | 0.42 |
| gaussiano | 0.64 | 0.66 | 0.71 | 0.29 |
| (entrada sin red) | 3.48 | 2.33 | 3.52 | 3.65 |
