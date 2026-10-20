# Ejercicio 2: recuperación de códigos QR con Hopfield

Vocabulario: [`GLOSSARY.md`](../../GLOSSARY.md). Decisión de diseño: [ADR 0001](../../docs/adr/0001-hopfield-solo-con-regla-de-hebb.md) (sólo regla de Hebb).

## Reproducción

Desde TP4: `python3 analysis/hopfield_qr.py`. Requiere las dependencias de `requirements.txt` (incluye Pillow) y las cinco imágenes de `data/hopfield/`. Tarda unos segundos. Genera en esta carpeta, fuera de git:

- `resultados.csv`: una fila por recuperación (QR, nivel de ruido, semilla, modo de actualización, categoría, Hamming al patrón almacenado de origen, Hamming del patrón de consulta, iteraciones, motivo de parada, energía inicial y final, y un número de estado final).
- `categorias_sincronico.csv` y `categorias_sincronico.png`: cantidad de recuperaciones por categoría según el nivel de ruido.
- `ejemplos_sincronico.png`: patrón almacenado, patrón de consulta y salida de la red para el QR del menú principal.
- `iteraciones.csv` y `iteraciones.png`: recuperaciones por cantidad de iteraciones, según el modo de actualización y el nivel de ruido.
- `energia_ejemplos.png`: energía por iteración de consultas representativas del QR del menú principal.
- `subidas_energia.csv`: por modo y nivel de ruido, cuántas recuperaciones tuvieron una subida de energía entre dos iteraciones (columna `energy_rises` de `resultados.csv`).

## Modelo

La red está implementada desde cero con NumPy (`src/tp4/hopfield.py`) y sigue el apunte de la Unidad 9:

- **Patrones almacenados.** Los cinco QR de 37×37, en el orden menú principal, vinos, cócteles y otras bebidas, postres y especiales del día. Cada módulo es una neurona (N = 1369), negro = +1 y blanco = −1, leído fila por fila (`src/tp4/qr.py`). La carga falla si una imagen no es de 37×37 o no es binaria.
- **Pesos.** Regla de Hebb: W = (1/N)·Σ_μ ξ^μ (ξ^μ)ᵀ, con diagonal cero. Es simétrica y no cambia después de almacenar.
- **Modo de actualización sincrónico.** S(t+1) = sign(W·S(t)); una neurona con campo exactamente cero conserva su estado.
- **Iteración.** Un recálculo completo de las N neuronas. Se cuentan sólo las iteraciones que cambian el estado: el recálculo que confirma un estado estable no suma. Así, el ejemplo del apunte (consulta (1, −1, −1, −1)) llega a ξ¹ en 1 iteración.
- **Motivo de parada.** Estado estable (el recálculo no cambia nada), ciclo (se vuelve a un estado ya visitado) o tope (100 iteraciones sin estabilizarse).
- **Energía.** H = −½·sᵀWs, registrada para el estado inicial y después de cada iteración.

La clasificación es independiente de la recuperación (`src/tp4/recall_outcome.py`) y se hace respecto del patrón almacenado del que salió la consulta. Un estado estable es **recuperación correcta** (Hamming 0 al original), **otro patrón almacenado**, **patrón inverso** (−ξ de cualquier patrón almacenado) o **estado espurio** (cualquier otro). Una recuperación que termina por ciclo o tope se informa con su motivo de parada, porque sólo un estado estable puede ser un estado espurio.

## Patrones de consulta

Cada patrón de consulta invierte cada módulo de un patrón almacenado, de forma independiente, con probabilidad p (nivel de ruido). Se usan p ∈ {0; 0,05; …; 0,5} y 20 semillas por QR y por nivel: 5 × 11 × 20 = **1100 patrones de consulta**. El generador de cada consulta se siembra con (índice del QR, índice del nivel, semilla), así que cualquier fila de la tabla se puede rehacer. Con p = 0 las 20 consultas son el QR original sin cambios.

## Barrido en modo sincrónico

### Categorías

Las 1100 recuperaciones terminan en un **estado estable** (0 ciclos, 0 recuperaciones cortadas por el tope), y las 1100 son **estado espurio**. Ninguna es recuperación correcta, otro patrón almacenado ni patrón inverso, ni siquiera sin ruido:

| p | Recuperación correcta | Otro patrón almacenado | Patrón inverso | Estado espurio | Ciclo | Tope |
| --- | --- | --- | --- | --- | --- | --- |
| 0 a 0,5 (cada nivel) | 0 | 0 | 0 | 100 | 0 | 0 |

![Categorías según el nivel de ruido](categorias_sincronico.png)

### Adónde cae la red

Las 1100 salidas son sólo **dos estados distintos**, uno el opuesto exacto del otro (los dos tienen la misma energía, porque H(−s) = H(s)):

| p | Estado A | Estado B = −A |
| --- | --- | --- |
| 0 a 0,45 (cada nivel) | 100 | 0 |
| 0,5 | 51 | 49 |
| Total | 1051 | 49 |

Distancia de Hamming de cada estado al QR original (es la misma para todas las consultas de un mismo QR):

| QR | Hamming a A | Hamming a B |
| --- | --- | --- |
| Menú principal | 71 | 1298 |
| Vinos | 91 | 1278 |
| Cócteles y otras bebidas | 225 | 1144 |
| Postres | 225 | 1144 |
| Especiales del día | 103 | 1266 |

![Original, consulta y salida](ejemplos_sincronico.png)

### Iteraciones

Cantidad de recuperaciones (de 100 por nivel) según las iteraciones que tardan en llegar al estado estable:

| p | 0 | 0,05 | 0,1 | 0,15 | 0,2 | 0,25 | 0,3 | 0,35 | 0,4 | 0,45 | 0,5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 iteración | 100 | 87 | 86 | 72 | 76 | 74 | 67 | 62 | 61 | 43 | 7 |
| 2 iteraciones | 0 | 13 | 14 | 28 | 24 | 26 | 33 | 38 | 39 | 57 | 85 |
| 3 iteraciones | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |

![Iteraciones según el nivel de ruido](iteraciones.png)

La media pasa de 1 iteración con p = 0 a 1,57 con p = 0,45 y 2,01 con p = 0,5; el máximo es 2 hasta p = 0,45 y 3 sólo con p = 0,5. Con poco ruido la primera iteración ya lleva al estado final; con más ruido, en los ejemplos de la sección "Energía", la primera iteración deja a la red en un estado intermedio con energía apenas mayor que la del estado final (−2130,78 contra −2172,88) y la segunda completa el camino.

### Interpretación

- **Ningún QR es un estado estable.** Con p = 0 la consulta es el QR original y aun así la red se mueve en 1 iteración al estado A. Por lo tanto, el resultado no depende del ruido: la red no puede devolver un QR que no es un atractor.
- **Todas las consultas van al mismo atractor.** Para p ≤ 0,45 las 1000 recuperaciones llegan al estado A, salga la consulta del QR que salga. La red no distingue entre los cinco menús: siempre responde lo mismo. A queda más cerca del menú principal (71 módulos distintos, 5,2 % de 1369) que de cócteles y postres (225 módulos, 16,4 %).
- **El ruido sólo cambia el camino.** Al subir p, más consultas necesitan 2 iteraciones en lugar de 1, pero el destino es el mismo.
- **Con p = 0,5 la consulta ya no tiene información del original.** Cada módulo se invierte con probabilidad 1/2, así que la consulta es independiente del QR y se reparte casi a la mitad entre A (51) y su opuesto B (49). B es el inverso de un estado espurio, no de un patrón almacenado, así que también se clasifica como estado espurio.

Queda para la sección de ortogonalidad identificar A: la medición previa de la spec indica que es el estado mezcla sign(ξ₁ + … + ξ₅), pero este análisis todavía no lo verifica. Esa sección también va a explicar el colapso con el solapamiento y el crosstalk entre los QR.

## Energía

La energía de un estado S es H = −½·Σᵢⱼ wᵢⱼ SᵢSⱼ = −½·sᵀWs (fórmula del apunte; un test la verifica en el ejemplo del apunte: H(ξ¹) = H(ξ²) = −3). Se registra para el patrón de consulta y después de cada iteración.

### Curvas de ejemplo

Energía por iteración de la consulta con semilla 0 del QR del menú principal (la línea punteada es la energía del QR original, −2077,45):

| p | Iteración 0 (consulta) | Iteración 1 | Iteración 2 |
| --- | --- | --- | --- |
| 0 | −2077,45 | −2172,88 | |
| 0,2 | −769,80 | −2172,88 | |
| 0,45 | −3,84 | −2130,78 | −2172,88 |
| 0,5 | 0,93 | −2127,67 | −2172,88 |

![Energía por iteración](energia_ejemplos.png)

- **Las 1100 recuperaciones terminan con la misma energía, −2172,88**: la de A y la de B = −A, que es igual porque H(−s) = H(s).
- **Esa energía es menor que la de cualquier QR original**: −2077,45 (menú principal), −2041,53 (vinos), −1789,71 (cócteles y otras bebidas), −1801,85 (postres) y −1982,47 (especiales del día). Los QR no son mínimos locales de H: como ninguno es estado estable, alguna neurona tiene campo de signo contrario a su estado, y según la demostración del apunte invertir sólo esa neurona baja la energía. Por eso aun sin ruido (p = 0) la red se va del QR hacia A.
- **Cuanto más ruido, más alta arranca la energía**: la energía media del patrón de consulta sube de −1938,60 con p = 0 a −696,45 con p = 0,2, −18,36 con p = 0,45 y −0,33 con p = 0,5 (`resultados.csv`, columna `energy_initial`). Con p = 0,5 la consulta es aleatoria y no se parece a ningún patrón almacenado, así que su energía queda cerca de 0.

### Subidas de energía en modo sincrónico

La demostración del apunte de que H nunca sube supone que **cambia una neurona por vez**. En modo sincrónico cambian varias juntas y la cuenta tiene un término más. Si C es el conjunto de neuronas que cambian en una iteración y hᵢ = Σⱼ wᵢⱼ Sⱼ es el campo de la neurona i antes de la iteración:

ΔH = −2·Σ_{i∈C} |hᵢ| − 2·Σ_{i, j∈C, i≠j} wᵢⱼ SᵢSⱼ

El primer término es la suma de lo que bajaría la energía si cada neurona cambiara sola (el caso del apunte) y siempre es negativo. El segundo depende de los pesos entre las neuronas que cambian a la vez y puede ser positivo, así que en modo sincrónico la energía puede subir. Un test lo muestra con tres patrones ortogonales de 4 neuronas: desde su estado mezcla (1, 1, 1, −1), con H = 0, la red cambia tres neuronas a la vez y llega a (1, −1, −1, 1), con H = 1,5; después oscila entre ese estado y su opuesto (ciclo).

En los QR no pasa: **0 de las 1100 recuperaciones sincrónicas tuvieron una subida de energía** entre dos iteraciones, en ningún nivel de ruido (`subidas_energia.csv`). La garantía del apunte no aplica al modo sincrónico, pero en este barrido la energía bajó en todas las iteraciones.
