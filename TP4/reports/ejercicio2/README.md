# Ejercicio 2: recuperación de códigos QR con Hopfield

Vocabulario: [`GLOSSARY.md`](../../GLOSSARY.md). Decisión de diseño: [ADR 0001](../../docs/adr/0001-hopfield-solo-con-regla-de-hebb.md) (sólo regla de Hebb).

## Reproducción

Desde TP4: `python3 analysis/hopfield_qr.py`. Requiere las dependencias de `requirements.txt` (incluye Pillow) y las cinco imágenes de `data/hopfield/`. Tarda unos segundos. Genera en esta carpeta, fuera de git:

- `resultados.csv`: una fila por recuperación (QR, nivel de ruido, semilla, modo de actualización, categoría, Hamming al patrón almacenado de origen, Hamming del patrón de consulta, iteraciones, motivo de parada, energía inicial y final, y un número de estado final).
- `categorias_sincronico.csv` y `categorias_sincronico.png`: cantidad de recuperaciones por categoría según el nivel de ruido.
- `ejemplos_sincronico.png`: patrón almacenado, patrón de consulta y salida de la red para el QR del menú principal.

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

### Interpretación

- **Ningún QR es un estado estable.** Con p = 0 la consulta es el QR original y aun así la red se mueve en 1 iteración al estado A. Por lo tanto, el resultado no depende del ruido: la red no puede devolver un QR que no es un atractor.
- **Todas las consultas van al mismo atractor.** Para p ≤ 0,45 las 1000 recuperaciones llegan al estado A, salga la consulta del QR que salga. La red no distingue entre los cinco menús: siempre responde lo mismo. A queda más cerca del menú principal (71 módulos distintos, 5,2 % de 1369) que de cócteles y postres (225 módulos, 16,4 %).
- **El ruido sólo cambia el camino.** Al subir p, más consultas necesitan 2 iteraciones en lugar de 1, pero el destino es el mismo.
- **Con p = 0,5 la consulta ya no tiene información del original.** Cada módulo se invierte con probabilidad 1/2, así que la consulta es independiente del QR y se reparte casi a la mitad entre A (51) y su opuesto B (49). B es el inverso de un estado espurio, no de un patrón almacenado, así que también se clasifica como estado espurio.

Queda para la sección de ortogonalidad identificar A: la medición previa de la spec indica que es el estado mezcla sign(ξ₁ + … + ξ₅), pero este análisis todavía no lo verifica. Esa sección también va a explicar el colapso con el solapamiento y el crosstalk entre los QR.
