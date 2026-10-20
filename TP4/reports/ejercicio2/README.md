# Ejercicio 2: recuperación de códigos QR con Hopfield

Vocabulario: [`GLOSSARY.md`](../../GLOSSARY.md). Decisión de diseño: [ADR 0001](../../docs/adr/0001-hopfield-solo-con-regla-de-hebb.md) (sólo regla de Hebb).

## Reproducción

Desde TP4: `python3 analysis/hopfield_qr.py`. Requiere las dependencias de `requirements.txt` (incluye Pillow) y las cinco imágenes de `data/hopfield/`. Tarda unos segundos. Genera en esta carpeta, fuera de git:

- `resultados.csv`: una fila por recuperación (QR, nivel de ruido, semilla, modo de actualización, categoría, Hamming al patrón almacenado de origen, Hamming del patrón de consulta, iteraciones, motivo de parada, energía inicial y final, y un número de estado final).
- `categorias_sincronico.csv` y `categorias_sincronico.png`: cantidad de recuperaciones por categoría según el nivel de ruido.
- `solapamientos.csv` y `solapamientos.png`: solapamientos entre los cinco QR y mapa de los módulos iguales en los cinco.
- `crosstalk.csv` y `crosstalk.png`: crosstalk, neuronas inestables, estabilidad sin ruido y comparación con el estado mezcla para cada QR.
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

## Ortogonalidad y colapso de los QR

Antes de recuperar nada, el análisis mide qué tan parecidos son los cinco QR para la red. Escribe `solapamientos.csv` y `crosstalk.csv`, y las figuras `solapamientos.png` y `crosstalk.png`.

### Solapamientos y módulos compartidos

El solapamiento entre dos patrones almacenados es m^μν = (1/N)·Σⱼ ξⱼ^μ ξⱼ^ν: vale 1 para patrones iguales y 0 para patrones ortogonales.

| | Menú principal | Vinos | Cócteles y otras bebidas | Postres | Especiales del día |
| --- | --- | --- | --- | --- | --- |
| Menú principal | 1 | 0,860 | 0,600 | 0,594 | 0,766 |
| Vinos | 0,860 | 1 | 0,576 | 0,576 | 0,763 |
| Cócteles y otras bebidas | 0,600 | 0,576 | 1 | 0,757 | 0,594 |
| Postres | 0,594 | 0,576 | 0,757 | 1 | 0,614 |
| Especiales del día | 0,766 | 0,763 | 0,594 | 0,614 | 1 |

Ningún par es ni cerca de ortogonal: los solapamientos van de 0,576 a 0,860. Para comparar, dos patrones aleatorios de N = 1369 módulos tienen un solapamiento típico del orden de 1/√N ≈ 0,027; el par menos parecido de los QR está unas 21 veces por encima de eso y el más parecido (menú principal y vinos) unas 32 veces.

La razón se ve contando módulos: **955 de los 1369 módulos (69,8 %) son iguales en los cinco QR** (501 negros y 454 blancos). Incluyen los tres patrones de posición de las esquinas, que el estándar QR fija, y toda la mitad derecha del código salvo 4 módulos: 410 de los 414 módulos que difieren están en las columnas 1 a 19 (de 37). Para el ojo son cinco QR distintos; para la red, son cinco variantes de casi el mismo patrón.

![Solapamientos y módulos compartidos](solapamientos.png)

### Crosstalk

La condición de estabilidad del apunte es sign(hᵢ^ν) = ξᵢ^ν, con hᵢ^ν = Σⱼ wᵢⱼ ξⱼ^ν. Separando el término μ = ν de la suma de Hebb:

hᵢ^ν = (1/N)·Σⱼ Σ_{μ≠ν} ξᵢ^μ ξⱼ^μ ξⱼ^ν + (1/N)·Σⱼ ξᵢ^ν ξⱼ^ν ξⱼ^ν = Cᵢ^ν + **ξᵢ^ν**

El segundo término es ξᵢ^ν y no (1/N)·ξᵢ^ν como figura en el apunte, porque Σⱼ (ξⱼ^ν)² = N. El primero es el **crosstalk**, que se puede reescribir con los solapamientos:

Cᵢ^ν = (1/N)·Σⱼ Σ_{μ≠ν} ξᵢ^μ ξⱼ^μ ξⱼ^ν = Σ_{μ≠ν} m^μν·ξᵢ^μ

Con patrones ortogonales Cᵢ^ν = 0 y cada patrón almacenado es estable. Si no, la neurona i queda **inestable** cuando el crosstalk tiene signo opuesto a ξᵢ^ν y lo supera en módulo, es decir, cuando ξᵢ^ν·Cᵢ^ν < −1. (La red usa diagonal cero, que resta (5/N)·ξᵢ^ν ≈ 0,004·ξᵢ^ν al campo; ninguna neurona de los QR cae en ese margen, y la cuenta coincide con lo que hace la red, como muestra la tabla.)

| QR | \|Cᵢ\| medio | Crosstalk en los módulos compartidos | Neuronas con crosstalk opuesto | Neuronas inestables | Neuronas que cambian en la 1.ª iteración |
| --- | --- | --- | --- | --- | --- |
| Menú principal | 2,21 | +2,82 | 113 | 71 | 71 |
| Vinos | 2,20 | +2,78 | 118 | 91 | 91 |
| Cócteles y otras bebidas | 2,06 | +2,53 | 279 | 225 | 225 |
| Postres | 2,07 | +2,54 | 274 | 225 | 225 |
| Especiales del día | 2,18 | +2,74 | 158 | 103 | 103 |

- **El crosstalk es más del doble de la señal.** El término propio vale ±1 en cada neurona; el crosstalk vale en promedio entre 2,06 y 2,21 en módulo. El signo de hᵢ^ν lo decide el crosstalk, no el patrón consultado.
- **En los módulos compartidos refuerza.** Donde los cinco QR coinciden, ξᵢ^ν·Cᵢ^ν = Σ_{μ≠ν} m^μν (por ejemplo 0,860 + 0,600 + 0,594 + 0,766 = 2,82 para el menú principal): el crosstalk empuja hacia el valor común. Ninguna neurona inestable está entre los 955 módulos compartidos.
- **En los módulos que difieren arrastra hacia la mayoría.** Las neuronas inestables de cada QR son exactamente los módulos en los que ese QR está en minoría frente a los otros cuatro: su cantidad coincide con la distancia de Hamming al estado mezcla (71, 91, 225, 225 y 103).
- **La cantidad no es el problema.** La capacidad del apunte es 0,15·N = 0,15 × 1369 ≈ 205 patrones; cinco QR son el 2,4 % de eso. Esa cota supone patrones aleatorios, casi ortogonales entre sí; los QR fallan por estar correlacionados, no por ser muchos.

![Neuronas inestables de cada QR y estado mezcla](crosstalk.png)

### Estabilidad sin ruido y estado mezcla

- **Ningún QR es un estado estable.** Presentado sin ruido, cada uno cambia en la primera iteración exactamente las neuronas inestables de la tabla y llega a un estado estable en 1 iteración.
- **Ese estado es el estado mezcla sign(ξ₁ + … + ξ₅).** Los cinco QR sin ruido caen en él (5 de 5). El estado mezcla es estable: presentado como consulta, termina con 0 iteraciones. Con cinco patrones la suma nunca es cero, así que está bien definido.
- **A es el estado mezcla.** En el barrido sincrónico, 1051 de las 1100 salidas son iguales al estado mezcla y las 49 restantes (todas con p = 0,5) son su inverso; ninguna es otro estado. El estado A de la sección siguiente es, entonces, sign(ξ₁ + … + ξ₅), y B = −A es el inverso del estado mezcla. Como el estado mezcla toma en cada módulo el valor de la mayoría de los cinco QR, es el patrón "promedio" de los menús: se parece más a los QR que coinciden con la mayoría (menú principal, 71 módulos distintos) y menos a los que se apartan de ella (cócteles y postres, 225).

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

A es el estado mezcla sign(ξ₁ + … + ξ₅) y B su inverso; la sección [Ortogonalidad y colapso de los QR](#ortogonalidad-y-colapso-de-los-qr) lo verifica y explica el colapso con los solapamientos y el crosstalk.
