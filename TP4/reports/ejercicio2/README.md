# Ejercicio 2: recuperación de códigos QR con Hopfield

La red usa sólo la regla de Hebb del apunte. Otras alternativas (pseudoinversa, dos redes, sacar los módulos compartidos, una red por zona del QR) se comparan aparte, en [`mejoras/`](mejoras/README.md), [`variante/`](variante/README.md), [`dos_redes/`](dos_redes/README.md) y [`zonas/`](zonas/README.md).

## Resumen

Una red de Hopfield con la regla de Hebb del apunte almacena los cinco QR de la Trattoria Aurora y recibe 1100 patrones de consulta (cada QR con niveles de ruido de 0 a 0,5), con actualización sincrónica. **Ninguna de las 1100 recuperaciones es una recuperación correcta**, ni siquiera con el QR original sin ruido: ningún QR es un estado estable, y todas las consultas caen en el estado mezcla sign(ξ₁ + … + ξ₅) o, sólo con p = 0,5, en su inverso. La causa no es la cantidad de patrones (5 frente a una capacidad de 0,15·N ≈ 205) sino que los cinco QR son casi iguales para la red: comparten 955 de sus 1369 módulos y su crosstalk es más del doble de la señal. Un lector de QR real decodifica el original pero ninguna de las 1100 salidas de la red. La sección [Limitaciones](#limitaciones) junta los números.

## Reproducción

Desde TP4: `python3 analysis/hopfield_qr.py`. Requiere las dependencias de `requirements.txt` (incluye Pillow y, para el lector de QR, `opencv-python`) y las cinco imágenes de `data/hopfield/`. Tarda unos 10 segundos. Genera en esta carpeta, fuera de git:

- `resultados.csv`: una fila por recuperación, 1100 en total (QR, nivel de ruido, semilla, categoría, Hamming al patrón almacenado de origen, Hamming del patrón de consulta, iteraciones, motivo de parada, energía inicial y final, y un número de estado final).
- `categorias.csv` y `categorias.png`: cantidad de recuperaciones por categoría según el nivel de ruido.
- `solapamientos.csv` y `solapamientos.png`: solapamientos entre los cinco QR y mapa de los módulos iguales en los cinco.
- `crosstalk.csv` y `crosstalk.png`: crosstalk, neuronas inestables, estabilidad sin ruido y comparación con el estado mezcla para cada QR.
- `ejemplos.png`: patrón almacenado, patrón de consulta y salida de la red para el QR del menú principal.
- `iteraciones.csv` y `iteraciones.png`: recuperaciones por cantidad de iteraciones, según el nivel de ruido.
- `energia_ejemplos.png`: energía por iteración de consultas representativas del QR del menú principal.
- `subidas_energia.csv`: por nivel de ruido, cuántas recuperaciones tuvieron una subida de energía entre dos iteraciones (columna `energy_rises` de `resultados.csv`).
- `lector_qr.csv`, `lector_qr_resumen.csv` y `lector_qr_umbral.csv`: experimento secundario con el lector de QR (ver su sección).

## Modelo

La red está implementada desde cero con NumPy (`src/tp4/hopfield.py`) y sigue el apunte de la Unidad 9:

- **Patrones almacenados.** Los cinco QR de 37×37, en el orden menú principal, vinos, cócteles y otras bebidas, postres y especiales del día. Cada módulo es una neurona (N = 1369), negro = +1 y blanco = −1, leído fila por fila (`src/tp4/qr.py`). La carga falla si una imagen no es de 37×37 o no es binaria.
- **Pesos.** Regla de Hebb: wᵢⱼ = (1/N)·Σ_μ ξᵢ^μ ξⱼ^μ para i ≠ j y wᵢᵢ = 0; en forma matricial, W = (1/N)·K·Kᵀ con la diagonal puesta en cero, donde K tiene los cinco patrones almacenados en columnas. Es simétrica y no cambia después de almacenar: la red no aprende nada más.
- **Campo.** hᵢ = Σⱼ wᵢⱼ Sⱼ; la neurona i pasa a sign(hᵢ).
- **Actualización sincrónica.** S(t+1) = sign(W·S(t)), el algoritmo del apunte: todas las neuronas se recalculan a la vez. Una neurona con campo exactamente cero conserva su estado.
- **Iteración.** Un recálculo completo de las N neuronas. La cantidad de iteraciones de una recuperación cuenta **sólo los recálculos que cambiaron el estado**: el recálculo final que confirma un estado estable se hace pero no suma. Así, 0 iteraciones significa que el patrón de consulta ya era un estado estable, y el ejemplo del apunte (consulta (1, −1, −1, −1)) llega a ξ¹ en 1 iteración. Todas las cantidades de iteraciones de este informe (tablas, figuras y `resultados.csv`) se cuentan así.
- **Motivo de parada.** Cada recálculo se evalúa en este orden: estado estable (no cambia nada), ciclo (vuelve a un estado ya visitado en esa recuperación; se compara con todos los visitados, no sólo con el anterior) o tope (100 iteraciones sin estabilizarse). Un estado estable nunca se informa como ciclo, y si el recálculo hecho al llegar al tope vuelve a un estado visitado, la recuperación es ciclo y no tope.
- **Energía.** H = −½·Σᵢⱼ wᵢⱼ SᵢSⱼ = −½·SᵀWS (fórmula del apunte), registrada para el estado inicial y después de cada iteración.

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

Esto es consistente con lo que codifican: las cinco URL que devuelve el lector (sección [Experimento secundario](#experimento-secundario-lector-de-qr)) comparten sus primeros 72 caracteres (`https://github.com/eugepineiro/sia-tips/blob/feature/tp4-images/TPs/TP4/`) y sólo difieren en el nombre del archivo (de 85 a 90 caracteres en total). El estándar QR ubica los datos desde la esquina inferior derecha hacia la izquierda y deja la corrección de errores, que depende de todo el contenido, para el final; un prefijo común ocupa entonces las mismas columnas de la derecha en los cinco códigos. Los cinco tienen además la misma información de formato (nivel de corrección y máscara).

![Solapamientos y módulos compartidos](solapamientos.png)

### Crosstalk

La condición de estabilidad del apunte es sign(hᵢ^ν) = ξᵢ^ν, con hᵢ^ν = Σⱼ wᵢⱼ ξⱼ^ν. Separando el término μ = ν de la suma de Hebb:

hᵢ^ν = (1/N)·Σⱼ Σ_{μ≠ν} ξᵢ^μ ξⱼ^μ ξⱼ^ν + (1/N)·Σⱼ ξᵢ^ν ξⱼ^ν ξⱼ^ν = Cᵢ^ν + **ξᵢ^ν**

El segundo término es ξᵢ^ν y no (1/N)·ξᵢ^ν como figura en el apunte, porque Σⱼ (ξⱼ^ν)² = N. El primero es el **crosstalk**, que se puede reescribir con los solapamientos:

Cᵢ^ν = (1/N)·Σⱼ Σ_{μ≠ν} ξᵢ^μ ξⱼ^μ ξⱼ^ν = Σ_{μ≠ν} m^μν·ξᵢ^μ

Con patrones ortogonales Cᵢ^ν = 0 y cada patrón almacenado es estable. La red, además, usa diagonal cero: la suma de Hebb pondría p/N en wᵢᵢ y la red lo anula, así que el campo real es

hᵢ^ν = ξᵢ^ν·(1 − p/N) + Cᵢ^ν, con p = 5 patrones almacenados,

y la neurona i queda **inestable** cuando hᵢ^ν tiene signo opuesto a ξᵢ^ν (ξᵢ^ν·hᵢ^ν < 0): el crosstalk se opone a ξᵢ^ν y supera en módulo a 1 − 5/N. El análisis calcula las neuronas inestables con ese campo y verifica que son exactamente las que cambian en la primera iteración sincrónica (si no coincidieran, el script se detiene); por eso las dos últimas columnas de la tabla son iguales.

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
- **La cantidad no es el problema.** La capacidad del apunte es 0,15·N = 0,15 × 1369 ≈ 205 patrones; cinco QR son el 2,4 % de eso. Esa cota supone patrones aleatorios, casi ortogonales entre sí; los QR fallan porque sus solapamientos son altos, no por ser muchos.

![Neuronas inestables de cada QR y estado mezcla](crosstalk.png)

### Estabilidad sin ruido y estado mezcla

- **Ningún QR es un estado estable.** Presentado sin ruido, cada uno cambia en la primera iteración exactamente las neuronas inestables de la tabla y llega a un estado estable en 1 iteración.
- **Ese estado es el estado mezcla sign(ξ₁ + … + ξ₅).** Los cinco QR sin ruido caen en él (5 de 5). El estado mezcla es estable: presentado como consulta, termina con 0 iteraciones. Con cinco patrones la suma nunca es cero, así que está bien definido.
- **Es también adonde cae el barrido de ruido.** 1051 de las 1100 salidas son iguales al estado mezcla y las 49 restantes (todas con p = 0,5) son su inverso; ninguna es otro estado. En el resto del informe, **A = sign(ξ₁ + … + ξ₅)** es el estado mezcla y **−A** su inverso. Como el estado mezcla toma en cada módulo el valor de la mayoría de los cinco QR, es el patrón "promedio" de los menús: se parece más a los QR que coinciden con la mayoría (menú principal, 71 módulos distintos) y menos a los que se apartan de ella (cócteles y postres, 225).

## Barrido de ruido

### Categorías

Las 1100 recuperaciones terminan en un **estado estable** (0 ciclos, 0 recuperaciones cortadas por el tope), y las 1100 son **estado espurio**. Ninguna es recuperación correcta, otro patrón almacenado ni patrón inverso, ni siquiera sin ruido:

| p | Recuperación correcta | Otro patrón almacenado | Patrón inverso | Estado espurio | Ciclo | Tope |
| --- | --- | --- | --- | --- | --- | --- |
| 0 a 0,5 (cada nivel) | 0 | 0 | 0 | 100 | 0 | 0 |

Sobre los 0 ciclos: con W simétrica, la actualización sincrónica siempre termina en un estado estable o en un ciclo de dos estados (Goles y colaboradores, 1985; Bruck, 1990). Es un resultado de fuera del apunte, que no trata la convergencia de la actualización sincrónica. Acota lo que podía pasar (un ciclo sincrónico sólo puede ser de período 2, como el del ejemplo de 4 neuronas de la sección [Energía](#energía)), pero no garantiza que no haya ciclos: que ninguna recuperación de los QR haya terminado en uno es una observación de este barrido.

![Categorías según el nivel de ruido](categorias.png)

### Adónde cae la red

Las 1100 salidas son sólo **dos estados distintos**: el estado mezcla A y su inverso −A (los dos tienen la misma energía, porque H(−s) = H(s)):

| p | A (estado mezcla) | −A |
| --- | --- | --- |
| 0 a 0,45 (cada nivel) | 100 | 0 |
| 0,5 | 51 | 49 |
| Total | 1051 | 49 |

Distancia de Hamming de cada estado al QR original (es la misma para todas las consultas de un mismo QR):

| QR | Hamming a A | Hamming a −A |
| --- | --- | --- |
| Menú principal | 71 | 1298 |
| Vinos | 91 | 1278 |
| Cócteles y otras bebidas | 225 | 1144 |
| Postres | 225 | 1144 |
| Especiales del día | 103 | 1266 |

![Original, consulta y salida de la red](ejemplos.png)

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

- **El resultado no depende del ruido.** Con p = 0 la consulta es el QR original y aun así la red se mueve en 1 iteración a A: la red no puede devolver un QR que no es un estado estable.
- **Todas las consultas van al mismo atractor.** Para p ≤ 0,45 las 1000 recuperaciones llegan al estado A, salga la consulta del QR que salga. La red no distingue entre los cinco menús: siempre responde lo mismo. A queda más cerca del menú principal (71 módulos distintos, 5,2 % de 1369) que de cócteles y postres (225 módulos, 16,4 %).
- **El ruido sólo cambia el camino.** Al subir p, más consultas necesitan 2 iteraciones en lugar de 1, pero el destino es el mismo.
- **Con p = 0,5 la consulta ya no tiene información del original.** Cada módulo se invierte con probabilidad 1/2, así que la consulta es independiente del QR y se reparte casi a la mitad entre A (51) y −A (49). −A es el inverso del estado mezcla, no de un patrón almacenado, así que también es un estado espurio: es estable, porque sign(W·(−s)) = −sign(W·s).

## Energía

La energía de un estado S es H = −½·Σᵢⱼ wᵢⱼ SᵢSⱼ = −½·sᵀWs (fórmula del apunte; verificada en el ejemplo del apunte: H(ξ¹) = H(ξ²) = −3). Se registra para el patrón de consulta y después de cada iteración.

### Curvas de ejemplo

Energía por iteración de la consulta con semilla 0 del QR del menú principal (la línea punteada es la energía del QR original, −2077,45):

| p | Iteración 0 (consulta) | Iteración 1 | Iteración 2 |
| --- | --- | --- | --- |
| 0 | −2077,45 | −2172,88 | |
| 0,2 | −769,80 | −2172,88 | |
| 0,45 | −3,84 | −2130,78 | −2172,88 |
| 0,5 | 0,93 | −2127,67 | −2172,88 |

![Energía por iteración](energia_ejemplos.png)

- **Las 1100 recuperaciones terminan con la misma energía, −2172,88**: la de A y la de −A, que es igual porque H(−s) = H(s).
- **Esa energía es menor que la de cualquier QR original**: −2077,45 (menú principal), −2041,53 (vinos), −1789,71 (cócteles y otras bebidas), −1801,85 (postres) y −1982,47 (especiales del día). El apunte dice que los mínimos locales de H son los patrones almacenados; acá no lo son: como ningún QR es estado estable, alguna neurona tiene campo de signo contrario a su estado, y según la demostración del apunte invertir sólo esa neurona baja la energía. Por eso aun sin ruido (p = 0) la red se va del QR hacia A.
- **Cuanto más ruido, más alta arranca la energía**: la energía media del patrón de consulta sube de −1938,60 con p = 0 a −696,45 con p = 0,2, −18,36 con p = 0,45 y −0,33 con p = 0,5 (`resultados.csv`, columna `energy_initial`). Con p = 0,5 la consulta es aleatoria y no se parece a ningún patrón almacenado, así que su energía queda cerca de 0.

### Subidas de energía con actualización sincrónica

La demostración del apunte de que H nunca sube supone que **cambia una neurona por vez**. Con actualización sincrónica cambian varias juntas y la cuenta tiene un término más. Si C es el conjunto de neuronas que cambian en una iteración y hᵢ = Σⱼ wᵢⱼ Sⱼ es el campo de la neurona i antes de la iteración:

ΔH = −2·Σ_{i∈C} |hᵢ| − 2·Σ_{i, j∈C, i≠j} wᵢⱼ SᵢSⱼ

El primer término es la suma de lo que bajaría la energía si cada neurona cambiara sola (el caso del apunte) y siempre es negativo. El segundo depende de los pesos entre las neuronas que cambian a la vez y puede ser positivo, así que con actualización sincrónica la energía puede subir. Un ejemplo chico lo muestra con tres patrones ortogonales de 4 neuronas: desde sign(ξ₁ + ξ₂ + ξ₃) = (1, 1, 1, −1), que con N = 4 no es estable, con H = 0, la red cambia tres neuronas a la vez y llega a (1, −1, −1, 1), con H = 1,5; después oscila entre ese estado y su opuesto (ciclo).

En los QR no pasa: **0 de las 1100 recuperaciones tuvieron una subida de energía** entre dos iteraciones, en ningún nivel de ruido (`subidas_energia.csv`). La garantía del apunte no aplica a la actualización sincrónica, pero en este barrido la energía bajó en todas las iteraciones.

## Experimento secundario: lector de QR

¿La red de Hopfield ayuda a escanear un QR con ruido, o la corrección de errores del propio QR ya lo resolvía? Para responderlo, el análisis pasa por un lector real (`cv2.QRCodeDetector` de OpenCV, en `analysis/hopfield_qr_reader.py`) el QR original, el patrón de consulta y la salida de la red de cada recuperación. Antes de leerlo, cada patrón se amplía a 8 píxeles por módulo y se le agrega un margen blanco de 4 módulos (la zona de silencio del estándar), así un fallo se debe al contenido y no al formato (`pattern_to_scan_image` en `src/tp4/qr.py`). Requiere `opencv-python` (está en `requirements.txt`).

Por cada lectura se registra si el lector **ubica** el QR (encuentra los patrones localizadores), si lo **decodifica** y si el texto **coincide** con el del original. Salidas, fuera de git:

- `lector_qr.csv`: una fila por recuperación (QR, nivel de ruido, semilla y las tres columnas para la consulta y para la salida).
- `lector_qr_resumen.csv`: los conteos según el nivel de ruido.
- `lector_qr_umbral.csv`: consultas con ruido menor a 0,05 (ver abajo).

Los cinco originales se decodifican, cada uno con la URL de su menú. Los cinco tienen el mismo bloque de información de formato, que indica el nivel de corrección de errores **L**, el más bajo del estándar.

### Barrido de ruido

Cantidad de lecturas (de 100 por nivel) (`lector_qr_resumen.csv`):

| p | Original decodificado | Consulta ubicada | Consulta decodificada | Salida ubicada | Salida decodificada |
| --- | --- | --- | --- | --- | --- |
| 0 | 100 | 100 | 100 | 100 | 0 |
| 0,05 | 100 | 40 | 0 | 100 | 0 |
| 0,1 | 100 | 18 | 0 | 100 | 0 |
| 0,15 | 100 | 10 | 0 | 100 | 0 |
| 0,2 | 100 | 14 | 0 | 100 | 0 |
| 0,25 | 100 | 8 | 0 | 100 | 0 |
| 0,3 | 100 | 13 | 0 | 100 | 0 |
| 0,35 | 100 | 12 | 0 | 100 | 0 |
| 0,4 | 100 | 15 | 0 | 100 | 0 |
| 0,45 | 100 | 12 | 0 | 100 | 0 |
| 0,5 | 100 | 11 | 0 | 100 | 0 |

Toda consulta decodificada tiene el mismo texto que su original. Ninguna de las 1100 salidas se decodifica.

### Umbral del lector

Como ya con p = 0,05 no se lee ninguna consulta, se agregan niveles más bajos sólo para el lector (estas consultas no pasan por la red): p ∈ {0,005; 0,01; 0,015; 0,02; 0,03; 0,04}, 20 semillas por QR y por nivel. El generador se siembra con (índice del QR, 100 + índice del nivel, semilla), para no repetir consultas del barrido.

| p | Módulos invertidos (media) | Consulta ubicada | Consulta decodificada |
| --- | --- | --- | --- |
| 0,005 | 7,4 | 88 | 66 |
| 0,01 | 13,4 | 74 | 38 |
| 0,015 | 20,3 | 76 | 17 |
| 0,02 | 28,3 | 70 | 1 |
| 0,03 | 40,5 | 60 | 0 |
| 0,04 | 55,2 | 45 | 0 |

### Interpretación

- **Hopfield no ayuda al lector: lo empeora.** Con p = 0 el lector decodifica las 100 consultas (son el QR original) y ninguna de las 100 salidas. La red convierte un QR legible en uno ilegible. Con p ≥ 0,05 no se lee ni la consulta ni la salida, así que la red tampoco rescata nada.
- **La salida se ubica pero no se decodifica.** Las salidas son el estado mezcla A y su inverso −A. Un módulo que vale lo mismo en los cinco QR queda igual en A (el signo de la suma de cinco valores iguales es ese valor), y los patrones localizadores son iguales en los cinco, así que A tiene los localizadores de un QR. El detector ubica el 100 % de las salidas, incluidas las 49 que son −A, con los colores invertidos. Pero A difiere del original en 71 a 225 módulos según el QR (y −A en 1144 a 1298), demasiados para la corrección de errores.
- **La corrección de errores del QR sólo resuelve ruido muy bajo.** Con nivel L, el lector decodifica dos de cada tres consultas con unos 7 módulos invertidos (p = 0,005), menos de la mitad con unos 13 (p = 0,01) y ninguna desde unos 40 (p = 0,03). El barrido empieza en p = 0,05, unas 0,05 · 1369 ≈ 68 inversiones esperadas, ya fuera de lo que puede corregir.
- **Una consulta ubicada con mucho ruido no es una lectura.** Para p ≥ 0,1, entre el 8 % y el 18 % de las consultas se ubican, pero ninguna se decodifica: con tantos módulos al azar, el detector a veces arma un candidato a QR que después no puede leer.

En síntesis: ni la red ni el lector resuelven el ruido del barrido. La corrección de errores del QR alcanza para unos pocos módulos, y la red de Hopfield, con estos cinco QR, devuelve siempre un estado que ningún lector puede decodificar.

## Limitaciones

Cada limitación está respaldada por números de las salidas del análisis (el archivo va entre paréntesis). Las tres limitaciones del apunte (capacidad p ≤ 0,15·N, patrones "más o menos" ortogonales, estados espurios) aparecen acá con los valores medidos en los QR.

### 1. La red colapsa al estado mezcla: no recupera ningún QR

- **0 de las 1100 recuperaciones son recuperación correcta**; tampoco termina ninguna en otro patrón almacenado ni en un patrón inverso (`categorias.csv`).
- **Todas terminan en un estado espurio**: 1051 en el estado mezcla A = sign(ξ₁ + … + ξ₅) y 49 en su inverso −A, estas últimas todas con p = 0,5 (salida de `hopfield_qr.py`; `resultados.csv`, columna `final_state`).
- **Pasa aun sin ruido**: con p = 0 las 100 consultas son los QR originales y las 100 terminan en A en 1 iteración, porque ninguno de los cinco QR es un estado estable (`crosstalk.csv`, columna `stable_without_noise`).
- **La salida está lejos del original**: A difiere en 71 módulos del menú principal (5,2 %), en 91 de vinos, en 103 de especiales del día y en 225 de cócteles y otras bebidas y de postres (16,4 %) (`crosstalk.csv`, columna `hamming_to_mixture`).

Para el restaurante, la red responde lo mismo sea cual sea el menú escaneado: no sirve ni siquiera para saber a cuál de los cinco pertenece un QR dañado. Desde p = 0,2 (273,1 módulos invertidos en promedio en la consulta, contra 71 a 225 de A; `resultados.csv`, columna `query_hamming`) la salida queda, en promedio, más cerca del original que la consulta; pero es acercarse a un patrón que no es ningún menú.

### 2. Los QR comparten la mayor parte de sus módulos

- **955 de 1369 módulos (69,8 %) son iguales en los cinco QR** (salida de `hopfield_qr.py`; `solapamientos.png`), y los solapamientos entre pares van de 0,576 a 0,860, entre 21 y 32 veces el solapamiento típico de dos patrones aleatorios, 1/√N ≈ 0,027 (`solapamientos.csv`).
- No es sólo una particularidad de estas imágenes: los patrones localizadores y la información de formato son comunes a QR del mismo tamaño y nivel de corrección, y acá además las URL comparten un prefijo de 72 caracteres. Es esperable que otros QR de un mismo restaurante tengan el mismo problema.
- Con la regla de Hebb, esos módulos compartidos dominan los pesos. Es la condición del apunte de patrones "más o menos" ortogonales, y los QR están lejos de cumplirla.

### 3. Cinco patrones son pocos para la capacidad, y aun así falla

- **5 patrones almacenados frente a una capacidad de 0,15·N = 0,15 × 1369 ≈ 205** (salida de `hopfield_qr.py`): los QR usan el 2,4 % de la cota del apunte.
- Aun así, el crosstalk medio en módulo es de 2,06 a 2,21 según el QR, más del doble del término propio ±1, y deja entre 71 y 225 neuronas inestables por QR (`crosstalk.csv`).
- La cota de capacidad supone patrones aleatorios, casi ortogonales; no dice nada de patrones con solapamientos altos. Para los QR, la limitación que importa es la ortogonalidad, no la cantidad.

### 4. La energía no está garantizada con actualización sincrónica

- La demostración del apunte de que H nunca sube supone que **cambia una neurona por vez**. El algoritmo y el ejemplo del apunte usan la actualización sincrónica, donde ΔH tiene un término más que puede ser positivo; un ejemplo con 4 neuronas lo muestra: la energía sube de 0 a 1,5 y la recuperación termina en un ciclo (sección [Energía](#energía)).
- En los QR, **0 de las 1100 recuperaciones tuvieron una subida de energía** y 0 terminaron en ciclo (`subidas_energia.csv`, `categorias.csv`). Es una observación de este barrido, no una propiedad de la actualización sincrónica: con otros patrones puede no cumplirse.
- El apunte afirma además que los mínimos locales de H son los patrones almacenados. Con estos QR no lo son: sus energías van de −2077,45 a −1789,71, por encima de la del estado mezcla, −2172,88 (salida de `hopfield_qr.py`).

### 5. El ruido aleatorio no es el daño real de un QR

- El ruido del barrido invierte cada módulo de forma independiente, repartido por todo el código: con p = 0,05 ya son 68,5 módulos invertidos en promedio (`resultados.csv`, columna `query_hamming`), incluidos módulos de los patrones localizadores, y el lector ubica sólo 40 de las 100 consultas (`lector_qr_resumen.csv`).
- El daño de un QR impreso (una mancha, un doblez, una esquina rota, un reflejo) suele concentrarse en una zona. La corrección de errores del QR agrupa los módulos en palabras de código y está pensada para ese tipo de daño; según el estándar, el nivel L corrige alrededor del 7 % de las palabras de código. Este análisis no mide daño estructurado (quedó fuera de alcance), así que no dice cómo se comportarían la red ni el lector ante una mancha.
- El nivel de ruido más alto no es el "peor daño": con p = 0,5 la consulta es independiente del QR original, y por eso las consultas se reparten entre A (51) y −A (49) (`resultados.csv`).
- **Actualización 26/10:** el daño estructurado se midió después, con desgaste, manchas irregulares, rayones y una foto gastada, sobre Hebb, desaprendizaje, pseudoinversa y el lector solo. Ver [daño realista](dano_realista_20261026/README.md) y [niveles extremos](dano_realista_extremo_20261026/README.md). El 27/10 se completaron manchas y rayones [hasta 40–50 %](dano_realista_40_20261027/README.md).

### 6. El lector de QR no se beneficia de la red

- Los cinco originales se decodifican; **ninguna de las 1100 salidas de la red se decodifica**, aunque el detector las ubica todas (`lector_qr_resumen.csv`).
- **La red empeora lo que ya se leía**: con p = 0 el lector decodifica las 100 consultas y ninguna de las 100 salidas.
- **La corrección de errores del propio QR alcanza sólo para muy poco ruido aleatorio**: decodifica 66 de 100 consultas con p = 0,005 (7,4 módulos invertidos en promedio), 1 de 100 con p = 0,02 (28,3) y ninguna desde p = 0,03 (40,5) (`lector_qr_umbral.csv`). El barrido de la consigna empieza en p = 0,05, ya fuera de ese rango, así que en ningún nivel la red podía rescatar una lectura que el lector no lograba solo.

### 7. Alcance de las conclusiones

- Los resultados de este informe valen para la regla de Hebb del apunte. La pseudoinversa, guardar menos QR, repartirlos en dos redes o usar una red por zona de la imagen se comparan aparte, en [`mejoras/`](mejoras/README.md), [`variante/`](variante/README.md), [`dos_redes/`](dos_redes/README.md) y [`zonas/`](zonas/README.md).
- El lector es `cv2.QRCodeDetector` de OpenCV; otro lector (por ejemplo, la cámara de un teléfono) puede tener otro umbral. Tampoco se midió.

## Conclusión

Con estos cinco QR y la regla de Hebb, la red de Hopfield no funciona como memoria asociativa: ningún QR es un estado estable, así que, haya o no ruido, la red devuelve siempre el estado mezcla (o su inverso, cuando la consulta ya es aleatoria). El problema no es la cantidad de patrones sino su parecido: los QR de un mismo restaurante comparten casi el 70 % de sus módulos, y el crosstalk que eso genera supera a la señal de cada patrón. Para el restaurante, la red no ayuda a escanear un menú dañado: la corrección de errores del propio QR resuelve unos pocos módulos invertidos, y más allá de eso ni el lector ni la red recuperan el menú.
