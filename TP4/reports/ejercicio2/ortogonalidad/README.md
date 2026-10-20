# Experimentos de ortogonalidad

Son los mismos experimentos que hizo otro grupo con letras de 5×5 (sus notas de presentación, que no están en el repo), aplicados a los QR. Ellos concluyen que **además de respetar la capacidad, es fundamental que los patrones sean ortogonales**. Acá se ve lo mismo con los QR, y más marcado.

| Experimento de las notas (letras) | Acá (QR) |
|---|---|
| Ranking de los conjuntos de 4 letras por ortogonalidad (GRTV el mejor, HMNW el peor) | 1. Ranking de todos los subconjuntos de 2, 3 y 4 QR |
| Crosstalk en cada neurona | 2. Mapa de crosstalk: QR contra patrones al azar |
| Consulta con 15 % y 40 % de ruido, paso a paso, con energía | 3. Lo mismo con los QR |
| Tasa de recuperación por letra; conjunto más contra menos ortogonal | 4. Tasa por QR: terna más contra menos ortogonal |
| Más neuronas (letras de 15×15) | 5. QR al doble de resolución |
| 26 patrones ortogonales de 16×16 | Ya está: [control 1](../controles/README.md) |

## Reproducción

Desde `TP4/`:

```
python3 analysis/hopfield_qr_ortogonalidad.py
```

Tarda unos 30 s y deja los CSV y PNG en esta carpeta (fuera de git). Regla de Hebb y actualización sincrónica. Se usan las mismas consultas que el barrido principal (11 niveles de ruido × 20 semillas por QR). Donde dice **"sin módulos compartidos"** es la [variante](../variante/README.md): la red trabaja sólo con los 414 módulos que difieren y al final se reponen los otros 955.

---

## Experimento 1: ranking por ortogonalidad

**Qué hice.** Como en las notas, tomé **todos** los subconjuntos posibles de 2, 3 y 4 QR y, para cada uno, calculé el solapamiento |m| entre cada par: el máximo y el promedio. Los ordené por el máximo y, a igualdad, por el promedio. Después guardé cada subconjunto en la red y medí qué fracción de las consultas (ruido de 0 a 0,3) vuelve exactamente a su QR. Lo hice con los QR completos y sin los módulos compartidos.

![Ranking](1_ranking.png)

**Resultado** (`ranking_subconjuntos.csv`):

| Subconjunto | Máx. \|m\| | Prom. \|m\| | Correctas (p de 0 a 0,3) |
|---|---|---|---|
| **QR completos, cualquier par** | 0,58 a 0,86 | | **100 %** |
| **QR completos, cualquier terna o grupo de 4** | 0,76 a 0,86 | | **0 %** |
| Sin compartidos, terna más ortogonal: cócteles · postres · especiales | 0,34 | 0,27 | 99,8 % |
| Sin compartidos, terna menos ortogonal: menú · vinos · postres | 0,54 | 0,43 | 89,8 % |
| Sin compartidos, grupo de 4 más ortogonal: menú · cócteles · postres · especiales | 0,34 | 0,29 | 87,1 % |
| Sin compartidos, grupo de 4 menos ortogonal: menú · vinos · cócteles · especiales | 0,54 | 0,34 | 22,9 % |

**Qué significa.**
- **Entre los subconjuntos del mismo tamaño, los más ortogonales recuperan más.** Sin compartidos, las ternas con máx. |m| de 0,34 o 0,40 dan entre 99,8 % y 100 %, y las de 0,54 entre 89,8 % y 98,1 %. Con 4 QR: 87,1 % (0,34), 60,4 % (0,40) y entre 22,9 % y 36,1 % (0,54). No es un orden perfecto dentro de cada grupo, pero la tendencia es clara. Es la misma conclusión que GRTV contra HMNW.
- **El par menú principal–vinos aparece siempre en los peores.** Es el HMNW de los QR.
- **Los pares son la excepción:** cualquier par de QR funciona, aun con solapamiento 0,86. Encaja con el umbral del [control 2](../controles/README.md): con 2 patrones el límite es 1/(2 − 1) = 100 %.
- **Diferencia con las letras:** ellos pudieron elegir 4 de 26 letras. Nosotros tenemos que guardar los 5 QR del restaurante, así que elegir el subconjunto más ortogonal no resuelve el problema; sirve para entenderlo.

## Experimento 2: crosstalk en cada neurona

**Qué hice.** Con los 5 guardados, calculé para cada QR el crosstalk de cada neurona, alineado con el QR: ξᵢ·Cᵢ.
- Positivo (rojo) = los otros patrones empujan **a favor**.
- Negativo (azul) = empujan **en contra**.
- Por debajo de −1 le gana a la señal propia (+1) y la neurona se da vuelta.

Comparé los 5 QR con los 5 patrones al azar del control 1.

![Crosstalk](2_crosstalk.png)

**Resultado:**

| | \|crosstalk\| medio | Neuronas inestables |
|---|---|---|
| 5 QR | 2,06 a 2,21 | 71 a 225 por QR |
| 5 patrones al azar | 0,03 a 0,06 | 0 |

**Qué significa.** En las notas, el crosstalk del conjunto más ortogonal quedaba "cerca de 0". Los patrones al azar se ven así: gris parejo. Los QR, en cambio, muestran dos zonas:
- **La mitad derecha (los módulos compartidos) es roja fuerte:** ahí los otros 4 QR coinciden y empujan a favor con casi 3. No hace daño, pero tampoco sirve para distinguir un QR de otro.
- **La franja de la izquierda (donde difieren) tiene azul:** ahí el QR está en minoría frente a los otros y el crosstalk empuja en contra. Los módulos que se dan vuelta están todos ahí.

## Experimento 3: paso a paso con 15 % y 40 % de ruido

**Qué hice.** Como en las notas (la G con 15 % y con 40 % de ruido), seguí una consulta paso a paso, mostrando el estado en cada iteración y la energía. Usé el QR de cócteles en dos casos:
- **3a:** guardado junto a los otros 4 QR completos (el caso del enunciado).
- **3b:** guardado sin módulos compartidos, en la terna más ortogonal (cócteles, postres y especiales). Para los pasos se reponen los módulos compartidos.

![Paso a paso con los cinco QR completos](3a_paso_a_paso_completo.png)

![Paso a paso con la terna más ortogonal](3b_paso_a_paso_variante.png)

**Resultado:**
- **3a (cinco QR completos):** con 15 % (178 módulos mal) y con 40 % (554 mal), la red llega en 1 paso al **mismo estado**, a 225 módulos del original: el estado mezcla. La energía baja, pero hacia la mezcla.
- **3b, 15 % y 40 % con semilla 0:** recupera el QR **exacto en 1 paso** (de 178 y de 554 módulos mal a 0).
- **3b, 40 % con semilla 6** (la primera semilla que falla): en 2 pasos cae en un **estado espurio** a 76 módulos del original.

**Qué significa.** En las notas, la energía baja y los mínimos "tienden a ser" los patrones guardados. Acá se ve que no siempre:
- En 3a, el mínimo al que se llega es la mezcla (−2172,88), más baja que la de cualquier QR.
- En 3b, el estado espurio termina en **−251,93**, más abajo que los tres QR guardados (de −245,55 a −229,32). La red baja de energía, pero eso no garantiza que termine en un patrón guardado.

## Experimento 4: tasa de recuperación por QR

**Qué hice.** Como en las notas (tasa por letra según el ruido, conjunto más ortogonal contra el menos ortogonal), medí la tasa de recuperación de cada QR en la terna más ortogonal y en la menos ortogonal, sin módulos compartidos.

![Tasa por QR](4_tasa_por_qr.png)

**Resultado** (`resultados_subconjuntos.csv`), fracción correcta:

| QR | p = 0,05 | p = 0,1 | p = 0,25 | p = 0,4 |
|---|---|---|---|---|
| **Terna más ortogonal (máx. \|m\| = 0,34)** | | | | |
| cócteles y otras bebidas | 1,00 | 1,00 | 1,00 | 0,90 |
| postres | 1,00 | 1,00 | 1,00 | 1,00 |
| especiales del día | 1,00 | 1,00 | 1,00 | 0,80 |
| **Terna menos ortogonal (máx. \|m\| = 0,54)** | | | | |
| menú principal | 1,00 | 0,95 | 0,75 | 0,70 |
| vinos | 0,90 | 0,85 | 0,80 | 0,65 |
| postres | 1,00 | 1,00 | 0,90 | 0,75 |

**Qué significa.** En la terna más ortogonal, los tres QR se recuperan siempre hasta p = 0,25. En la menos ortogonal, menú principal y vinos (el par más parecido) empiezan a fallar con muy poco ruido. Postres está en las dos ternas, y en la peor también rinde menos: su resultado depende de con quién se lo guarde.

## Experimento 5: ¿y si agregamos neuronas?

**Qué hice.** Las notas pasaron las letras a 15×15 para tener más neuronas y más capacidad, y aun así la red falló. Acá dupliqué la resolución de los QR: cada módulo pasa a ser un bloque de 2×2 neuronas iguales, así que N va de 1369 a 5476 y la capacidad del apunte (0,15·N) de 205 a 821. Corrí 400 consultas (ruido 0; 0,1; 0,2 y 0,3; 20 semillas por QR).

**Resultado** (`resolucion.csv`):

| | N | 0,15·N | Máx. \|m\| | QR estables | Correctas | Caen en la mezcla |
|---|---|---|---|---|---|---|
| QR originales | 1369 | 205 | 0,86 | 0 de 5 | 0 % | 100 % |
| QR al doble de resolución | 5476 | 821 | **0,86** | 0 de 5 | 0 % | 100 % |

**Qué significa.** Cuadruplicar las neuronas no cambia nada, porque **el solapamiento sigue siendo el mismo**: duplicar cada módulo no hace a los QR menos parecidos. Las notas llegan a lo mismo con las letras de 15×15: más neuronas no alcanza si los patrones se parecen.

---

## Conclusión

Llegamos a la misma conclusión que las notas, con más evidencia:

1. **La ortogonalidad anticipa el resultado.** Entre subconjuntos del mismo tamaño, los más ortogonales recuperan más; los que incluyen menú principal y vinos son los peores (experimentos 1 y 4).
2. **El crosstalk muestra dónde está el problema:** en la franja donde los QR difieren, los otros QR empujan en contra (experimento 2).
3. **La energía baja, pero no necesariamente hacia un QR:** la mezcla y otros estados espurios tienen menos energía que los QR guardados (experimento 3).
4. **Más neuronas no ayudan** si el parecido no cambia (experimento 5).

## Archivos

| Archivo | Contenido |
|---|---|
| `ranking_subconjuntos.csv` | Una fila por variante y subconjunto: solapamiento máximo y medio, y fracción correcta con p ≤ 0,3 |
| `resultados_subconjuntos.csv` | Una fila por recuperación (variante, subconjunto, QR, ruido, categoría) |
| `resolucion.csv` | QR originales contra el doble de resolución |
