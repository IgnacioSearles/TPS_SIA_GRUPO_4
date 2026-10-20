# Controles: ¿el problema es la red o son los QR?

En el [informe principal](../README.md), la red de Hopfield no recupera ningún QR. Antes de culpar a los QR hay que descartar que la red esté mal implementada. Para eso se usa **la misma red, con el mismo tamaño (N = 1369) y las mismas consultas**, pero con patrones generados al azar en lugar de los QR. Si con patrones al azar funciona y con los QR no, el problema está en los QR.

Idea tomada de otra presentación de Hopfield con letras, que comparó letras parecidas contra patrones ortogonales (sus notas no están en el repo).

## Reproducción

Desde `TP4/`:

```
python3 analysis/hopfield_qr_controles.py
```

Tarda unos 10 s y deja los CSV y PNG en esta carpeta (fuera de git). La red es la de `src/tp4/hopfield.py`: regla de Hebb y actualización sincrónica. Una recuperación es correcta cuando la red devuelve exactamente el patrón de origen.

---

## Control 1: cinco patrones al azar contra los cinco QR

**Qué hice.** Generé 5 patrones de 1369 módulos ±1 al azar (semilla 0) y les apliqué el mismo barrido que a los QR: 11 niveles de ruido de 0 a 0,5, con 20 semillas por patrón y nivel.

![Control 1](control1_azar_vs_qr.png)

**Resultado** (`control1_resumen.csv`):

| | Solapamiento máximo entre dos patrones | Correctas con p de 0 a 0,4 | Con p = 0,45 | Con p = 0,5 |
|---|---|---|---|---|
| 5 patrones al azar | 0,061 | **100 %** en cada nivel | 82 % | 10 % |
| 5 QR | 0,860 | **0 %** | 0 % | 0 % |

**Qué significa.** Con la misma red, el mismo tamaño y el mismo tipo de ruido, los patrones al azar se recuperan perfecto hasta con 40 % de módulos invertidos. **La red funciona: el problema son los QR.** Con p = 0,5 la consulta ya es ruido puro, así que ahí no se puede recuperar nada.

## Control 2: ¿cuánto parecido aguanta la red?

**Qué hice.** Generé 5 patrones al azar, pero obligando a que una fracción de los módulos valga lo mismo en los cinco, como pasa con los QR. Moví esa fracción de 0 % a 90 %. Para cada fracción armé 20 juegos de patrones distintos y consulté cada patrón con ruido 0; 0,1 y 0,25.

![Control 2](control2_compartidos.png)

**Resultado** (`control2_resumen.csv`), fracción de recuperaciones correctas:

| Módulos compartidos | p = 0 | p = 0,1 | p = 0,25 |
|---|---|---|---|
| 0 % a 10 % | 1,00 | 1,00 | 1,00 |
| 20 % | 1,00 | 1,00 | 0,96 |
| 22 % | 1,00 | 0,97 | 0,81 |
| 24 % | 0,72 | 0,59 | 0,47 |
| 26 % | 0,15 | 0,13 | 0,04 |
| 28 % o más | **0** | **0** | **0** |
| QR del restaurante: **69,8 %** | | | |

**Qué significa.** La red no se va degradando de a poco: **se corta de golpe cerca del 25 %**. Con más de un cuarto de los módulos compartidos, ya no recupera nada, ni siquiera sin ruido. Los QR comparten casi el 70 %, casi tres veces ese límite.

**Por qué justo 25 %.** Si los 5 patrones comparten una fracción f de sus módulos, cada par se solapa más o menos en f. Ahora tomemos un módulo **no compartido** en el que los otros 4 patrones valen lo contrario que el patrón que queremos recuperar. Ahí:

- la señal del patrón propio vale **1**,
- el crosstalk de los otros 4 empuja en contra con **4·f**.

Si 4·f > 1, es decir **f > 1/4 = 25 %**, el crosstalk gana y ese módulo se da vuelta. Esos módulos no son raros: son, en promedio, 1 de cada 16 de los no compartidos (que los otros 4 valgan lo contrario tiene probabilidad (1/2)⁴). Así que ningún patrón queda estable. En general, con k patrones el umbral es **1/(k − 1)**.

Este umbral es coherente con lo que medimos en los QR, aunque ahí lo que no se comparte no es al azar:
- Con 2 patrones, el umbral es 100 %: siempre funciona. **Los QR entran de a 2.**
- Con 3, el umbral es 50 %, y los QR comparten casi el 70 %. **De a 3 los QR completos fallan.**

## Control 3: ¿cuántos patrones entran?

**Qué hice.** Guardé cada vez más patrones (de 1 a 300) y medí qué fracción de consultas con 10 % de ruido vuelve exactamente a su patrón, en tres casos:
- **patrones al azar** de 1369 módulos (100 consultas por cantidad),
- **subconjuntos de los QR completos** (todos los subconjuntos de k QR, con las 20 semillas de ruido del barrido principal),
- **subconjuntos de los QR sin módulos compartidos** (la [variante](../variante/README.md), 414 módulos).

![Control 3](control3_capacidad.png)

**Resultado** (`control3_resumen.csv`), fracción correcta con p = 0,1:

| Patrones guardados | Al azar | QR completos | QR sin compartidos |
|---|---|---|---|
| 2 | 1,00 | 1,00 | 1,00 |
| 3 | 1,00 | **0** | 0,99 |
| 4 | | 0 | 0,50 |
| 5 | 1,00 | 0 | **0** |
| 50 | 1,00 | | |
| 100 | 0,94 | | |
| 150 | 0,10 | | |
| 200 | 0,02 | | |

**Qué significa.**
- Con patrones al azar, la red guarda unos **100 patrones** antes de empezar a fallar. Es menos que la cota del apunte (0,15·N ≈ 205) porque acá se exige más: volver al patrón exacto, sin un solo módulo mal, partiendo de una consulta con 10 % de ruido.
- **Con los QR, la misma red falla con 3.** Sacando los módulos compartidos, aguanta 3 y falla con 4 o 5.
- Es la diferencia entre **100 y 2**, con la misma red y el mismo N. La red tiene lugar de sobra; lo que no entra es el parecido entre los QR.

---

## Conclusión

1. **La red está bien:** con patrones al azar recupera todo hasta con 40 % de ruido (control 1).
2. **La causa es el parecido:** con patrones al azar que comparten más del 25 % de sus módulos, la red deja de recuperar de golpe, y hay una cuenta simple que explica ese número. Los QR comparten el 69,8 % (control 2).
3. **No es la cantidad:** la red guarda unos 100 patrones al azar, pero sólo 2 QR, o 3 sin los módulos compartidos (control 3).

## Archivos

| Archivo | Contenido |
|---|---|
| `control1_resultados.csv`, `control1_resumen.csv` | Una fila por recuperación y fracción correcta por nivel de ruido (al azar contra QR) |
| `control2_resultados.csv`, `control2_resumen.csv` | Ídem, según la fracción de módulos compartidos y el ruido |
| `control3_resultados.csv`, `control3_resumen.csv` | Ídem, según la cantidad de patrones guardados |
