# Intentos de mejora: pseudoinversa, dos redes y daño por zonas

El [informe principal](../README.md) muestra que, con la regla de Hebb, la red no recupera ningún QR. Acá se prueban dos formas de mejorarla y un daño más parecido al del enunciado ("una de sus esquinas está rota").

**La regla de la pseudoinversa no está en el apunte.** Se agrega como comparación, a pedido de Ivo. El resultado principal del ejercicio sigue siendo el de Hebb.

## Los métodos

Cada método recibe un QR dañado y devuelve un QR completo:

| Método | Qué hace |
|---|---|
| **Hebb, 5 QR completos** | La red del informe principal: W = (1/N)·Σ ξξᵀ |
| **Pseudoinversa, 5 QR completos** | W = Ξᵀ(ΞΞᵀ)⁻¹Ξ. En vez de sumar los patrones, calcula W para que cada QR guardado sea **exactamente estable**, aunque se parezcan. Con patrones ortogonales da lo mismo que Hebb (verificado con un ejemplo chico) |
| **Pseudoinversa sin módulos compartidos** | Lo mismo, pero sólo con los 414 módulos que difieren; los otros 955 se reponen al final |
| **Dos redes de Hebb sin compartidos** | Los 5 QR repartidos en dos redes: {menú principal, cócteles, postres} y {vinos, especiales}. Es el reparto que separa a menú y vinos y deja los solapamientos más bajos dentro de cada red. La consulta pasa por las dos; se queda con la salida que sea un QR guardado (si las dos lo son, la más cercana a la consulta) |
| **Sin red: QR original más cercano** | No es Hopfield: compara el QR dañado con los 5 originales y devuelve el más parecido. Sirve de techo: es lo mejor que se puede hacer sabiendo los 5 originales |

También se mide si **el lector de QR (OpenCV) lee la consulta dañada directamente**, sin pasar por ninguna red.

## Reproducción

Desde `TP4/`:

```
python3 analysis/hopfield_qr_mejoras.py
```

Tarda unos 25 s y deja los CSV y PNG en esta carpeta (fuera de git). Actualización sincrónica.

---

## Antes de empezar: la energía con cada regla

`energia.csv`:

| | Hebb | Pseudoinversa |
|---|---|---|
| Energía de cada QR | −2077,45 a −1789,71 | **−682,00 los cinco** |
| ¿Los QR son estables? | ninguno | **los cinco** |
| Energía del estado mezcla | **−2172,88** (la más baja) | −607,89 (más alta que los QR) |
| ¿La mezcla es estable? | sí | no |

**Con la pseudoinversa se da vuelta todo.** Los QR pasan a ser los mínimos de energía y la mezcla deja de ser un atractor. Es exactamente lo que el apunte espera de una red de Hopfield y que con Hebb no pasaba.

## Experimento 1: ruido aleatorio

**Qué hice.** Las mismas 1100 consultas del informe principal (11 niveles de ruido × 20 semillas × 5 QR), pasadas por cada método.

![Ruido aleatorio](1_ruido.png)

**Resultado** (`resumen.csv`), fracción que devuelve exactamente el QR original:

| p | Hebb | Pseudoinversa | Pseudoinv. sin compartidos | Dos redes | Sin red: más cercano |
|---|---|---|---|---|---|
| 0 a 0,35 | **0** | **1,00** | 1,00 | 1,00 | 1,00 |
| 0,40 | 0 | 0,96 | 0,85 | 0,90 | 0,98 |
| 0,45 | 0 | 0,59 | 0,40 | 0,65 | 0,71 |
| 0,50 | 0 | 0,06 | 0,06 | 0,17 | 0,22 |

El lector lee la salida exactamente cuando es correcta, así que su gráfico es igual. Leyendo la consulta directamente, el lector sólo decodifica con p = 0.

**Qué significa.**
- La pseudoinversa y las dos redes **recuperan los 5 QR hasta con 35 % de módulos invertidos**, y el lector los lee.
- Quedan casi pegadas al techo (el QR más cercano sin red). Con p = 0,5 la consulta es ruido puro y ningún método puede saber de qué QR vino.
- **Las dos redes son la mejor opción dentro de Hebb, la regla vista en clase:** rinden casi como la pseudoinversa.

## Experimento 2: una esquina rota

**Qué hice.** Como dice el enunciado, rompí una esquina: un bloque cuadrado en blanco, de 3×3 a 29×29 módulos (de 1 % a 61 % del QR), en cada una de las 4 esquinas de cada QR (20 consultas por tamaño).

![Esquina rota](2_esquina.png)

**Resultado** (`resumen.csv`):

| Bloque | Hebb | Pseudoinversa | Pseudoinv. sin compartidos | Dos redes | Más cercano | Lector sin red |
|---|---|---|---|---|---|---|
| 3×3 a 9×9 (1 % a 6 %) | 0 | 1,00 | 1,00 | 1,00 | 1,00 | **0,25** |
| 11×11 a 17×17 (9 % a 21 %) | 0 | 1,00 | 1,00 | 1,00 | 1,00 | **0** |
| 21×21 (32 %) | 0 | 1,00 | 1,00 | 0,90 | 1,00 | 0 |
| 25×25 (46 %) | 0 | 1,00 | 0,85 | 0,90 | 1,00 | 0 |
| 29×29 (61 %) | 0 | 0,90 | 0,75 | 0,75 | 0,90 | 0 |

**Qué significa.**
- **Sin red, el lector sólo aguanta la esquina inferior derecha**, la única sin cuadrado de posición, y sólo hasta 9×9. Si se rompe una esquina con cuadrado de posición, el lector ni siquiera encuentra el QR.
- **Con la pseudoinversa o con las dos redes, se recupera el QR exacto aun con un 21 % del código arrancado**: la red reconstruye el cuadrado de posición que faltaba y el lector lo lee.
- **Este es el caso de uso del enunciado, y acá la red sí ayuda al lector.**

![Ejemplo con una esquina rota](4_ejemplo_esquina.png)

## Experimento 3: una mancha

**Qué hice.** Un bloque negro (una mancha), de 3×3 a 29×29 módulos, en un lugar al azar (20 posiciones por QR y tamaño).

![Mancha](3_mancha.png)

**Resultado** (`resumen.csv`):

| Mancha | Hebb | Pseudoinversa | Pseudoinv. sin compartidos | Dos redes | Más cercano | Lector sin red |
|---|---|---|---|---|---|---|
| 3×3 | 0 | 1,00 | 1,00 | 1,00 | 1,00 | 0,89 |
| 5×5 | 0 | 1,00 | 1,00 | 1,00 | 1,00 | 0,77 |
| 7×7 | 0 | 1,00 | 1,00 | 1,00 | 1,00 | 0,69 |
| 9×9 | 0 | 1,00 | 1,00 | 1,00 | 1,00 | 0,37 |
| 11×11 a 21×21 | 0 | 1,00 | 1,00 | 1,00 | 1,00 | 0 |
| 25×25 | 0 | 0,99 | 1,00 | 1,00 | 1,00 | 0 |
| 29×29 | 0 | 0,94 | 0,94 | 0,89 | 0,96 | 0 |

**Qué significa.** Sin red, el lector tolera manchas chicas porque la corrección de errores del QR está pensada para daño por zonas, pero deja de leer desde 11×11. Con la pseudoinversa o las dos redes, se recupera el QR hasta con manchas de 21×21 (32 % del código).

---

## Conclusión

1. **El problema de Hebb tiene arreglo, de dos formas:**
   - cambiando la regla por la pseudoinversa, que no está en el apunte;
   - o, sin salir de Hebb, repartiendo los QR en **dos redes** para que en cada una haya pocos QR y poco parecidos.

   Las dos recuperan los 5 QR hasta con 35 % de ruido, o con un 21 % del QR arrancado.
2. **Con daño por zonas, como en el enunciado, la red sí le sirve al lector:** el lector solo no lee una esquina rota con cuadrado de posición, y la red se la reconstruye.
3. **Limitación honesta:** ninguna red le gana a "comparar con los 5 originales y quedarse con el más parecido". Si los originales se conocen, la red no hace falta para elegir cuál es; lo que sí aporta es devolver la imagen completa para el lector.

## Archivos

| Archivo | Contenido |
|---|---|
| `energia.csv` | Energía y estabilidad de cada QR y del estado mezcla, con cada regla |
| `resultados.csv` | Una fila por consulta y método (daño, nivel, QR, semilla, correcto, el lector lee) |
| `resumen.csv` | Fracción correcta y fracción leída por daño, método y nivel |
