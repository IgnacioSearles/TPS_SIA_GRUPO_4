# Ejercicio 2 (Hopfield con QR): problemas encontrados

Material para armar la presentación. Cada problema tiene el **mensaje** de la diapo en una línea, los números
que lo respaldan, de dónde salen y qué gráfico usar. Se actualiza a medida que aparecen problemas nuevos.

Fuentes:
- Informe principal: `reports/ejercicio2/README.md` (salidas de `python3 analysis/hopfield_qr.py`).
- Controles: `reports/ejercicio2/controles/README.md` (salidas de `python3 analysis/hopfield_qr_controles.py`).
- Variante: `reports/ejercicio2/variante/README.md` (salidas de `python3 analysis/hopfield_qr_variante.py`).

Se usa sólo la actualización sincrónica del apunte.
**Nombres en los gráficos:** cada QR aparece con la palabra en inglés que trae su URL: **food** = menú principal, **wine** = vinos, **beverages** = cócteles y otras bebidas, **dessert** = postres, **specials** = especiales del día. La asincrónica se implementó y se quitó porque el enunciado no la pide.

Rutas de los gráficos relativas a `TP4/`.

**Hilo de la presentación:** la red no recupera ningún QR → ¿está mal la red? no: con patrones al azar funciona →
no es por cantidad sino porque los QR se parecen → con más de 25 % compartido la red se corta → por qué se
parecen → qué probamos (sacar lo común) → mejora pero no alcanza para 5 → por qué no alcanza → qué se podría hacer.

---

## Problema 1. La red no recupera ningún QR

**Mensaje:** pase lo que pase, la red devuelve siempre la misma imagen, que no es ningún menú.

- **0 de 1100** recuperaciones correctas.
- 1051 terminan en el **estado mezcla** A = sign(ξ₁ + … + ξ₅): en cada módulo, el color de la mayoría de los 5 QR. Las otras 49 terminan en su inverso −A, todas con p = 0,5.
- **Pasa incluso sin ruido:** un QR perfecto cae a A en 1 iteración. Ninguno de los 5 QR es estable.
- A queda a entre 71 (menú principal) y 225 (cócteles, postres) módulos del QR original.

**Gráficos:** `reports/ejercicio2/categorias.png`, `reports/ejercicio2/ejemplos.png`

## Control: ¿está mal la red?

**Mensaje:** con la misma red y el mismo tamaño, 5 patrones al azar se recuperan perfecto; los 5 QR, nunca.

- 5 patrones al azar de 1369 módulos (solapamiento máximo 0,06): **100 % correctas** con ruido de 0 a 0,4; 82 % con 0,45.
- 5 QR (solapamiento máximo 0,86): **0 %** en todos los niveles.
- Descarta que sea un error de implementación. Es la primera pregunta que va a hacer un docente.

**Gráfico:** `reports/ejercicio2/controles/control1_azar_vs_qr.png`

## Problema 2. No es un problema de cantidad

**Mensaje:** la red podría guardar unos 200 patrones y sólo le pedimos 5.

- Capacidad del apunte: 0,15·N = 0,15 × 1369 ≈ **205** patrones. Con 5 usamos el **2,4 %**.
- Esa cota supone patrones al azar, casi ortogonales. Los QR no lo son.
- **Medido (control 3):** la misma red guarda unos **100 patrones al azar** (94 % correctas con 10 % de ruido), pero **sólo 2 QR** (3 sin los módulos compartidos).

**Gráfico:** `reports/ejercicio2/controles/control3_capacidad.png`

## Problema 3. Los 5 QR se parecen demasiado

**Mensaje:** para el ojo son 5 QR distintos; para la red son 5 versiones del mismo.

- **955 de 1369 módulos (69,8 %) son idénticos** en los 5 QR.
- Solapamiento entre pares: **0,58 a 0,86**. Dos patrones al azar darían ≈ 0,03.
- **Interferencia (crosstalk)** de los otros QR: **2,1 a 2,2** contra una señal propia de 1. La interferencia le gana a la señal, y por eso todo cae a la mezcla.

**Gráficos:** `reports/ejercicio2/solapamientos.png` (el panel derecho muestra los módulos iguales), `reports/ejercicio2/crosstalk.png`

### Cuánto parecido aguanta la red: el umbral del 25 %

**Mensaje:** con 5 patrones, si comparten más del 25 % de sus módulos, la red deja de funcionar de golpe. Los QR comparten el 70 %.

- **Medido (control 2):** 5 patrones al azar que comparten una fracción f de sus módulos. Con f ≤ 20 % se recupera todo; con 24 % el 72 % sin ruido; con 26 % el 15 %; con 28 % o más, **nada**, ni siquiera sin ruido.
- **Por qué 25 %:** en un módulo no compartido en el que los otros 4 patrones valen lo contrario, la señal propia vale 1 y el crosstalk empuja en contra con 4·f. Si f > 1/4, gana el crosstalk. Esos módulos son 1 de cada 16 de los no compartidos, así que ningún patrón queda estable. Con k patrones el umbral es 1/(k − 1).
- **Es coherente con los QR:** con 2 patrones el umbral es 100 %, y los QR entran de a 2; con 3 es 50 %, y los QR (70 %) fallan de a 3.

**Gráfico:** `reports/ejercicio2/controles/control2_compartidos.png` (el más fuerte para explicar el problema)

### Explicación de fondo: ortogonalidad

**Mensaje:** Hebb funciona con patrones ortogonales; nuestros QR coinciden en el 79 % al 93 % de sus módulos, cuando lo ideal sería 50 %.

- **Solapamiento** m = (módulos iguales − módulos distintos) / N: 1 = idénticos, **0 = ortogonales** (50 % iguales, como al azar), −1 = opuestos. Porcentaje de módulos iguales = (1 + m) / 2.
- **Por qué importa:** el campo de cada neurona es señal (1) + crosstalk, y el crosstalk es Σ m^μν · ξ^μ sobre los otros QR. Con m ≈ 0 el crosstalk es casi 0 y la señal gana; con m alto el crosstalk le gana a la señal. La cota 0,15·N del apunte supone patrones ortogonales.
- **Nuestros QR:**

  | | m | Módulos iguales |
  |---|---|---|
  | Dos patrones al azar (N = 1369) | ≈ 0,03 | ≈ 50 % |
  | Par menos parecido | 0,58 | 79 % |
  | Par más parecido (menú–vinos) | 0,86 | 93 % |

- **Después de sacar los compartidos (N = 414):** menú–vinos 0,54 (77 % iguales); vinos–cócteles −0,40 (30 % iguales). Tampoco es ortogonal: un parecido "al revés" interfiere igual, porque el crosstalk usa m con su signo. Con 5 QR, sacar lo común obliga a un m medio de −0,09.
- **Cómo se conseguiría:** no depende de la red sino de los patrones. Por ejemplo, generando cada QR con una máscara distinta (ver soluciones).

### Experimentos de ortogonalidad (los mismos que las notas de otro grupo con letras de 5×5)

Fuente: `reports/ejercicio2/ortogonalidad/README.md`.

- **Ranking de subconjuntos** (`1_ranking.png`): cualquier par de QR funciona (umbral 100 %); cualquier terna o grupo de 4 completo, 0 %. Sin compartidos, la terna más ortogonal (cócteles · postres · especiales, máx. |m| 0,34) recupera el 99,8 % y la menos ortogonal (menú · vinos · postres, 0,54) el 89,8 %. **Menú + vinos es nuestro "HMNW".**
- **Mapa de crosstalk** (`2_crosstalk.png`, muy visual): en los QR, la mitad derecha es roja (los compartidos empujan a favor) y la franja izquierda tiene azul (en contra; ahí están todos los módulos que se dan vuelta). Los patrones al azar se ven grises: |C| 0,03–0,06 contra 2,06–2,21.
- **Paso a paso con 15 % y 40 %** (`3a_…`, `3b_…`): con los 5 completos, las dos consultas caen en 1 paso a la mezcla. En la terna sin compartidos recupera el QR exacto en 1 paso aun con 554 módulos mal. Un caso con 40 % que falla termina en un espurio con energía −251,93, **más baja que los 3 QR guardados** (−245,55 a −229,32): la energía baja, pero no necesariamente hacia un QR.
- **Tasa por QR** (`4_tasa_por_qr.png`): en la terna más ortogonal los tres dan 100 % hasta p = 0,25; en la menos ortogonal vinos ya falla con p = 0,05 (90 %).
- **Más neuronas** (`resolucion.csv`): QR al doble de resolución (N = 5476, 0,15·N = 821) → mismo solapamiento 0,86, 0 % correctas, 100 % a la mezcla. Igual que las letras de 15×15 de las notas.

## Problema 4. Por qué se parecen: así se arma un QR

**Mensaje:** el parecido no es casualidad; viene del formato QR y de las URL.

- **Partes fijas del formato:** los 3 cuadrados de las esquinas, la línea punteada (timing), el cuadradito de alineación y la información de formato (mismo nivel de corrección y misma máscara en los 5).
- **URL casi iguales:** las 5 comparten los primeros **72 caracteres** (`https://github.com/eugepineiro/sia-tips/blob/feature/tp4-images/TPs/TP4/`). Sólo cambia el nombre del archivo.
- El QR escribe los datos desde la esquina inferior derecha hacia la izquierda, así que el prefijo común queda en la **mitad derecha**: 410 de los 414 módulos que difieren están en las columnas 1 a 19.
- **Conclusión para el restaurante:** cualquier conjunto de QR de un mismo sitio web va a tener este problema.

**Gráfico:** `reports/ejercicio2/solapamientos.png` (panel derecho). Se puede agregar un dibujo de las partes de un QR.

## Problema 5. La red empeora al lector de QR

**Mensaje:** la red no sólo no ayuda: arruina QR que se leían bien.

- El lector (OpenCV) lee los 5 originales, pero **0 de 1100** salidas de la red.
- Con p = 0, el lector lee las 100 consultas y ninguna de las salidas.
- La corrección de errores propia del QR (nivel L) aguanta muy poco ruido al azar: lee 66 % con p = 0,005, 1 % con p = 0,02 y 0 % desde p = 0,03.

**Gráfico:** no hay uno propio en el informe principal (los datos están en `lector_qr_resumen.csv` y `lector_qr_umbral.csv`). El experimento 6 de la variante lo muestra mejor.

## Problema 6. La energía: lo que el apunte garantiza y lo que no

**Mensaje:** los QR no son los mínimos de energía; la mezcla tiene menos energía que todos.

- Energía de cada QR: de −2077,45 a −1789,71. Energía del estado mezcla: **−2172,88**, más baja que la de todos. La red baja de energía y por eso termina ahí.
- Con actualización sincrónica (todas las neuronas a la vez) la energía puede subir (un ejemplo con 4 neuronas va de 0 a 1,5). En los QR no subió nunca, pero eso es lo que se observó en este barrido, no una garantía.

**Gráfico:** `reports/ejercicio2/energia_ejemplos.png`

## Problema 7. El ruido del experimento no es el daño real de un QR

**Mensaje:** probamos con ruido al azar; un QR real se daña por zonas.

- El barrido invierte módulos sueltos por todo el código. Una mancha, un doblez o un reflejo dañan una zona; **el propio enunciado habla de "una de sus esquinas está rota"**.
- La corrección de errores del QR está pensada para daño por zonas. El daño estructurado no se midió.

**Gráfico:** ninguno (es una limitación del experimento).

---

## Lo que probamos: guardar sólo los módulos que difieren

**Idea:** si 955 módulos son iguales en los 5 QR, no sirven para distinguirlos. Se sacan, la red trabaja sólo con los **414** que difieren y al final se reponen los 955 con su valor conocido. La regla sigue siendo Hebb.

## Problema 8. Mejora, pero no alcanza para los 5 QR

**Mensaje:** sacando lo común, la red pasa de guardar 2 QR a guardar 3; con 5 sigue fallando.

| | Antes (1369 módulos) | Después (414 módulos) |
|---|---|---|
| Solapamiento entre pares | 0,58 a 0,86 | −0,40 a 0,54 |
| Interferencia media (señal = 1) | 2,1 a 2,2 | 0,67 a 0,95 |
| Grupos de 3 QR que la red guarda bien | 0 de 10 | **10 de 10** |
| Grupos de 4 QR que la red guarda bien | 0 de 5 | 1 de 5 |
| Los 5 QR juntos | no | **no** |
| Correctas con 25 % de ruido (3 QR guardados) | 0 % | **95,7 %** |
| El lector lee la salida con 25 % de ruido (3 QR) | 0 % | **95,7 %** |

- Con 3 QR guardados, la red recupera el QR exacto aun con 40 % de módulos dañados (78,7 % de los casos).

**Gráficos:** `reports/ejercicio2/variante/3_capacidad.png` (el más claro), `4_barrido_ruido.png`, `6_lector.png`, `7_ejemplos.png` (el más visual: antes siempre la misma mezcla, después el QR exacto).

## Problema 9. Una trampa al medir la mejora

**Mensaje:** parte de la mejora la hace reponer los módulos conocidos, no la red; hay que separar las dos cosas.

- Al reponer los 955 módulos compartidos, el ruido que había caído en ellos desaparece sin que la red haga nada.
- Por eso se agregó una referencia **sin red**: la consulta con sólo los compartidos repuestos.
- Con 25 % de ruido, reponer deja **103,5** módulos mal; la red los baja a **2,1**. Con 5 % de ruido, el lector lee el 13 % de las consultas repuestas sin red y el 99,7 % de las salidas de la red.
- **La mejora es real**, pero sin esa línea se habría exagerado.

**Gráficos:** `reports/ejercicio2/variante/4_barrido_ruido.png` (línea verde contra línea naranja), `6_lector.png`

## Problema 10. Por qué con 5 QR sigue sin funcionar

**Mensaje:** los módulos que difieren también se parecen entre sí, por cómo están escritas las URL.

- **El `_menu.png` se corre según el largo del nombre.** food (4 letras), wine (4), beverages (9), dessert (7), specials (8). Sólo food y wine tienen el mismo largo, así que en esos dos `_menu.png` cae en los mismos módulos, y también coincide el largo de la URL que el QR guarda al principio. Por eso **menú principal y vinos son el par más parecido aun después de sacar lo común (0,54)**, y las ternas que los incluyen son las peores.
- **Las letras distintas comparten bits.** Cada carácter son 8 módulos y en ASCII todas las minúsculas empiezan con `011`: f/w, o/i y b/d comparten 6 de sus 8 bits.
- **Sacar lo común obliga a parecidos "al revés".** En cada módulo que queda, los 5 QR se reparten 4 contra 1 (113 módulos) o 3 contra 2 (301). Eso fuerza un solapamiento medio de −0,09, y para Hebb un parecido negativo interfiere igual que uno positivo.
- **Comparación:** 5 patrones al azar de 414 módulos tienen un solapamiento máximo de 0,08; los QR llegan a 0,54.
- *Pendiente de verificar:* no se mapeó módulo por módulo qué carácter cae en cada módulo (no hay librería de QR instalada). Lo del `_menu.png` sale del estándar QR y de los largos medidos.

**Gráficos:** `reports/ejercicio2/variante/1_solapamientos.png`, `5_ternas.png` (las 3 peores filas son las que tienen menú + vinos)

---

## Soluciones probadas

Fuente: `reports/ejercicio2/mejoras/README.md`.

**Mensaje:** con la pseudoinversa se recuperan los 5 QR. Dos redes de Hebb también, pero porque el desempate entre redes compara con los originales (ver "Problema 11"). Ninguna le gana a comparar con los originales.

| Método | Ruido aleatorio (correctas) | Esquina rota (correctas) | Estado |
|---|---|---|---|
| Hebb, 5 QR completos | 0 % siempre | 0 % siempre | Resultado principal |
| **Pseudoinversa**, 5 QR | 100 % hasta p = 0,35; 96 % en 0,4 | 100 % hasta 25×25 (46 % del QR) | No está en el apunte: sólo comparación |
| **Dos redes de Hebb** sin compartidos ({menú, cócteles, postres} + {vinos, especiales}) | 100 % hasta p = 0,35; 90 % en 0,4 | 100 % hasta 17×17 (21 %) | El desempate compara con los originales: ver Problema 11 |
| Sin red: QR original más cercano | 100 % hasta p = 0,35; 98 % en 0,4 | 100 % hasta 25×25 | Techo; no es Hopfield |
| Lector solo, sin red | sólo con p = 0 | sólo la esquina sin cuadrado de posición, hasta 9×9 | Referencia |

- **Energía con la pseudoinversa:** los 5 QR quedan estables con la misma energía (−682,00) y la mezcla deja de ser estable (−607,89, más alta). Es lo que el apunte espera y Hebb no cumplía.
- **Gráficos:** `mejoras/1_ruido.png`, `mejoras/2_esquina.png` (el mejor para el caso de uso), `mejoras/3_mancha.png`, `mejoras/4_ejemplo_esquina.png` (muy visual: la red reconstruye el cuadrado de posición).

### Varias redes de Hebb, a fondo (`reports/ejercicio2/dos_redes/README.md`)

**Mensaje:** repartiendo los QR en varias redes, cada QR queda estable y se recupera si la consulta llega a su red. Elegir la red es otra cosa: ver Problema 11.

- **Cómo se elige:** hasta p = 0,35, en el 80 % de los casos sólo una red devuelve un QR válido. En el 20 % restante (siempre food: la otra red devuelve wine) se elige el más parecido y sale bien. 0 errores (`3_diagnostico.png`).
- **Una red contra dos** (`c1`–`c7`): las neuronas inestables pasan de 71–225 a 0; el estado espurio del 100 % pasa a 100 % correcto hasta p = 0,35; un QR sin ruido ya no se mueve (0 iteraciones); los QR son los pozos más hondos de cada red; el lector lee el 100 %.
- **El reparto importa con ternas:** de 67 % a 85 % con mucho ruido; juntar food y wine es lo peor. **Con 2 + 2 + 1 (QR completos, sin sacar nada) cualquier reparto anda** (90 % a 92 %), porque un par siempre es estable.
- **Límite:** con 5 redes de un QR cada una es exactamente comparar con los originales.

### Problema 11. Con dos redes, el que resuelve el par menú–vinos es el desempate, no la red

Fuente: `reports/ejercicio2/dos_redes/README.md`, experimento 5 (`eleccion_resumen.csv`).

- **La red de vinos le devuelve vinos a las 140 consultas de menú** con p ≤ 0,3. Las dos salidas son QR guardados, así que hay que desempatar.
- **Sin desempate:** 80 % (menú 0 %). **Con el desempate "la más parecida a la consulta":** igual que comparar con los originales hasta p = 0,35, y nunca mejor (acierta 6 que Hamming falla, falla 25 que Hamming acierta, de 1100). O sea, el problema 2 no se resuelve: se pasa a quien elige la red, y quien elige es Hamming.
- **Si le mandamos la consulta a la red correcta (oráculo):** 100 % hasta p = 0,35. Cada red, sola, funciona.
- **Elegir por energía** (la red donde la consulta tiene menor energía; sólo usa los pesos): 100 % hasta p = 0,25, 99 % en 0,3, 98 % en 0,35. No mira los originales, pero con Hebb esa energía es −N/2 · Σ (solapamiento con cada QR de la red)², así que sigue midiendo parecido con los QR guardados.
- **Conclusión:** dos redes no le agregan nada a comparar con los originales. Para el problema 2, la solución con una sola red y sin desempate es el desaprendizaje (90,5 %).

## Soluciones que quedan sin probar

| Opción | Idea | Estado |
|---|---|---|
| **Regenerar los QR con máscaras distintas** | El estándar QR tiene 8 máscaras; una por menú hace que los datos se vean casi al azar entre QR | Requiere `segno` |
| **Más corrección de errores** | Nivel H (~30 %) en vez de L (~7 %): el lector solo aguanta más | Propuesta |

## Mensaje final para la presentación

- Hopfield con Hebb necesita patrones que se parezcan poco entre sí.
- Los QR de un mismo restaurante son lo contrario: comparten formato y casi toda la URL.
- Sacar lo común ayuda (de 2 a 3 QR) pero no alcanza, porque lo que queda también se parece.
- **El problema no es la red ni la cantidad: es cómo se generaron los QR.**
