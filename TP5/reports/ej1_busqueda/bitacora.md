# Bitácora de la búsqueda del ejercicio 1a

El razonamiento detrás de cada decisión, en el orden en que se tomaron, con los errores que se
corrigieron en el camino. Los resultados finales están en `resumen.md`.

## 1. Punto de partida

- La baseline (`ej1a_baseline`: 16-8, tanh, Adam η 0.01, semilla 42) no cumplía la meta: 27/32
  letras dentro y un máximo de 4 píxeles mal. Los barridos previos (`ej1a_*`) cambiaban un
  parámetro por vez alrededor de una base común (32-16, tanh, Adam 0.01, latente 2).
- Por esa base común, **una misma corrida aparecía en varios barridos**: 32-16 · tanh · Adam 0.01
  · semilla 1 fallaba (15/32) en activations, architectures, latent_size, optimizers y
  denoising. Era un solo resultado visto cinco veces, no cinco evidencias independientes.
- La primera recomendación (64-32-16) salía de un resumen ajeno sin verificar y de un criterio
  elegido sin anunciarlo. De ahí la decisión: **definir los criterios antes de correr y
  justificar cada límite del diseño.**

## 2. Diseño de la etapa 1

**Arquitecturas: dos ejes en vez de una lista arbitraria.** Ancho w ∈ {16, 32, 64, 128} ×
profundidad d ∈ {1, 2, 3}, con cada capa a la mitad de la anterior. Cada resultado se puede
leer como "más capacidad" o "más profundidad". La regla de la mitad era una restricción propia:
se avisó que había que verificarla (ver la sección 6).
- **Límite inferior, por datos:** sin capas ocultas y con [8] fallaba 0/5. 16 era la transición
  (4/5 y 3/5).
- **Límite superior, por saturación**, con la regla fijada antes de correr: si 128 no mejora a
  64, no se prueba 256.
- **d ≤ 3:** con d = 4 y w = 16, la última capa tendría 2 neuronas, el tamaño del latente.

**Inicializador emparejado con la activación, por teoría.** Para que la señal no se apague ni
explote, la varianza por capa debe mantenerse en ~1: Var(salida) ≈ n · Var(w) · E[x²].
- **Xavier**, 2/(n_ent + n_sal), supone una activación ≈ identidad cerca de 0 y simétrica:
  vale para tanh.
- **He**, 2/n_ent, compensa que relu anula la mitad negativa (un factor 2 por capa). Con
  leaky_relu de pendiente 0.01, el factor es 1,0001: es prácticamente He.
- **Sigmoid no encaja en ninguno.** Tiene pendiente 1/4 en 0 (con Xavier, la varianza se achica
  ~16 veces por capa) y no está centrada (media 0,5). Por eso se probó con xavier, con he y con
  un inicializador nuevo, **`xavier_sigmoid` = xavier × 4**. Sin el × 4 no se podía distinguir
  "sigmoid es lenta" de "sigmoid está mal inicializada".

**Por qué no tiene sentido recentrar sigmoid.** sigmoid(x) − 0,5 = tanh(x/2)/2: los dos factores
1/2 se absorben en los pesos, y el 0,5 constante en el sesgo de la capa siguiente. **Una red
sigmoid y una red tanh representan exactamente las mismas funciones.** Toda diferencia entre
ellas es de optimización (escala y centrado), y el control "sigmoid centrada" ya existe: es tanh.

**Por qué no corresponde `xavier_sigmoid` con tanh ni con relu.** El × 4 compensa la pendiente
1/4 de sigmoid. En tanh solo satura las neuronas (pendiente casi 0), y en relu multiplica la
varianza por ~8 por capa. Su único valor sería como prueba de "inicialización demasiado
grande", que es otra pregunta.

**Otros factores.**
- **η:** 0.01, 0.003 y 0.001, porque la estabilidad dependía mucho de η.
- **Optimizador:** solo Adam, fijado por costo y para verificar en la etapa 2. SGD ya había
  fallado 0/5.
- **10 semillas.**

**Controles cada 10 épocas.** Un control mide los píxeles mal de las 32 letras limpias. Medido:
con un control por época, el tiempo sube un 25 % y el disco 10 veces, sin cambiar conclusiones.
El intervalo conveniente es el más grande cuyo error (≤ 10 épocas) sea chico frente a la
variación entre semillas (decenas a cientos de épocas).

**10000 épocas, sin detención temprana.** Detener al no mejorar ahorraría tiempo, pero
eliminaría la medición de recaídas, que necesita un horizonte largo después de llegar a la meta.

**Criterios, en orden:** semillas que terminan en la meta, recaídas, pasos medianos hasta la
meta, parámetros. Con 10 semillas, 10/10 contra 9/10 no se distingue (Fisher p ≈ 1) y 10/10
contra 5/10 sí (p ≈ 0,03): la tasa de éxito descarta y los pasos separan.

**Infraestructura agregada:**
- `--jobs N` reparte las corridas entre procesos. Cada corrida depende solo de su semilla, así
  que el resultado es idéntico al de correr en serie (lo verifica un test).
- El script de tabla aplica los criterios.

## 3. Resultados de la etapa 1 y lo que se ajustó

- **Ajuste del criterio de recaídas, hecho después de ver los datos y declarado.** En forma
  exacta, un solo control fallido entre ~1000 (0,01 %) pesaba más que una convergencia 30 veces
  más lenta. Desde entonces se compara redondeado al 1 %. De las 136 configuraciones con 10/10,
  solo 13 tienen recaídas de 1 % o más.
- **El ancho se satura en 64:** 64 y 128 con una capa cumplen siempre, así que no se probó 256.
- **tanh, relu y leaky_relu empatan:** en 23 configuraciones comparables, relu da 0,99× y leaky
  1,04× los pasos de tanh.
- **Sigmoid:** el × 4 mejora la confiabilidad (313 semillas contra 242 con xavier) pero no la
  velocidad (1,35× más lenta en 12 configuraciones comparables). Aun así es 2,85× más lenta que
  tanh. Lo que queda es atribuible al centrado, que ninguna inicialización arregla.
- **El "zigzag" de la sigmoid no centrada, bien explicado.** Que todos los pesos de una neurona
  reciban gradientes del mismo signo es cierto solo **por muestra**. Con un lote de 32 los signos
  se mezclan. El argumento correcto (LeCun, 1998) es de curvatura: entradas con media distinta
  de 0 estiran la superficie de la pérdida en esa dirección. Es un efecto general, no propio de
  este problema, y su tamaño depende de la profundidad, el optimizador y el batch.

## 4. Selección de finalistas: un error corregido

- **La primera propuesta fue casi toda tanh, sin razón.** Comparando la misma arquitectura,
  las tres activaciones empatan en 64 y 64-32, y solo en 128-64 tanh es más rápida. Las
  finalistas pasaron a ser **tanh, relu y leaky_relu × {64, 64-32, 128-64}**.
- **El mejor η estaba en el borde del rango.** Las 11 más rápidas usaban η = 0.01, el más alto
  probado. Cuando el óptimo cae en el borde, puede estar más allá. Por eso se agregaron 0.03 y
  0.1.

## 5. Diseño de la etapa 2

**Una sola cosa por vez** respecto de la referencia (Adam 0.01, BCE, batch 32, inicializador
emparejado), así cada diferencia tiene una causa clara.

**Qué se cruza y qué no.** Cruzar todo serían ~43000 corridas. Solo se cruzaron pares con un
mecanismo conocido de interacción:
- **Optimizador × η:** el η bueno depende del optimizador.
- **Activación × inicializador:** cada inicializador está derivado para una familia.
- **Batch × η:** el ruido que entra en cada paso escala aproximadamente con η/batch, así que
  un batch chico pide un η menor. Sin cruzarlo, un mal resultado con batch chico podía deberse
  al η y no al batch.
- **Pérdida × η no se cruzó:** Adam normaliza la escala del gradiente de cada peso, así que el
  tamaño menor de los gradientes de MSE se cancela en buena parte.

**La pérdida, a pesar de la teoría.** Con salida sigmoid, el gradiente de BCE es
salida − objetivo. El de MSE es 2(salida − objetivo)·sigmoid′(z), que se anula justo cuando un
píxel está saturado en el valor equivocado. BCE es la log-verosimilitud de una distribución
Bernoulli. Igual se midió, para no asumir.

**Batch, comparado con cuidado.** Con batch 32 hay un paso por época. Con batch 8 hay 4 pasos
por época, cada uno más ruidoso. Se reportan pasos, y se miran también épocas, que equivalen al
mismo cálculo.

**Divergencias como resultado.** Una corrida que diverge (por ejemplo con η 0.1 o con momentum)
abortaba el experimento entero. Se cambió para registrarla (`diverged_at_epoch`) y contarla
como falla.

## 6. Resultados de la etapa 2 y lo que se corrigió

- **η:** 0.03 es más rápido, pero 128-64 ya pierde una semilla. 0.1 falla casi siempre. El
  rango queda acotado por los dos lados. **Se eligió 0.01** por margen de estabilidad.
- **Adam** gana en todas las redes. Momentum divergió 2 veces con η 0.01.
- **Inicializador:** el emparejamiento se confirma en tanh. En relu, he y xavier empatan:
  difieren en √2 y con 2 capas eso se acumula poco.
- **MSE: 0/90, y el mecanismo quedó a la vista.** La pérdida final coincide casi exactamente
  con la cantidad de píxeles mal (0,4698 contra 0,469; 7,40625 contra 7,40625). Cada píxel mal
  aporta ~1, o sea que está saturado en el valor opuesto, donde el gradiente se anula.
- **Batch × η, la interacción confirmada.** Con η 0.01, batch 4 cumple 33/90 con 49 % de
  recaídas. Con η 0.001 cumple 90/90. Probado solo con η 0.01, el batch chico habría parecido
  malo.
- **La prueba de ancho constante partía de una premisa equivocada.** La penalización de la
  tercera capa venía sobre todo de **sigmoid** (64-32-16: 68/90) y de las redes angostas. Con
  tanh, relu y leaky_relu, 64-32-16 cumple 90/90, así que probar ancho constante 64 no podía
  mostrar nada. Se rehízo donde el efecto existe: un 2×2 con ancho 16, comparando [16, 16],
  [16, 16, 16], [16, 4] y [16, 8, 4].
  - relu falla por angostura. En 6 de 10 semillas de relu [16, 4] hay al menos una neurona
    muerta en la capa de 4, pero las otras 4 semillas también fallan sin neuronas muertas.
  - tanh falla por profundidad.
  - Con 6 comparaciones y la corrección de Bonferroni (p < 0,008), el efecto de tanh
    (p = 0,0095) queda justo en el borde.

## 7. relu 32: un segundo error de selección

- **Quedó afuera por dos motivos equivocados:**
  - La saturación en ancho 64 se fijó mirando el total de las 6 combinaciones, y con ancho 32
    sigmoid y tanh fallan. Es el mismo error de mezclar familias de activación.
  - Después se eligió solo tanh.
- **Con relu, una capa de 32 cumple 10/10 con los tres η.** Se le corrieron las pruebas de la
  etapa 2. Dan las mismas conclusiones, pero con small_normal o con momentum y η bajo cae a
  5/10. **Una red cerca del límite de capacidad tolera peor los cambios en otros
  hiperparámetros.** Esto es lo que justifica llamar "más robusta" a tanh 64.

## 8. Parámetros y tiempos: qué significa "más rápida"

- **Parámetros** = pesos + sesgos (n·m + m por capa). Las candidatas tienen entre 4 y 23 veces
  más parámetros que valores en el dataset (32 × 35 = 1120): la red memoriza, que es lo que
  pide el ejercicio. El cuello de botella real es el latente.
- **Menos pasos no es menos tiempo.** Medido en un solo núcleo, una red más grande necesita
  menos pasos, pero cada uno cuesta más:

  | Red | Tiempo por paso | Entrenamiento hasta la meta |
  |---|---|---|
  | relu 32 | 176 µs | 70 ms |
  | tanh 64 | 223 µs | **55 ms** |
  | tanh 64-32 | 342 µs | 60 ms |
  | tanh 128-64 | 496 µs | 67 ms |

  Con matrices tan chicas domina el costo fijo de cada operación de NumPy: 9 veces más
  parámetros cuestan solo ~2 veces más tiempo.
- **Más complejidad no da más confiabilidad pasado cierto punto.** Con una capa cumplen 119/180
  semillas con ancho 16, 178/180 con 32 y 180/180 con 64 y con 128.
- **Inferencia:** entre 12 y 21 µs por letra en todas. El tamaño de la red se nota recién con
  muchas letras juntas (36 a 204 µs para las 32).
- **Candidatas:** relu 32 (la más simple), tanh 64 (la más robusta y la que menos tarda en
  entrenar), tanh 64-32 (el equilibrio, por decisión: no gana en ningún criterio por sí sola) y
  tanh 128-64 (gana solo en pasos).
