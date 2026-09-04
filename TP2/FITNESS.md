# Fitness del TP2

Este documento explica las funciones de fitness implementadas para aproximar una
imagen usando triángulos semitransparentes.

La idea principal es simple: cada individuo representa una imagen generada. Para
evaluarlo, el programa renderiza sus triángulos sobre un canvas blanco y compara
esa imagen generada contra la imagen objetivo. El fitness es el número que resume
qué tan buena o mala fue esa comparación.

En la versión actual todos los evaluadores devuelven una similitud normalizada en
`[0, 1]`. Por eso, mayor es mejor.

```text
fitness = similitud normalizada
1       = coincidencia perfecta
0       = peor caso teórico de la métrica
```

El comparador `MSEComparator` define justamente eso: un fitness es mejor si su
valor es mayor que el de otro.

## Cómo se elige un fitness

La versión actual ya no configura los experimentos con muchos flags de CLI. Cada
corrida se describe en un archivo JSON y `main.py` lo recibe como argumento:

```bash
python main.py configs/default_config.json --image objetivo.png
```

La métrica se configura dentro de la sección `fitness`:

```json
{
  "image": "target.png",
  "fitness": { "metric": "mse" }
}
```

Si `fitness` se omite, se usa MSE global:

```json
{
  "image": "target.png"
}
```

La imagen, salida, semilla y previews pueden reemplazarse puntualmente por CLI:

```bash
python main.py configs/default_config.json --image otra.png --seed 42 --no-preview
```

El resto de los hiperparámetros vive en el JSON.

## Contexto general

Antes de entrar en cada métrica, conviene entender qué compara el programa.

1. El usuario declara una imagen objetivo con `image`.
2. `TriangleImageTarget` carga esa imagen con PIL y la lleva al RGB que usa el
   renderizador. Si el archivo tiene transparencia, primero compone los píxeles
   sobre fondo blanco; no descarta el canal alpha como un `convert("RGB")`
   directo, porque eso haría que píxeles transparentes con RGB oculto negro
   contaminen el fitness. Esta composición se hace antes de achicar por
   `max_size`, así el resize tampoco mezcla colores ocultos de zonas
   transparentes.
3. Si la imagen es más grande que `max_size`, se reduce para acelerar la
   evaluación.
4. Cada individuo (`TriangleIndividual`) tiene una lista ordenada de genes.
5. Cada gen (`TriangleGene`) describe un triángulo: posición, tamaño, ángulos,
   rotación, color RGB y transparencia.
6. `render(...)` dibuja todos los triángulos sobre fondo blanco.
7. El evaluador de fitness compara el render contra el objetivo.

El flujo conceptual es:

```text
individuo -> triángulos -> imagen RGB renderizada -> comparación contra objetivo
```

Todas las métricas comparan dos imágenes del mismo tamaño:

```text
target_arr   = imagen objetivo
rendered_arr = imagen generada por el individuo
```

Ambas tienen forma conceptual:

```text
alto x ancho x 3 canales RGB
```

Cada canal tiene valores entre `0` y `255`.

## Por qué hay muchas funciones de fitness

Una sola métrica no siempre captura bien "parecido visual".

Por ejemplo, si comparamos píxel contra píxel con MSE, un triángulo apenas
desplazado puede recibir una penalización grande aunque visualmente esté cerca.
Al revés, una métrica basada solo en colores puede decir que una imagen es buena
porque tiene la misma cantidad de rojo, azul y blanco, aunque esos colores estén
en cualquier lugar.

Por eso el proyecto incluye varias funciones de fitness. Cada una presiona al
algoritmo en una dirección distinta:

- `mse`: coincidencia exacta de colores por píxel.
- `regional`: parecido por regiones, con más peso en zonas detalladas.
- `ssim`: parecido estructural/perceptual.
- `blur`: parecido de formas generales.
- `multiscale`: parecido en varias resoluciones.
- `histogram`: parecido de distribución global de colores.
- `edge`: parecido entre mapas de bordes.
- `gradient`: parecido entre intensidad y orientación de bordes.
- `chamfer`: parecido de bordes tolerante a desplazamientos.
- `saliency`: MSE que pesa más las zonas visualmente importantes.
- `combo`: combinación ponderada de varias métricas.

Ninguna es universalmente mejor. Sirven para objetivos distintos y se pueden
combinar.

## Resumen rápido

| `fitness.metric` | Clase | Qué intenta mejorar | Escala |
| --- | --- | --- | --- |
| `mse` | `MSEEvaluator` | Color exacto por píxel | `0` a `1` |
| `regional` | `RegionalMSEEvaluator` | Error por zonas, priorizando detalle | `0` a `1` |
| `ssim` | `SSIMEvaluator` | Similitud estructural local | `0` a `1` |
| `blur` | `BlurredMSEEvaluator` | Formas y color general | `0` a `1` |
| `multiscale` | `MultiScaleMSEEvaluator` | Estructura gruesa y detalle fino | `0` a `1` |
| `histogram` | `ColorHistogramEvaluator` | Distribución global de colores | `0` a `1` |
| `edge` | `EdgeMSEEvaluator` | Ubicación exacta de contornos | `0` a `1` |
| `gradient` | `GradientOrientationEvaluator` | Fuerza y orientación de contornos | `0` a `1` |
| `chamfer` | `ChamferEdgeEvaluator` | Distancia entre contornos | `0` a `1` |
| `saliency` | `SaliencyMSEEvaluator` | Error en zonas visualmente importantes | `0` a `1` |
| `combo` | `CompositeEvaluator` | Mezcla de criterios | depende de pesos |

Internamente, varias métricas calculan primero un error y después lo convierten a
fitness. Para errores basados en diferencias RGB, el máximo teórico usado suele
ser:

```text
255^2 = 65025
```

Es el máximo error cuadrático posible por canal si un píxel vale `0` en una
imagen y `255` en la otra.

## `MSEEvaluator`: MSE global

Configuración:

```json
"fitness": { "metric": "mse" }
```

`mse` es la métrica por defecto.

### Qué mide

MSE significa Mean Squared Error, o error cuadrático medio.

En imágenes con transparencia, el MSE usa además pesos derivados del alpha:

- píxeles opacos pesan `1.0`;
- píxeles totalmente transparentes pesan `0.5`;
- píxeles semitransparentes quedan en el medio.

Esto evita dos problemas comunes en PNGs de logos: que el RGB oculto de zonas
transparentes se interprete como color real, y que el fondo domine por completo
la comparación. El fondo sigue importando, porque pintar triángulos fuera del
objeto también debería penalizarse, pero el contenido visible recibe más presión
evolutiva.

Compara cada píxel de la imagen objetivo contra el píxel de la misma posición en
la imagen renderizada. Para cada canal RGB calcula la diferencia, la eleva al
cuadrado y luego promedia todo.

Conceptualmente:

```text
diferencia = objetivo - render
error_pixel_canal = diferencia^2
MSE = promedio de todos los errores
```

Si el objetivo y el render son iguales, todas las diferencias son `0`. Ese error
crudo luego se convierte a fitness:

```text
fitness = 1 - MSE / 65025
```

Al elevar al cuadrado, los errores grandes pesan mucho más que los errores
chicos. Una diferencia de `80` no pesa el doble que una de `40`; pesa cuatro
veces más, porque `80^2 = 6400` y `40^2 = 1600`.

Si el MSE es `0`, el fitness es `1`. Si el MSE llega al peor caso teórico, el
fitness se acerca a `0`.

### Ventajas

Es la métrica más directa y fácil de interpretar. Si baja, la imagen se está
acercando al objetivo píxel a píxel. También es rápida, estable y sirve como
baseline para comparar cualquier experimento.

### Problemas

MSE es muy literal. No entiende formas ni percepción humana. Solo mira:

```text
este píxel contra este píxel
```

Por eso penaliza mucho una forma que está casi bien pero corrida unos píxeles.
Para el ojo humano puede ser una aproximación razonable; para MSE, cada píxel
desplazado cuenta como error.

### Cuándo conviene usarlo

Conviene para validar el flujo completo y como primer experimento. Funciona bien
en imágenes simples, banderas, logos o pictogramas con colores planos.

## `RegionalMSEEvaluator`: MSE por regiones

Configuración:

```json
"fitness": {
  "metric": "regional",
  "regional": {
    "grid_rows": 8,
    "grid_cols": 8,
    "detail_weight": 1.0
  }
}
```

### Qué mide

También usa MSE, pero no promedia toda la imagen de una sola vez. Primero divide
la imagen en una grilla `grid_rows x grid_cols`. Con los valores por defecto:

```text
8 x 8 = 64 regiones
```

Luego calcula el MSE de cada región por separado.

La diferencia importante es que no todas las regiones pesan igual. Las regiones
con más detalle o contraste en la imagen objetivo reciben más peso.

### Cómo decide qué región tiene más detalle

Para cada celda de la grilla, mira los valores RGB del objetivo y calcula el
desvío estándar. Si una región es casi plana, por ejemplo todo blanco, el desvío
estándar es bajo. Si una región tiene bordes, cambios fuertes de color o textura,
el desvío estándar sube.

El peso final de cada región se construye así:

```text
peso = 1 + detail_weight * detalle_normalizado
```

Luego todos los pesos se normalizan para que sumen `1`.

### Efecto de `fitness.regional.detail_weight`

Si `detail_weight` es `0`, todas las regiones pesan igual.

Si `detail_weight` es mayor, las regiones con más contraste pesan más. Valores
altos enfocan el algoritmo en zonas detalladas, pero pueden descuidar fondos o
regiones lisas grandes.

### Ventajas

Reduce el problema de que el promedio global esconda errores importantes. Si una
imagen tiene mucho fondo blanco y un logo pequeño, MSE global puede obtener un
valor aceptable aproximando bien el fondo y mal el logo. MSE regional le da más
presión a las zonas con detalle.

### Problemas

Depende de la grilla. Si una forma importante cae justo entre celdas, el peso se
reparte. También puede sobrevalorar regiones con mucho contraste aunque no sean
semánticamente importantes.

### Cuándo conviene usarlo

Conviene cuando MSE global mejora demasiado el fondo y demasiado poco el objeto
principal.

## `SSIMEvaluator`: similitud estructural

Configuración:

```json
"fitness": {
  "metric": "ssim",
  "ssim": {
    "window_size": 7,
    "mse_weight": 0.5
  }
}
```

### Qué mide

SSIM significa Structural Similarity Index. MSE compara diferencias exactas de
píxeles; SSIM compara estadísticas locales de la imagen.

Para cada ventana local compara tres ideas:

- luminancia: si el brillo medio se parece;
- contraste: si la variación local se parece;
- estructura: si los patrones locales se parecen.

En este proyecto se calcula por canal RGB y luego se promedia.

SSIM normalmente vale:

```text
1  = imágenes muy similares
0  = poca similitud
-1 = casos extremadamente opuestos
```

Como el motor maximiza fitness, el proyecto transforma los términos de error en
similitud:

```text
ssim_error = 1 - mean_ssim
```

Si SSIM es `1`, el error queda `0`.

### Por qué se mezcla con RMSE

La implementación no usa SSIM puro por defecto. Lo mezcla con RMSE normalizado:

Luego combina SSIM con RMSE normalizado y convierte el error combinado a una
similitud donde `1` es mejor que `0`.

Con `"mse_weight": 0.5`, mitad de la señal viene de SSIM y mitad de RMSE.

Esto se hace porque SSIM puro puede comportarse mal en este problema. En zonas
lisas o imágenes simples, puede tolerar diferencias de color que visualmente son
importantes. El algoritmo podría explotar eso y converger a una imagen plana que
preserva cierta estructura estadística pero no se parece bien al objetivo.

El término RMSE obliga a respetar también los colores reales.

### Efecto de `fitness.ssim.window_size`

Define el tamaño de la ventana local usada para calcular medias y varianzas.
Debe ser impar y al menos `3`. Ventanas chicas miran detalles más locales;
ventanas grandes miran estructura más amplia. Si la imagen de trabajo es más
chica que la ventana, el evaluador reduce el tamaño internamente para que entre.

### Ventajas

Puede correlacionar mejor con percepción humana que MSE. No se limita a decir
"este píxel está mal", sino que mira estructura local.

### Problemas

Es menos intuitivo que MSE. Un valor bajo no siempre significa que todos los
colores estén bien. En imágenes generadas con pocos triángulos, puede premiar
estructuras generales antes que precisión cromática.

### Cuándo conviene usarlo

Conviene para experimentar con parecido visual/perceptual, especialmente dentro
de `combo`. No conviene usar SSIM puro con `"mse_weight": 0` salvo que se quiera
estudiar específicamente ese comportamiento.

## `BlurredMSEEvaluator`: MSE con blur

Configuración:

```json
"fitness": {
  "metric": "blur",
  "blur": { "sigma": 1.5 }
}
```

### Qué mide

Aplica un desenfoque gaussiano a la imagen objetivo y al render. Después calcula
MSE entre esas imágenes desenfocadas:

```text
blur(objetivo) vs blur(render)
```

En vez de exigir coincidencia exacta de cada borde y cada píxel, compara formas
y colores generales.

### Qué es el blur gaussiano

Un blur gaussiano reemplaza cada píxel por una mezcla ponderada de sus vecinos.
Los vecinos cercanos pesan más que los lejanos.

El parámetro `sigma` controla cuánto se difumina:

- `sigma = 0`: no hay blur;
- `sigma` chico: suavizado leve;
- `sigma` grande: se pierden más detalles finos.

En el código, el blur se aplica en ancho y alto, pero no mezcla canales RGB entre
sí. Rojo, verde y azul se suavizan espacialmente, pero no se mezclan entre
canales.

### Ventajas

Ayuda en etapas tempranas, cuando los triángulos todavía no están perfectamente
ubicados pero sí pueden estar acercándose a la forma general.

### Problemas

Puede ignorar detalles finos. Si `sigma` es alto, dos imágenes borrosas pueden
parecerse aunque la versión nítida sea mala.

### Cuándo conviene usarlo

Conviene en imágenes donde primero se quiere capturar la estructura gruesa.
También puede servir dentro de `combo`, mezclado con MSE o bordes.

## `MultiScaleMSEEvaluator`: MSE en varias escalas

Configuración:

```json
"fitness": {
  "metric": "multiscale",
  "multiscale": { "scales": [1.0, 0.5, 0.25] }
}
```

### Qué mide

Calcula MSE varias veces, pero a distintas resoluciones. Con los valores por
defecto compara:

```text
100% de tamaño
50% de tamaño
25% de tamaño
```

Para cada escala:

1. reduce objetivo y render a esa escala;
2. calcula MSE;
3. combina los errores con pesos internos iguales.

### Por qué sirve mirar varias escalas

Una imagen reducida pierde detalles chicos, pero conserva estructura grande.

- escala `0.25`: evalúa composición global;
- escala `0.5`: evalúa formas medianas;
- escala `1.0`: evalúa detalle fino.

Al combinar escalas, el algoritmo recibe una señal más equilibrada.

### Ventajas

Ayuda a no depender de un único nivel de detalle. Puede ser más estable que MSE
puro en etapas tempranas.

### Problemas

Es más caro que MSE porque calcula varias comparaciones y resize. Si se usan
muchas escalas bajas, puede perder presión para mejorar detalles.

### Cuándo conviene usarlo

Conviene cuando se quiere una métrica generalista. Suele ser una buena alternativa
a `blur`: ambas suavizan la exigencia de coincidencia exacta, pero `multiscale`
lo hace comparando resoluciones distintas.

## `ColorHistogramEvaluator`: histograma de color

Configuración:

```json
"fitness": {
  "metric": "histogram",
  "histogram": {
    "bins": 32,
    "mse_weight": 0.5
  }
}
```

### Qué mide

Compara la distribución global de colores entre objetivo y render. No mira dónde
está cada color. Solo mira cuánto aparece cada rango de valores en cada canal RGB:

```text
R: histograma de intensidades rojas
G: histograma de intensidades verdes
B: histograma de intensidades azules
```

Luego normaliza cada histograma para que represente proporciones y calcula una
distancia L2 entre histogramas.

### Qué son los bins

Un histograma agrupa valores en intervalos. Con `"bins": 32`, el rango `0..255`
se divide en 32 grupos. Más bins significa más detalle en la distribución de
colores; menos bins significa una comparación más gruesa.

### Por qué se mezcla con RMSE

Como el histograma es ciego a la posición, usarlo solo puede ser peligroso. Un
render con los colores correctos en lugares equivocados podría puntuar bien.

El código mezcla el término de histograma con RMSE normalizado:

```text
fitness = (1 - mse_weight) * hist_term + mse_weight * rmse_normalizado
```

Con `"mse_weight": 0.5`, la mitad del fitness exige distribución de colores y la
otra mitad exige parecido espacial.

### Ventajas

Ayuda cuando el algoritmo todavía no encontró buenos colores. Puede presionar a
que aparezca la paleta correcta.

### Problemas

No entiende formas ni posiciones. Si se usa solo, puede producir soluciones
visualmente malas.

### Cuándo conviene usarlo

Conviene dentro de `combo` o con su mezcla RMSE activada. Puede ayudar en
banderas, logos simples o íconos donde la paleta de colores es importante.

## `EdgeMSEEvaluator`: MSE entre bordes

Configuración:

```json
"fitness": {
  "metric": "edge",
  "edge": { "sigma": 1.0 }
}
```

### Qué mide

Detecta bordes en la imagen objetivo y en la imagen renderizada. Después calcula
MSE entre esos mapas de bordes.

El evaluador no compara directamente los colores RGB. Primero convierte la imagen
a escala de grises usando luminancia Rec. 709:

```text
luminancia = 0.2126 * R + 0.7152 * G + 0.0722 * B
```

Luego aplica Sobel para estimar cambios horizontales y verticales.

### Qué es Sobel

Sobel es un operador de detección de bordes. Calcula algo parecido a una derivada:

- `gradient_x`: cambios hacia izquierda/derecha;
- `gradient_y`: cambios hacia arriba/abajo.

Con ambos calcula magnitud:

```text
magnitud = sqrt(gradient_x^2 + gradient_y^2)
```

La magnitud indica qué tan fuerte es el borde. El proyecto normaliza esa magnitud
para mantenerla en rango `0..255`.

### Efecto de `fitness.edge.sigma`

Antes de detectar bordes puede aplicar blur. Si `sigma` es `0`, detecta bordes
sin suavizar. Si `sigma` es mayor, reduce ruido y pequeños detalles antes de
Sobel.

### Ventajas

Presiona a que las formas estén en el lugar correcto. Es útil cuando el contorno
importa más que el color exacto.

### Problemas

No mide bien color interno. Si el contorno coincide pero los colores están mal,
`edge` puede no castigar lo suficiente. También es sensible a desplazamientos:
si el borde está cerca pero no exactamente en la misma posición, puede penalizar
bastante.

### Cuándo conviene usarlo

Conviene para siluetas, señales, íconos y figuras con bordes claros. Es mejor
usarlo combinado con MSE, histograma o saliency para no perder color.

## `GradientOrientationEvaluator`: magnitud y orientación de gradientes

Configuración:

```json
"fitness": {
  "metric": "gradient",
  "gradient": {
    "sigma": 1.0,
    "orientation_weight": 0.5
  }
}
```

### Qué mide

También usa gradientes Sobel, pero mira dos cosas:

1. si la fuerza del borde es parecida;
2. si la orientación del borde es parecida.

La fuerza sale de la magnitud del gradiente. La orientación describe la dirección
del cambio: un borde vertical tiene una orientación distinta de un borde
horizontal.

### Componente de magnitud

Compara la magnitud de bordes del objetivo y del render:

```text
magnitude_error = MSE(magnitud_objetivo, magnitud_render) / 255^2
```

Se divide por `255^2` para normalizarlo aproximadamente a `0..1`.

### Componente de orientación

Para comparar orientación usa el coseno entre vectores de gradiente:

```text
cos(theta) = dot(g_objetivo, g_render) / (|g_objetivo| * |g_render|)
```

Luego usa valor absoluto:

```text
abs(cos(theta))
```

Esto hace que un borde claro-oscuro y uno oscuro-claro se consideren con la misma
orientación. Importa la línea del borde, no de qué lado está lo claro y lo oscuro.

Si las orientaciones coinciden, `orientation_error` es `0`. Si son perpendiculares,
se acerca a `1`. Además, el error de orientación se pondera por la fuerza del
borde, así que importa más donde realmente hay contorno.

### Efecto de `fitness.gradient.orientation_weight`

Combina magnitud y orientación:

```text
score = (1 - orientation_weight) * magnitude_error
      + orientation_weight * orientation_score
```

Con `0.5`, ambas partes pesan igual. Si se acerca a `0`, importa más la fuerza
del borde. Si se acerca a `1`, importa más la orientación.

### Ventajas

Es más rica que `edge` porque no solo pregunta "hay borde acá", sino también
"el borde apunta en una dirección parecida".

### Problemas

No mide color directamente. Un puntaje bueno puede venir de orientaciones
correctas aunque la imagen todavía no tenga colores correctos.

### Cuándo conviene usarlo

Conviene para objetivos con geometría marcada: triángulos, señales, letras
grandes, íconos lineales y contornos diagonales. Suele tener más sentido en
`combo` que como único fitness.

## `ChamferEdgeEvaluator`: distancia Chamfer entre bordes

Configuración:

```json
"fitness": {
  "metric": "chamfer",
  "chamfer": {
    "sigma": 1.0,
    "threshold": 20.0
  }
}
```

### Qué mide

Compara bordes, pero de manera más tolerante que `edge`.

Primero detecta bordes con Sobel. Después convierte el mapa de bordes a una
máscara booleana:

```text
hay_borde = magnitud >= threshold
```

Luego mide qué tan lejos están los bordes del render de los bordes del objetivo,
y también al revés:

```text
distancia = (render -> objetivo + objetivo -> render) / 2
```

Se normaliza por la diagonal de la imagen, por eso queda aproximadamente entre
`0` y `1`.

### Diferencia contra `edge`

`edge` compara mapas de bordes píxel a píxel. Si un borde está corrido dos
píxeles, penaliza donde falta el borde y donde sobra el borde.

`chamfer` pregunta otra cosa:

```text
para cada borde de una imagen, ¿a qué distancia está el borde más cercano de la otra?
```

Entonces, si un contorno está cerca pero no exacto, el castigo es menor.

### Qué hace `distance_transform_edt`

La implementación usa `distance_transform_edt`, que para cada píxel calcula la
distancia al píxel de borde más cercano. Con eso puede tomar todos los bordes del
render y consultar qué distancia tienen al borde más cercano del objetivo. Luego
hace la dirección inversa.

### Efecto de `fitness.chamfer.threshold`

Define cuánta magnitud Sobel hace falta para considerar que hay borde. Si el
threshold es bajo, aparecen muchos bordes. Si es alto, solo quedan bordes fuertes.

### Ventajas

Es muy útil en etapas tempranas porque no exige que los contornos estén
perfectamente alineados. Premia que estén cerca.

### Problemas

No mide color. Al binarizar, pierde información de cuán fuerte era el borde.

### Cuándo conviene usarlo

Conviene para siluetas, pictogramas, señales e imágenes donde el contorno es
clave. Es buena candidata para combinar con MSE o histograma.

## `SaliencyMSEEvaluator`: MSE con saliencia visual

Configuración:

```json
"fitness": {
  "metric": "saliency",
  "saliency": {
    "weight": 3.0,
    "sigma": 2.0
  }
}
```

### Qué mide

Calcula MSE, pero no todos los píxeles pesan igual.

Primero estima qué zonas del objetivo son más salientes o visualmente
importantes. En esta implementación, la saliencia se estima usando magnitud de
gradiente:

```text
zonas con bordes / alto contraste -> más salientes
zonas planas -> menos salientes
```

Después calcula el error cuadrático por píxel y lo promedia usando pesos.

### Cómo se calculan los pesos

1. Calcula gradientes Sobel sobre el objetivo.
2. Toma la magnitud del gradiente.
3. Opcionalmente suaviza esa magnitud con blur gaussiano.
4. Normaliza el resultado a `0..1`.
5. Construye pesos:

```text
peso = 1 + weight * saliencia_normalizada
```

Todos los píxeles mantienen peso base `1`. Eso es importante: las zonas no
salientes no desaparecen, solo pesan menos.

Con `"weight": 3.0`, un píxel muy saliente puede pesar hasta cuatro veces más que
un píxel sin saliencia.

### Efecto de `fitness.saliency.sigma`

Suaviza el mapa de saliencia. Esto hace que no pese solo el borde exacto, sino
también su vecindad.

### Diferencia contra `regional`

`regional` asigna peso por celda de una grilla. `saliency` asigna peso por píxel.
Entonces `saliency` es más fino espacialmente.

### Ventajas

Mantiene la lógica de MSE pero enfoca el esfuerzo en zonas importantes. Suele ser
más compatible con color que las métricas puras de borde, porque sigue comparando
colores RGB.

### Problemas

La saliencia está basada en contraste, no en significado. Si una zona tiene mucho
contraste pero no es importante para nosotros, va a pesar más igual.

### Cuándo conviene usarlo

Conviene para objetivos donde los bordes o contrastes definen la figura, pero
también importa el color. Es una opción equilibrada entre MSE puro y métricas
estructurales.

## `CompositeEvaluator`: fitness combinado

Configuración:

```json
"fitness": {
  "metric": "combo",
  "combo": {
    "mse": 0.25,
    "gradient": 0.25,
    "chamfer": 0.25,
    "saliency": 0.25
  }
}
```

También se pueden declarar parámetros de las métricas usadas:

```json
"fitness": {
  "metric": "combo",
  "combo": {
    "mse": 0.4,
    "chamfer": 0.25,
    "saliency": 0.25,
    "histogram": 0.1
  },
  "chamfer": { "sigma": 1.0, "threshold": 20.0 },
  "saliency": { "weight": 2.5, "sigma": 2.0 },
  "histogram": { "bins": 32, "mse_weight": 0.4 }
}
```

### Qué mide

Combina varias métricas en una sola. Cada componente se evalúa por separado y
luego se hace una suma ponderada:

```text
fitness = w1 * fitness_1 + w2 * fitness_2 + ...
```

Los pesos se renormalizan para que sumen `1`.

Por ejemplo:

```json
"combo": {
  "mse": 0.5,
  "histogram": 0.2,
  "chamfer": 0.3
}
```

significa aproximadamente:

```text
50% parecido píxel a píxel
20% distribución de colores
30% cercanía de contornos
```

### Por qué se puede combinar directamente

Todas las métricas del módulo devuelven fitness normalizado en `[0, 1]`. Eso
evita que una métrica domine por escala numérica y permite que los pesos del
combo representen influencia relativa.

Si se declara:

```json
"combo": { "mse": 0.5, "chamfer": 0.5 }
```

ambas señales entran con el mismo peso conceptual. Los pesos se renormalizan, así
que `{ "mse": 5, "chamfer": 5 }` equivale a `{ "mse": 0.5, "chamfer": 0.5 }`.

### Cache de render

Cuando se usa `combo`, el mismo individuo se evalúa con varias métricas. Sin
cuidado, eso renderizaría la misma imagen muchas veces.

Para evitarlo, `CompositeEvaluator` abre un scope de cache en `TriangleContext`.
Durante la evaluación de un individuo, si varias métricas piden el render, se
reutiliza la misma imagen renderizada. Luego el cache se limpia.

### Ventajas

Permite equilibrar objetivos: color exacto, estructura global, bordes,
distribución de colores y zonas importantes. En problemas visuales, muchas veces
una combinación funciona mejor que una métrica aislada.

### Problemas

Tiene más parámetros y cuesta más interpretar el resultado. También puede ser más
lento porque calcula varias métricas.

### Cuándo conviene usarlo

Conviene cuando una métrica sola produce soluciones con un defecto claro:

- MSE respeta color pero no encuentra buenos contornos.
- Histograma encuentra paleta pero desordena la imagen.
- Chamfer encuentra silueta pero ignora color.
- Blur encuentra composición general pero no ajusta detalles.

## Sobre normalización

En versiones anteriores hacía falta envolver métricas con `NormalizedEvaluator`
para combinarlas, porque algunas devolvían MSE crudo y otras valores cercanos a
`[0, 1]`.

En la versión actual cada evaluador ya devuelve fitness normalizado en `[0, 1]`.
Por eso `combo` puede sumar directamente métricas distintas con pesos relativos.

## Relación entre fitness y evolución

El algoritmo genético no sabe qué significa cada fitness. Solo sabe comparar
valores.

El ciclo es:

1. generar individuos;
2. renderizarlos;
3. calcular fitness;
4. ordenar de mayor a menor fitness;
5. seleccionar padres;
6. cruzar y mutar;
7. decidir supervivencia;
8. repetir.

Cambiar el fitness cambia la presión evolutiva. Si usamos `mse`, el algoritmo
busca reducir diferencias exactas de color. Si usamos `edge`, busca alinear
contornos. Si usamos `histogram`, busca igualar la distribución de colores. Si
usamos `combo`, busca una mezcla.

El fitness es la definición operacional de "parecido". Lo que la función premie
es lo que el algoritmo va a intentar producir.

## Tabla de parámetros

| Parámetro JSON | Métrica | Significado |
| --- | --- | --- |
| `fitness.metric` | todas | Métrica principal: `mse`, `regional`, `ssim`, `blur`, `multiscale`, `histogram`, `edge`, `gradient`, `chamfer`, `saliency` o `combo` |
| `fitness.combo` | `combo` | Objeto de pesos por métrica |
| `fitness.regional.grid_rows` | `regional` | Cantidad de filas de la grilla |
| `fitness.regional.grid_cols` | `regional` | Cantidad de columnas de la grilla |
| `fitness.regional.detail_weight` | `regional` | Cuánto más pesan regiones con detalle |
| `fitness.ssim.window_size` | `ssim` | Tamaño de ventana local |
| `fitness.ssim.mse_weight` | `ssim` | Peso del RMSE dentro de SSIM |
| `fitness.blur.sigma` | `blur` | Intensidad del blur gaussiano |
| `fitness.multiscale.scales` | `multiscale` | Resoluciones usadas para comparar |
| `fitness.histogram.bins` | `histogram` | Cantidad de grupos del histograma |
| `fitness.histogram.mse_weight` | `histogram` | Peso del RMSE dentro de histograma |
| `fitness.edge.sigma` | `edge` | Blur previo a detectar bordes |
| `fitness.gradient.sigma` | `gradient` | Blur previo a calcular gradientes |
| `fitness.gradient.orientation_weight` | `gradient` | Peso de orientación vs magnitud |
| `fitness.chamfer.sigma` | `chamfer` | Blur previo a detectar bordes |
| `fitness.chamfer.threshold` | `chamfer` | Umbral para decidir qué píxeles son borde |
| `fitness.saliency.weight` | `saliency` | Refuerzo de zonas salientes |
| `fitness.saliency.sigma` | `saliency` | Suavizado del mapa de saliencia |

## Recomendaciones prácticas

### Para probar que todo anda

Usar una configuración chica. Por ejemplo:

```json
{
  "image": "objetivo.png",
  "output": "results/smoke.png",
  "triangles": 5,
  "seed": 42,
  "progress_every": 1,
  "population": {
    "size": 20,
    "parents": 10,
    "generations": 30
  },
  "fitness": { "metric": "mse" }
}
```

MSE es suficiente para validar el flujo completo.

### Para banderas o logos simples

Probar:

```json
"fitness": { "metric": "mse" }
```

o:

```json
"fitness": { "metric": "multiscale" }
```

Si los colores están mal, sumar histograma en combo:

```json
"fitness": {
  "metric": "combo",
  "combo": { "mse": 0.7, "histogram": 0.3 }
}
```

### Para siluetas, pictogramas o señales

Probar:

```json
"fitness": { "metric": "saliency" }
```

o:

```json
"fitness": {
  "metric": "combo",
  "combo": { "mse": 0.4, "chamfer": 0.3, "saliency": 0.3 }
}
```

Chamfer ayuda a ubicar contornos aunque estén desplazados.

### Para estructura general

Probar:

```json
"fitness": { "metric": "blur" }
```

o:

```json
"fitness": { "metric": "multiscale" }
```

También:

```json
"fitness": {
  "metric": "combo",
  "combo": { "blur": 0.4, "mse": 0.4, "chamfer": 0.2 }
}
```

### Para contornos geométricos

Probar:

```json
"fitness": { "metric": "gradient" }
```

o:

```json
"fitness": {
  "metric": "combo",
  "combo": { "mse": 0.4, "gradient": 0.3, "chamfer": 0.3 }
}
```

Gradient puede ayudar cuando la orientación de líneas importa.

## Cómo leer los valores

Como todas las métricas devuelven valores en `[0, 1]`, se pueden leer con la
misma dirección general: más alto es mejor. Aun así, no conviene comparar
literalmente valores de métricas distintas como si significaran lo mismo
perceptualmente.

Por ejemplo:

```text
MSE fitness = 0.92
Chamfer = 0.12
SSIM fitness = 0.75
```

Ahí sí se sabe que `0.92` es mejor que `0.75` dentro de la misma métrica. Pero
`0.75` de SSIM no necesariamente equivale visualmente a `0.75` de histogram.

Sí tiene sentido comparar valores dentro de la misma métrica:

```text
MSE fitness generación 10 = 0.80
MSE fitness generación 50 = 0.92
```

Ahí sí se puede decir que mejoró.

En `combo`, el resultado depende de los pesos elegidos.

## Costos relativos

De menor a mayor complejidad aproximada:

1. `mse`: muy simple.
2. `regional`: MSE más grilla y pesos.
3. `blur`: MSE más filtro gaussiano.
4. `edge`: Sobel más MSE.
5. `saliency`: Sobel, blur de saliencia y MSE ponderado.
6. `histogram`: histogramas más posible RMSE.
7. `multiscale`: varios resize y varios MSE.
8. `gradient`: Sobel, magnitud, orientación y ponderación.
9. `chamfer`: Sobel, binarización y transformada de distancia.
10. `ssim`: estadísticas locales por canal con ventanas.
11. `combo`: depende de cuántos componentes incluya.

Esto es solo una guía. El costo real depende del tamaño de imagen, cantidad de
individuos, generaciones y hardware.

## Errores comunes de interpretación

### "Fitness más alto es mejor"

En esta versión del TP, eso sí es correcto. Fitness alto significa mejor.

```text
mayor fitness = mejor individuo
```

### "Histograma bueno significa imagen buena"

No necesariamente. Significa que la distribución de colores se parece. Los
colores podrían estar en lugares incorrectos.

### "Edge bueno significa imagen buena"

No necesariamente. Puede tener contornos parecidos y colores internos malos.

### "Blur bueno significa imagen final nítida"

No necesariamente. Significa que las versiones desenfocadas se parecen. Puede
faltar detalle fino.

### "SSIM siempre es más perceptual, entonces siempre es mejor"

No necesariamente. SSIM puede tener puntos débiles en imágenes simples o regiones
lisas. Por eso se mezcla con RMSE.

## Ejemplos completos de configuración

### MSE clásico

```json
{
  "image": "objetivo.png",
  "output": "results/mse.png",
  "triangles": 50,
  "seed": 42,
  "population": { "size": 100, "parents": 50, "generations": 1000 },
  "fitness": { "metric": "mse" }
}
```

### MSE regional

```json
{
  "image": "objetivo.png",
  "output": "results/regional.png",
  "seed": 42,
  "fitness": {
    "metric": "regional",
    "regional": { "grid_rows": 8, "grid_cols": 8, "detail_weight": 2.0 }
  }
}
```

### Bordes tolerantes con Chamfer

```json
{
  "image": "objetivo.png",
  "output": "results/chamfer.png",
  "seed": 42,
  "fitness": {
    "metric": "chamfer",
    "chamfer": { "sigma": 1.0, "threshold": 20.0 }
  }
}
```

### Combo equilibrado

```json
{
  "image": "objetivo.png",
  "output": "results/combo.png",
  "seed": 42,
  "fitness": {
    "metric": "combo",
    "combo": {
      "mse": 0.4,
      "chamfer": 0.25,
      "saliency": 0.25,
      "histogram": 0.1
    }
  }
}
```

## Conclusión

La función de fitness define qué entiende el algoritmo por "mejor imagen".

Si se premia solo color exacto, el algoritmo perseguirá color exacto. Si se
premian bordes, perseguirá bordes. Si se premia distribución global de colores,
puede olvidarse de la ubicación. Por eso es importante elegir una métrica que
coincida con lo que queremos observar en la salida.

Para empezar, `mse` es el baseline más claro. Para mejorar resultados visuales,
lo más razonable suele ser probar `multiscale`, `saliency` o un `combo` que mezcle
MSE con información estructural.
