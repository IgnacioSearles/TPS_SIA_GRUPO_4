# Guía de estudio y defensa del TP4

Revisión del **25/10/2026**, sobre la presentación de `main` después del pull a `ca3229f`. Se revisaron las 90 diapositivas, sus notas y el código de los métodos. Los experimentos locales del 22/10 se preservaron y se distinguen de lo que ya aparece en las diapositivas. La presentación no se modificó.

El objetivo es que puedas hacer cuatro cosas con cada método: explicar qué entra y qué sale, ejecutar mentalmente una actualización, señalar la función que la implementa y defender qué conclusión permiten los experimentos.

Tenés un [mapa de las 96 diapositivas](mapa_estudio_diapositivas.md) y [ejemplos numéricos ejecutables](ejemplos_estudio.py). Desde TP4: `python3 presentation/ejemplos_estudio.py`. Los ejemplos son didácticos y no reentrenan los experimentos ni guardan redes.

## 0. Qué estudiar primero

| Prioridad | Lo que tenés que dominar | Diapositivas |
| --- | --- | --- |
| 1 | Diferenciar entrenamiento, proyección/recuperación y evaluación | Todas |
| 1 | Estandarización; PCA frente a Oja; qué representa PC1 | 4–11, 25–26 |
| 1 | Una actualización de Kohonen; pesos frente a posiciones de grilla | 13–20 |
| 1 | Matriz U, Ward contiguo y selección comercial posterior | 16–23, 28–36 |
| 1 | Hebb, campo local, crosstalk, mezcla y estabilidad | 39–44 y apéndice 78 |
| 1 | Pseudoinversa y desaprendizaje, con diagonal cero | 51–56; ampliación en esta guía |
| 1 | Qué es Hopfield y qué hacen máscaras, Hamming y lectores | 39, 45–59 |
| 2 | Diseño de experimentos, porcentajes comparables y límites | 13–15, 42–44, 49–58 |
| 2 | Por qué menú y vinos difieren tan poco | 47–48 |
| 3 | Detalle de Reed–Solomon, implementación de checkpoints y barridos | Preguntas de ampliación |

No empieces memorizando los nombres de los doce creadores. Primero entendé por qué se elige una región y después se ordena por costo. Tampoco memorices todos los porcentajes de todos los barridos: aprendé los resultados que justifican una decisión y sus condiciones.

### Qué tipo de cosa es cada método

| Método o etapa | Qué aprende o calcula | Qué no hace |
| --- | --- | --- |
| PCA | Direcciones ortogonales de variación, por SVD | No aprende ventas ni grupos comerciales |
| Kohonen / SOM | Prototipos de perfiles organizados en una grilla | No decide por sí solo qué cliente prefiere qué región |
| Ward contiguo | Agrupación de neuronas ya entrenadas | No es otra red neuronal |
| Oja | Una dirección principal, mediante actualizaciones de un vector | No es un ranking universal de conveniencia |
| Hopfield con Hebb | Matriz recurrente de memoria asociativa | Almacenar una matriz no garantiza recuperar cada recuerdo |
| Hopfield con pseudoinversa | Otra regla para construir la misma clase de matriz recurrente | No son varias redes ni una búsqueda por Hamming |
| Hopfield con desaprendizaje | Modificaciones antihebbianas a una red previamente entrenada con Hebb | No sigue entrenándose durante las consultas de evaluación |
| Quitar/restaurar compartidos | Máscara y plantilla conocidas | No aprende los módulos que repone |
| Varias redes por grupos o zonas | Conjunto de redes de Hopfield | El selector o el ensamblado no es la dinámica de una única red |
| Vecino por Hamming | Selección del original con menos diferencias | No hay Hopfield, pesos recurrentes ni convergencia |
| Lector QR | Detección y decodificación de una imagen QR | No es la memoria asociativa del TP |
| Augmentation | Generación de presentaciones de entrenamiento | No cambia el tipo de red ni aporta creadores reales independientes |

## 1. Datos y estandarización

**Diapositivas 4–8. Código:** `src/tp4/preprocessing.py`, `Preprocessor.fit` y `transform`.

Hay 50 creadores, 31 variables numéricas y columnas de identificación. Se excluyen cuatro variables derivadas:

- `price_ars`: proporcional a `price_usd`.
- `organic_posts_pct`: complementaria de patrocinadas.
- `total_engagement_per_1000`: suma de cuatro interacciones.
- `estimated_argentina_reach`: derivada de alcance y audiencia argentina.

Quedan **27 coordenadas por creador**. No se usan logaritmos en la representación final. Excluir el alcance argentino del entrenamiento no impide usarlo después para calcular costos.

Para cada columna se calcula `z = (x − media) / desvío`, con desvío muestral `ddof=1`. La matriz que reciben PCA, Kohonen y Oja es de **50×27**. Las medias y desvíos se ajustan una vez y se reutilizan al proyectar creadores nuevos.

**Por qué:** una diferencia de 100.000 seguidores y otra de 10 puntos porcentuales tienen escalas distintas. Sin escalar, las magnitudes grandes dominan la distancia euclídea. Estandarizar pone cada coordenada en unidades de su desvío.

**Matiz importante:** igual varianza por coordenada no significa igual peso por concepto. Si doce columnas describen tamaño o visibilidad, ese concepto sigue ocupando doce coordenadas correlacionadas. El z-score no elimina redundancia entre columnas.

**Cómo contarlo:** «Quitamos cuentas redundantes y ponemos las variables en una escala comparable. Conservamos tema y geografía aunque expliquen menos variación: importan para el cliente».

**Pregunta probable:** ¿un atípico se elimina? No automáticamente. Los especialistas temáticos aparecen como atípicos en porcentajes porque son pocos; son información útil, no necesariamente errores.

## 2. PCA: qué calcula y cómo se interpreta

**Diapositivas 9–11. Código:** `src/tp4/pca.py`, `compute_pca`.

Sobre datos centrados se hace `X = U S Vᵀ` con `np.linalg.svd`. Las filas de `Vᵀ` son las direcciones principales. Los scores son las coordenadas de cada observación sobre esos ejes: `X @ componentes.T`. Los autovalores se calculan como `S² / (n−1)` y se dividen por su suma para obtener varianza explicada.

El código fija el signo de cada componente para tener una salida reproducible. Matemáticamente, `v` y `−v` son la misma dirección. Cambiar el signo cambia izquierda por derecha, sin alterar la información.

### Ejemplo mínimo

Datos `(1,1), (2,2), (3,3)`. Al estandarizar quedan `(-1,-1), (0,0), (1,1)`. La dirección principal es `(1,1)/√2`, y los scores son `−√2, 0, √2`. Todo varía sobre la misma diagonal; una sola componente conserva el 100 % en este ejemplo.

### Hallazgos del TP

- PC1 explica **49,71 %**: en la orientación presentada, valores negativos se asocian a visibilidad/tamaño y positivos a profundidad/tutoriales/texto/guardados.
- PC1 + PC2 conservan **58,89 %**. El biplot deja fuera más del 40 %: cercanía o ángulos en ese plano son aproximaciones de la estructura completa.
- PC2 mezcla arte/exhibición frente a perros/audiencia argentina. PC3 agrega publicidad y fútbol.
- Diez componentes conservan **91,5 %**. La matriz estandarizada tiene rango 25 por dependencias composicionales, aunque tenga 27 columnas.
- Entrenar Kohonen con PCA 10 produjo **ARI 0,91** frente a las regiones de 27 variables y **12/12 recomendaciones iguales**. No son particiones exactamente idénticas: cambiaron dos creadores de región en la comparación descrita en el informe.

Se mantuvieron las 27 variables por lectura directa de los prototipos y porque el costo con 50 perfiles no era un problema. PCA 10 era una alternativa válida, no una prueba que falló.

**Whitening:** además de rotar, reescala cada componente para igualar varianzas. No se usó; habría cambiado el peso de componentes fuertes y débiles en las distancias.

**Cómo contarlo:** «PCA identifica qué cambia más, no qué conviene comprar. Su eje principal explica estilos de creadores, pero las variables comerciales importantes pueden estar en otros ejes».

## 3. Kohonen: una actualización, de entrada a salida

**Diapositivas 13–20. Código:** `src/tp4/kohonen.py`, especialmente `_train_epoch`, `_step_parameters`, `_neighborhood`, `_update_sample` y `predict`.

Hay **16 prototipos**, cada uno con **27 pesos**: `weights_.shape == (16,27)`. Cada prototipo es un perfil promedio/adaptado. Aparte hay posiciones fijas de esas neuronas en una grilla 4×4. **El vector de 27 pesos se aprende; su posición en la grilla no se aprende.**

Para una presentación `x`:

```python
winner = ((weights - x)**2).sum(axis=1).argmin()
influence = exp(-grid_distance[winner]**2 / (2*radius**2))
weights += learning_rate * influence[:, None] * (x - weights)
```

1. Se buscan parecidos en el **espacio de las 27 variables**: gana el prototipo más cercano.
2. Se calcula influencia usando la **distancia sobre la grilla**, que es otra distancia.
3. Se acerca a la muestra la ganadora y también sus vecinas. Con gaussiana se calcula influencia para todas las neuronas; decrece con la distancia, sin un corte brusco ideal.

La actualización es `w_j ← w_j + η h_j (x − w_j)`. La ganadora tiene `h=1`. Esto coordina neuronas próximas para que representen perfiles próximos; sin vecindad efectiva queda algo más parecido a una cuantización sin organización espacial.

### Ejemplo con dos variables, ya estandarizadas

`x=(1,−1)`, ganadora `w=(0,6;−0,5)`, tasa `η=0,2`:

`w_nueva = (0,6;−0,5) + 0,2[(1;−1)−(0,6;−0,5)] = (0,68;−0,60)`.

Una vecina a distancia 1, con radio 1, recibe `h=exp(−1/2)=0,6065`. Si su prototipo era `(0,0)`, pasa a `(0,1213;−0,1213)`: se mueve menos que la ganadora.

### Cómo transcurre el entrenamiento real

Una época presenta los 50 creadores una vez, en un orden aleatorio. Son 1.000 épocas y 50.000 actualizaciones. Se inicializa seleccionando perfiles de entrenamiento. Tasa 0,3 → 0,01 y radio 2 → 0,5 decaen exponencialmente en las **primeras 500 épocas**; las otras 500 mantienen los valores finales. La parada temprana existe en el código, pero está desactivada en este modelo final.

La inferencia de un creador nuevo no modifica pesos: se escala con la referencia guardada y `predict` devuelve el índice de su neurona ganadora.

**Cómo contarlo:** «Cada perfil atrae al prototipo que mejor lo representa y, en menor medida, a los vecinos de ese prototipo. Por eso obtenemos un mapa ordenado, no dieciséis grupos aislados».

### Gaussiana o uniforme

**Diapositiva 15 y apéndice 68–69. Informes:** [vecindad uniforme](../reports/ejercicio1/uniforme_20261026/README.md) y [versus calibrado](../reports/ejercicio1/versus_vecindades_20261026/README.md). Configuración: `neighborhood='uniform'` en el mismo núcleo.

Con uniforme, toda neurona a distancia `≤ radio` recibe `h=1` y el resto `h=0`. En la grilla la distancia mínima entre neuronas distintas es 1, así que **un radio menor que 1 actualiza solo a la ganadora**. Con el calendario 2 → 0,5 eso ocurre desde la época 251: las últimas 750 épocas no mueven vecinas. Con gaussiana de ancho 0,5, una vecina inmediata todavía recibe `exp(−1/(2·0,5²)) = exp(−2) ≈ 0,135`. El mismo número de radio no significa la misma suavidad.

Hubo dos tandas el 26/10:

1. **Mismas condiciones que el modelo final** (1.260 entrenamientos): uniforme 2 → 0,5 baja QE de entrenamiento (2,65 contra 2,79) pero sube TE a 24 % y empeora en creadores excluidos. Uniforme 2 → 1 ordena casi perfecto (TE 2,5 %) con más QE (3,18).
2. **Versus con igual calibración** (1.890 entrenamientos): diez candidatos por familia (cinco radios × dos tasas), 15 particiones con diez creadores fuera de escalado, entrenamiento y elección. QE externo 3,728 gaussiana y 3,731 uniforme; uniforme gana QE en 8/15 particiones. Uniforme tiene menos TE (24,9 % contra 34,7 %); gaussiana más ARI regional entre semillas (0,760 contra 0,681).

**Cómo contarlo:** «Calibradas igual, las dos cuantizan casi lo mismo en creadores no vistos. Uniforme conserva mejor la topología; gaussiana repite mejor las regiones, y las campañas se apoyan en regiones. Por eso mantenemos gaussiana».

No digas que gaussiana es mejor en general ni que hay equivalencia probada: no se hizo una prueba formal, son 50 creadores y las particiones se superponen. Tampoco que el radio constante es malo: uniforme lo eligió en 8 de 15 particiones y gaussiana en 4.

## 4. Matriz U, regiones y métricas del SOM

### Matriz U

**Diapositiva 16. Código:** `src/tp4/geometry.py`, `Grid.u_matrix`.

Para cada neurona se promedia la distancia entre su prototipo y los prototipos de sus vecinas inmediatas. Un valor alto indica cambios de perfil en su entorno; uno bajo indica prototipos parecidos. No mide cuántos creadores hay, no es un costo comercial y no es la matriz de pesos de Hopfield. El promedio de una celda no identifica por sí solo cuál de sus bordes produce la distancia alta.

Los planos de componentes de las diapositivas 19–20 muestran otra cosa: el valor de una variable en cada prototipo, desestandarizado para leerlo en unidades originales.

### De 16 neuronas a cinco regiones

**Diapositivas 17–18. Código:** `analysis/som_regions.py`, `contiguous_ward` y `_ward_cost`.

Cada neurona empieza como una región. Se consideran pares de regiones vecinas en la grilla y se une el par con menor costo de Ward:

`costo(A,B) = |A||B|/(|A|+|B|) × ||centro_A − centro_B||²`.

`|A|` cuenta **neuronas**, no creadores. Los centros son promedios de prototipos. Se repite hasta quedar en cinco regiones contiguas. En una cadena de prototipos 0, 1, 4, unir 0 y 1 cuesta 0,5; unir 1 y 4 cuesta 4,5. La restricción de grilla limita qué pares se pueden unir.

Ward es un procesamiento posterior, no otra red. Se compararon cantidades de regiones y se eligieron cinco por compromiso entre lectura, tema, tamaños y estabilidad. R1–R5 se renombran por mediana de seguidores en la referencia, no porque el entrenamiento produzca nombres comerciales.

### Qué significa cada métrica

| Métrica | Cómo se calcula | Cómo se puede malinterpretar |
| --- | --- | --- |
| QE | Media de distancia al prototipo ganador | Baja al aumentar capacidad; no demuestra mejores campañas |
| TE | Fracción cuyo primer y segundo ganador no son vecinos | La adyacencia cambia entre rectangular y hexagonal |
| Ocupación | Creadores asignados a cada neurona | Una celda vacía no invalida sola todo el mapa |
| ARI | Concordancia de particiones ajustada por azar | No es exactitud contra grupos verdaderos; puede ser negativo |
| Pureza | Suma de los tamaños del tema mayoritario de cada región, dividida por 50 | Aumenta al fragmentar; el tema ya está en las entradas |
| Firmeza | Frecuencia con que un creador sigue con sus compañeros de región de referencia | No es probabilidad de éxito comercial |
| η² | Variación entre regiones / variación total de una variable | Asociación descriptiva, sin causalidad |
| V de Cramér | Asociación entre región y tema a partir de la tabla de contingencia | No dice qué criterio comercial es mejor |

En el código de ARI se arma una tabla de contingencia `n_ij` que cuenta cuántos creadores están en cada cruce de grupos. Se cuentan parejas con `C(n,2)=n(n−1)/2`: `a=Σ_ij C(n_ij,2)`, `r=Σ_i C(n_i.,2)`, `c=Σ_j C(n_.j,2)` y `T=C(n,2)`. Entonces `ARI=(a−rc/T)/[(r+c)/2−rc/T]`, con manejo del caso de denominador cero. Cambiar nombres de grupos no modifica esa tabla de coincidencias.

La referencia `E[η²] ≈ (k−1)/(n−1) = 4/49 ≈ 0,082` corresponde a asignaciones aleatorias bajo ese control, no a un umbral universal de relevancia.

### Qué hallamos

Se informaron 300 corridas de geometría, 360 de representación y 840 de tasa/radio: 1.500 corridas de configuraciones, no 1.500 conjuntos independientes. Aumentar el mapa reduce QE y fragmenta estos 50 perfiles. Se mantuvo 4×4 por compromiso entre ocupación, estabilidad y lectura. La comparación de vecindad uniforme y gaussiana del 26/10 suma 3.150 entrenamientos más; está resumida en [Gaussiana o uniforme](#gaussiana-o-uniforme).

El ARI 0,83–0,91 de geometría se refiere a asignaciones a **neuronas** en determinadas configuraciones; el 0,79 del relato de regiones se refiere a las **cinco regiones**. No los presentes como si fueran la misma medición.

Las regiones separan tamaño (η²≈0,71), descubrimiento y tutoriales (≈0,47–0,48); separan poco audiencia argentina (0,08) y costo (0,02). Por eso se necesita una decisión comercial posterior.

## 5. Oja: por qué encuentra PC1 y qué hace el código

**Diapositivas 25–26. Código:** `src/tp4/oja.py`, `_update_sample`, `_normalize`, `fit` y `transform`.

Oja tiene **un vector de 27 pesos**, no un mapa ni cinco regiones. Sobre una muestra centrada:

```python
y = sample @ weights
weights += learning_rate * y * (sample - y * weights)
```

La parte `η y x` refuerza coordenadas que acompañan a la salida. La parte `−η y² w` controla el crecimiento. En promedio, la dinámica favorece la dirección de mayor varianza de la covarianza. Se agrega normalización explícita al final de cada época por protección numérica; no significa que la norma sea exactamente uno después de cada muestra.

### Ejemplo

Con `x=(2,1)`, `w=(1,0)` y `η=0,1`, resulta `y=2` y:

`Δw = 0,1×2×[(2,1)−2(1,0)] = (0;0,2)`.

El nuevo vector es `(1;0,2)`. Al normalizarlo queda aproximadamente `(0,9806;0,1961)`.

Se centra la entrada, se inicializa un vector unitario aleatorio y se presentan muestras en orden mezclado. Se eligió tasa 0,001 → 0,00001 durante 1.000 épocas. A diferencia del SOM elegido, este decaimiento se reparte sobre **las 1.000 épocas**.

**Detalle de código que puede preguntarte el docente:** `fit` también calcula PCA, pero lo usa como referencia para registrar alineación y varianza. `_update_sample` no recibe PC1 ni la copia. La regla aprende con muestras; la comparación por SVD evalúa si aprendió la dirección esperada.

Para pesos unitarios y datos centrados, la actualización esperada se escribe `E[Δw]=η[Cw−(wᵀCw)w]`, donde C es la covarianza salvo su factor de normalización. Un autovector es una dirección estacionaria; bajo las condiciones de aprendizaje y un autovalor dominante, la dirección principal es la atractora relevante. Esto conecta las operaciones de `_update_sample` con PCA, sin convertir cualquier tasa finita en una garantía de convergencia.

Se compara `|w·PC1|` con vectores normalizados porque el signo es arbitrario. El índice es la proyección `y=(x−media)·w`. En el catálogo reprodujo PC1, sus scores y su orden, con alineación cercana a `1−2×10⁻¹⁰`.

**Por qué no elegir al de índice máximo:** con la orientación actual significa más profundidad; con la orientación opuesta significa más visibilidad. Ninguna orientación incorpora por sí sola el objetivo, el tema, el costo o el público local.

**Cómo contarlo:** «Oja aprende online el mismo eje dominante que PCA calcula por álgebra. Ese eje describe estilo, no calidad comercial universal».

## 6. Cómo se convierten estos resultados en campañas

**Diapositivas 28–36. Código:** `analysis/som_campanas.py`, `TARGET_REGIONS`, `strategy_selections` y `substitutes`.

El flujo completo es:

`datos → seleccionar/escalar → SOM → Ward contiguo → interpretar regiones → elegir regiones por objetivo → filtrar tema → ordenar por costo`.

La tabla `TARGET_REGIONS` expresa una **decisión humana explícita**: fútbol y maquillaje R4/R5; perros R1; arte R3. El SOM no aprende objetivos comerciales ni completa esa tabla por optimización de ventas.

Después se calcula `CPM_AR = 1.000 × precio_USD / alcance_argentino_estimado` y se eligen los tres menores dentro de los candidatos del tema y regiones objetivo. Por ejemplo, USD 100 para 5.000 cuentas argentinas estimadas da USD 20 por mil.

- **Fútbol:** se busca descubrimiento con público argentino; seguidores y alcance a no seguidores están del lado visible de PC1 en este catálogo.
- **Maquillaje:** alcance con audiencia local, admitiendo tutoriales como complemento.
- **Perros:** comunidad/profundidad y audiencia local; USD 339 por las tres publicaciones frente a USD 6.993 del top por seguidores.
- **Arte:** demostración/exhibición, con una tensión clara respecto del público argentino; es el caso más débil.

Las reducciones de costo por mil del 15–47 % son comparaciones con el top tres por seguidores, usando los indicadores del catálogo. No son incrementos medidos de ventas o adopciones. Tampoco sabemos cuánto se superponen las audiencias entre creadores: el alcance agregado no es necesariamente público único.

Para creadores nuevos se conserva el escalador y se busca BMU. Con muchas observaciones nuevas o cambios de variables/distribución se revisa y eventualmente reentrena todo. Las regiones pueden rotar o renumerarse; los nombres R1–R5 no son identidades permanentes que puedan copiarse sin revisar.

## 7. Hopfield con Hebb: almacenar y recuperar son cosas distintas

**Diapositivas 39–44 y apéndice 79 (fórmulas). Código:** `src/tp4/hopfield.py`, `__init__`, `_synchronous`, `recall`, `energy` y `overlaps`.

Un módulo QR es un cuadrado lógico; no cada píxel de una foto. Se usan imágenes binarias de **37×37 módulos**, aplanadas por filas. Negro es `+1`, blanco `−1`. Los cinco recuerdos forman `Ξ` de **5×1.369**.

### Almacenamiento de Hebb

```python
weights = patterns.T @ patterns / neurons
np.fill_diagonal(weights, 0)
```

`W` tiene forma **N×N**, es simétrica y no tiene autoconexiones. Si dos neuronas tienen el mismo signo en muchos recuerdos se refuerza su conexión positiva; si suelen tener signos opuestos, la conexión es negativa.

La matriz de Kohonen era neuronas×variables. Esta matriz es neuronas×neuronas y define una dinámica recurrente. Hebb y pseudoinversa calculan sus pesos algebraicamente; no tienen las 1.000 épocas del SOM.

### Recuperación sincrónica

```python
fields = W @ state
new_state = np.where(fields > 0, 1., np.where(fields < 0, -1., state))
```

Cada campo `h_i` suma los aportes de las otras neuronas. Todos los campos se calculan con el estado anterior y se actualizan juntos. Si un campo da cero se conserva el valor anterior: puede ocurrir por cancelación exacta de aportes y no hay que transformarlo en un tercer valor 0.

La recuperación termina por estado estable, ciclo o tope. `iterations` cuenta pasos que cambiaron el estado; la comprobación final de estabilidad no agrega uno. Un recuerdo perfecto que ya es estable requiere cero cambios. Un estado estable también puede ser equivocado.

Se usa toda la matriz en `W @ state`, incluidos ambos triángulos simétricos. Cada fila calcula el campo de una neurona distinta. En la energía hay un factor 1/2 para no contar dos veces cada pareja.

### Solapamiento y crosstalk

`m(A,B) = A·B/N = 1−2 d_H(A,B)/N`.

- `m=1`: idénticos.
- `m≈0`: coinciden aproximadamente en la mitad, casi ortogonales.
- `m=−1`: inversos. Negativo no significa ausencia de parecido: significa alineación opuesta.

Para consultar un recuerdo `ξ^ν`, el campo de Hebb con diagonal cero es exactamente:

`h_i = (1−p/N) ξ_i^ν + Σ_(μ≠ν) m(μ,ν) ξ_i^μ`.

El segundo término es el crosstalk. El desarrollo simplificado «señal 1 + crosstalk» omite el pequeño `p/N` de la diagonal eliminada. `Hopfield.crosstalk` calcula ese segundo término; el campo exacto se calcula con `weights_ @ state`.

No basta que el crosstalk medio absoluto sea grande: en módulos comunes empuja a favor. El error aparece cuando cambia el signo del campo respecto del recuerdo, es decir `ξ_i h_i < 0`. Si vale cero, la regla conserva el signo previo.

**Verificado con el código actualizado:** neuronas que cambian signo al consultar cada original completo: **71, 91, 225, 225, 103**. Sin compartidos: **16, 18, 29, 34, 33**. Por eso ninguno de los cinco originales es estable con Hebb en ninguna de esas dos representaciones.

### Energía, mezcla y cuencas

`E(s) = −(1/2) sᵀWs`. Para Hebb y diagonal cero también:

`E(s) = −N/2 × Σ_μ m(s,ξ^μ)² + p/2`.

El `p/2` es constante para una misma red; omitirlo no cambia qué estado tiene menor energía. Con el menú: **−2.077,45**; con la mezcla: **−2.172,88**. La mezcla se parece bastante a todos y gana en la suma de solapamientos cuadrados, aunque no sea ninguno.

La garantía usual de energía no creciente usa actualizaciones de una neurona por vez. La implementación es sincrónica: puede tener ciclos y aumentos de energía. Que no aumente en un barrido es un resultado observado, no un teorema sobre este código.

Una cuenca de atracción es el conjunto de consultas que termina en un estado. Un original estable no garantiza una cuenca amplia ni que una consulta dañada llegue a él.

**Cómo contarlo:** «Almacenar con Hebb no significa que cada QR se haya convertido en un recuerdo estable. Los aportes cruzados de otros QR pueden dominar justamente los módulos que identifican cuál es».

## 8. Por qué estos QR interfieren tanto

**Diapositivas 43 y 47–48. Código:** `analysis/qr_bits.py`; imágenes cargadas por `src/tp4/qr.py`.

De los 1.369 módulos, **955 son iguales en los cinco QR** y solo **414** cambian entre identidades. El solapamiento completo entre pares va aproximadamente de 0,58 a 0,86. Casi el 70 % común incluye estructuras fijas y datos compartidos: no todo ese 70 % es una obligación estructural de cualquier QR.

Las URL tienen un prefijo común de **72 caracteres**. Menú y vinos tienen igual largo, 85 caracteres, y cambian `food` por `wine`. Sus caracteres ASCII difieren en **6 de los 32 bits** de esas cuatro letras. Mantener el largo conserva alineado lo que sigue; otras URL del conjunto tienen otros largos y desplazan el flujo posterior.

El script reconstruye el QR: modo byte y largo → bytes del mensaje → terminación/relleno → corrección Reed–Solomon → ubicación de bits en la matriz → máscara y formato. Para estos datos, **versión 5, nivel L, máscara 0** reconstruyen exactamente las cinco imágenes, módulo por módulo. El script es específico de este caso; no es un generador general de todos los QR.

La región variable contiene **220 módulos de datos y 194 de corrección de errores**. Menú y vinos difieren en **6 bits de datos y 90 de corrección**: 96 diferencias. En 414 posiciones eso da `m = 1−2×96/414 ≈ 0,536`, y en la imagen completa `m = 1−2×96/1.369 ≈ 0,860`.

**Qué estudiar de Reed–Solomon:** añade redundancia calculada a partir del mensaje, por eso un cambio pequeño puede producir muchos cambios de corrección. No necesitás empezar memorizando la aritmética de `gf_mul`; sí saber que `rs_remainder` calcula bytes de corrección y que esos bytes también se convierten en módulos que la red ve.

### El 25 % compartido y la capacidad

La explicación del control artificial aproxima el solapamiento por la fracción común `q`. En una neurona en la que los otros cuatro recuerdos empujan contra el propio, el campo alineado se aproxima a `1−4q`. Eso anticipa un cambio alrededor de `q=1/4`. Hay que considerar fluctuaciones, patrones concretos y la pequeña corrección de diagonal: **no es un umbral universal que dependa solo del porcentaje común**.

La capacidad `≈0,15N` del apunte supone patrones adecuados, típicamente aleatorios y casi ortogonales, y un determinado criterio de recuperación. No garantiza 205 QR arbitrarios ni exactitud perfecta. Con cinco patrones aleatorios del mismo tamaño la implementación recupera mucho mejor: es un control que apoya el diagnóstico de correlación, no una demostración de ausencia de cualquier bug.

**Cómo contarlo:** «El problema no es tener cinco recuerdos en abstracto; es que esos cinco recuerdos se parecen demasiado. Quitar lo común reduce una causa del parecido, pero deja un par especialmente parecido».

## 9. Pseudoinversa, explicada con A, B y C

**Diapositivas 55–57, apéndice 79–80 y material de respaldo. Código:** rama `rule == 'pseudoinverse'` de `Hopfield.__init__`.

La matriz de patrones sigue teniendo recuerdos por filas. Se calcula:

```python
gram = patterns @ patterns.T
projector = patterns.T @ np.linalg.pinv(gram) @ patterns
weights = projector.copy()
np.fill_diagonal(weights, 0)
```

Si los patrones fueran ortogonales, `gram=N I` y su pseudoinversa sería `I/N`: se recupera la misma fórmula que Hebb. Con correlaciones, los términos fuera de la diagonal de Gram hacen que la compensación sea diferente.

Para cinco QR, `gram` es de **5×5**, no de 1.369×1.369. Su pseudoinversa compensa dependencias entre recuerdos antes de armar los pesos. `pinv` también permite manejar rangos deficientes.

Antes de quitar la diagonal, `P` es el proyector ortogonal sobre el espacio generado por los recuerdos. Para un recuerdo almacenado `ξ`, `Pξ=ξ`. Después de quitarla, `W=P−diag(P)` y el campo sobre ese recuerdo es `h_i=(1−P_ii)ξ_i`. Como la diagonal de un proyector está entre cero y uno, no se invierte el signo; si hay empate, el código conserva el estado. Esto explica la estabilidad ideal de los originales, no su recuperación desde cualquier ruido.

**Por qué al final:** la proyección se calcula con la fórmula completa y después se impone la convención de Hopfield sin autoconexiones. No se resta uno a cada elemento de la diagonal: se pone cada entrada diagonal en cero. Se podrían calcular solo las entradas fuera de la diagonal, con la misma fórmula; no cambia el resultado. Tras ese paso, `W` ya no es en general un proyector.

### Los mismos tres recuerdos en una única red

Usamos seis neuronas, solo para hacer visibles las cuentas:

```text
A = (+1,+1,+1,+1,−1,−1)
B = (+1,+1,+1,−1,+1,−1)
C = (+1,+1,−1,+1,+1,−1)
```

Para comparar reglas construimos dos modelos de comparación, cada uno almacena **los tres juntos**: uno usa Hebb y otro pseudoinversa. La pseudoinversa no construye una red por recuerdo.

La matriz de Gram es:

```text
G = [[6,2,2],       G⁻¹ = [[ 0.20,-0.05,-0.05],
     [2,6,2],              [-0.05, 0.20,-0.05],
     [2,2,6]]              [-0.05,-0.05, 0.20]]
```

Con diagonal cero, los pesos resultantes son:

```text
Hebb = (1/6) ×             Pseudoinversa = (1/10) ×
[[ 0, 3, 1, 1, 1,−3],     [[ 0, 3, 1, 1, 1,−3],
 [ 3, 0, 1, 1, 1,−3],      [ 3, 0, 1, 1, 1,−3],
 [ 1, 1, 0,−1,−1,−1],      [ 1, 1, 0,−3,−3,−1],
 [ 1, 1,−1, 0,−1,−1],      [ 1, 1,−3, 0,−3,−1],
 [ 1, 1,−1,−1, 0,−1],      [ 1, 1,−3,−3, 0,−1],
 [−3,−3,−1,−1,−1, 0]]      [−3,−3,−1,−1,−1, 0]]
```

Al presentar A perfecto:

```text
campo con Hebb = (1,1667;1,1667;0,5;0,5;+0,1667;−1,1667)
campo con pinv = (0,7;   0,7;   0,3;0,3;−0,3;   −0,7)
```

La quinta neurona de A debería ser negativa. Hebb le da campo positivo y la cambia; pseudoinversa le da campo negativo y la conserva. En la proyección previa a borrar la diagonal, esa quinta entrada diagonal era 0,7: por eso su campo final es `(1−0,7)×(−1)=−0,3`.

Ahora invertimos la primera neurona de A y usamos la misma consulta en ambos:

```text
consulta  = (−1,+1,+1,+1,−1,−1)
Hebb:      consulta → A → (+1,+1,+1,+1,+1,−1), mezcla estable
pinv:      consulta → A, que ya es estable
```

**Qué demuestra este ejemplo:** corregir momentáneamente una consulta no alcanza si el recuerdo original no es un punto fijo. La nueva regla cambia los pesos y las cuencas. No demuestra que pseudoinversa gane siempre: con patrones ortogonales coincide con Hebb; en otros conjuntos el tamaño de las cuencas depende de datos y regla.

### Completa frente a reducida

Quitar posiciones comunes cambia la matriz de Gram y la dinámica. Aunque no distingan identidades, pueden modificar los campos y la tolerancia al ruido. Por eso la pseudoinversa completa puede rendir mejor que la reducida. No se contradice con que ambas dejen estables los originales.

## 10. Desaprendizaje: qué se penaliza y cuándo se termina

**Diapositivas 51–54. Código:** `src/tp4/unlearning.py`, `UnlearningTrainer.step` y `anti_hebbian_update`.

Se parte de **una red de Hebb** y se hacen intentos de encontrar estados espurios. Si un intento llega a un estado estable `a` que no es un original ni su inverso, se aplica:

`W ← W − η/N × aaᵀ`, y después se vuelve a dejar la diagonal en cero.

Como matriz fuera de diagonal, el cambio es `−η/N × (aaᵀ−I)`. Se conserva simetría. Se debilitan conexiones que sostienen a ese estado; cambia el paisaje de toda la red, no solo una celda de memoria.

La energía de `a` aumenta exactamente `η(N−1)/2`. Por ejemplo, con seis neuronas y `η=0,1`, aumenta 0,25. **Eso no garantiza recuperar el original**: otros estados también cambian y algunos recuerdos pueden desestabilizarse. Por eso hay un requisito experimental de conservar estables los cinco originales.

### Dirigido y aleatorio

| Modo | Desde dónde arranca cada intento | Qué se penaliza |
| --- | --- | --- |
| `directed` | La mezcla mayoritaria de los recuerdos originales | El espurio estable al que llegue la red actual, no necesariamente la mezcla inicial |
| `dream` | Un vector aleatorio uniforme de ±1 | El espurio estable encontrado desde ese inicio |

En ambos se ejecuta `recall`, se inspecciona el motivo de parada y se decide si actualizar. Ciclos, topes y originales/inversos protegidos consumen un intento, pero no modifican pesos. Proteger a un original de ser penalizado directamente no impide que penalizar otro estado lo afecte.

### Qué se eligió y por qué 50

Se probaron modos, tasas 0,01/0,05/0,1 y presupuestos 10/50/100. Primero se exigió estabilidad de los cinco originales en todas las repeticiones; después se comparó recuperación exploratoria y se congeló una configuración para consultas nuevas.

Se eligió **aleatorio, tasa 0,1, 50 intentos, representación de 414 neuronas**. Hubo **20, 22 y 20 actualizaciones efectivas**, según la semilla. Cincuenta no es una cantidad óptima universal ni una parada automática por precisión: es el presupuesto seleccionado de esa exploración. En ella 50 dio 88,4 % y 100 dio 86,0 %. El 90,5 % es la confirmación del candidato de 50, con otras consultas; no es la misma tabla.

No se enseña a la red usando las consultas de evaluación. En la confirmación los pesos permanecen fijos. Las tres semillas son tres repeticiones, cada una con su única red; no se vota entre sus salidas.

La versión completa de 1.369 neuronas **no tuvo candidato que conservara los cinco originales en todas las semillas**. Ese resultado negativo es parte de lo aprendido, no algo para omitir.

**Cómo contarlo:** «En vez de reforzar otro recuerdo, dejamos que la red encuentre recuerdos falsos y debilitamos los estados estables falsos. Después comprobamos que los recuerdos verdaderos sigan estables».

## 11. Compartidos, redes por zonas, reparto y Hamming

### Quitar y restaurar compartidos

**Diapositiva 45. Código:** `src/tp4/shared_modules.py`.

`differing = ~(patterns == patterns[0]).all(axis=0)` marca posiciones que cambian en algún recuerdo. `strip` conserva esas 414 coordenadas. `restore` copia una plantilla conocida de 1.369 y coloca ahí la salida de la red. Las 955 restantes se reponen directamente.

La consulta ruidosa se genera completa antes del recorte. La red recibe solo la parte variable. Si el daño cae en un módulo común, lo corrige la plantilla, sin inferencia de Hopfield. Eso debe contarse como parte explícita de la solución.

Con cinco QR, quitar compartidos **no bastó para Hebb**: cero originales estables. Mejoró qué subconjuntos de tres pueden almacenarse. Por eso la diapositiva 60 dice que sacar lo común reduce el primer problema, no que lo resuelve solo.

### Una red por zona

**Diapositiva 45 (mencionada) y apéndice 88–89. Código:** `src/tp4/zoned_hopfield.py`.

Se divide espacialmente el QR. Cada zona almacena los recortes de los cinco recuerdos; si todos son iguales ahí, se repone la zona conocida sin red. Se recupera cada zona y se ensamblan salidas. Zonas distintas pueden corresponder a identidades distintas o a espurios: que cada pieza parezca razonable no garantiza un QR completo válido. El parecido local siguió causando problemas.

### Dos redes por grupos de recuerdos

**Diapositiva 49, presentada como diagnóstico. Código:** `src/tp4/split_hopfield.py`; selección en `analysis/hopfield_qr_mejoras.py`, `best_partition`.

Aquí no se cortan posiciones: se reparten **identidades** entre redes. Se enumeran las diez ternas y pares complementarios y se minimiza el máximo **valor absoluto del solapamiento** dentro de los grupos, usando solo los recuerdos almacenados. Resultado: menú/cócteles/postres en una y vinos/especiales en otra, con módulos variables y Hebb.

La consulta pasa por ambas. Se restauran salidas completas y se comprueba cuáles son recuerdos de la propia red. Si hay alguna, se elige entre esas; si hay más de una, se usa menor Hamming a la consulta. Si ninguna es un recuerdo almacenado, se elige la salida más cercana entre todas.

Esto es **redes de Hopfield + selector externo**. El selector conoce los recuerdos y no forma parte de la dinámica recurrente. El código usa Hamming, aunque usar dos redes no obliga teóricamente a elegir ese criterio. Sus resultados hasta 35 % de ruido pertenecen al barrido correspondiente; no están emparejados con las consultas nuevas del gráfico de la diapositiva 57.

**Por qué en la presentación es un diagnóstico y no una solución.** Si la consulta va directo a la red que guarda su QR, recupera el 100 % hasta 35 % de ruido: eso prueba que el par menú-vinos era el único obstáculo. Pero con una consulta de menú las dos redes devuelven un QR guardado (la de vinos, vinos: 140 de 140), así que alguien tiene que elegir. Sin desempate el total queda en 80 %. Desempatar con la salida más cercana a la consulta equivale a Hamming y nunca le gana (6 consultas mejor, 25 peor). Elegir la red de menor energía no mira los originales, pero con Hebb `E = −N/2 Σ m² + p/2`: sigue midiendo el parecido con los QR guardados, y queda por debajo (3 mejor, 43 peor). Con cinco redes de un QR cada una, el método sería exactamente comparar con los originales. **Cómo contarlo:** «Separar el par confirma que es el obstáculo; elegir la red nos devuelve a comparar con los originales».

### Vecino más cercano por Hamming, sin red

Cuenta coordenadas distintas y devuelve un original:

```python
distances = np.count_nonzero(patterns != query, axis=1)
answer = patterns[distances.argmin()]
```

No ajusta pesos ni recorre estados intermedios. Además, las coordenadas comunes suman la misma cantidad de diferencias a todos los originales para una consulta dada. Por eso **Hamming completo y recortado dan el mismo orden de candidatos** (con el mismo desempate), mientras Hopfield completo y recortado pueden tener dinámicas distintas.

Hay tres usos que no se deben confundir: Hamming como métrica del error final, Hamming como selector entre salidas de varias redes y Hamming como método completo sin red.

## 12. Qué hallazgos defender y qué porcentaje usar

### Resultado base

**Diapositivas 40–41.** Once niveles de ruido 0–50 %, cinco QR y veinte semillas por nivel: **1.100 consultas**. Hebb completo recuperó **0/1.100**, incluso sin ruido. Hubo 1.051 salidas en la mezcla y 49 en su inverso. Caer rápido a un estado de baja energía no implica recuperación correcta.

El lector decodifica los cinco originales y ninguna salida de ese barrido. Poder leer y haber reconstruido exactamente el original son métricas distintas.

### Comparación emparejada nueva

**Diapositiva 57.** Se usan las mismas **700 consultas**: cinco QR, siete niveles `0/10/20/30/35/40/45 %` y veinte semillas 100–119. El promedio siguiente cuenta solo los seis niveles positivos, con igual peso por nivel:

| Método | Recuperación media |
| --- | ---: |
| Hebb reducido | 0 % |
| Pseudoinversa reducida + restauración | 89,5 % |
| Hebb con desaprendizaje reducido + restauración | 90,5 % |
| Pseudoinversa completa | 93,3 % |
| Vecino por Hamming sin red | 95,7 % |

Entre las redes de esta tabla gana pseudoinversa completa. La diferencia de un punto entre las dos reducidas **no es concluyente**: intervalo emparejado de desaprendizaje menos pseudoinversa reducida, **−0,2 a +2,3 puntos**. Frente a la completa, la diferencia fue −2,8 puntos, con intervalo **−4,6 a −1,4**, condicionado a esos cinco QR y a las redes entrenadas.

Tres redes de desaprendizaje evaluadas con las mismas 700 consultas producen 2.100 evaluaciones, no 2.100 consultas independientes. El bootstrap remuestrea semillas de consulta como bloques, manteniendo juntos QR, niveles y métodos. No demuestra generalización a otras colecciones de QR.

«Ninguna red gana a Hamming» debe referirse a la **media y métodos de esta comparación**. No significa que Hamming gane en cada consulta o en cada tipo de daño. Las dos redes de Hebb tuvieron otro barrido y no entran en esa tabla emparejada.

**Por qué Hamming es el techo con este ruido.** Si cada módulo se invierte por separado con la misma probabilidad p < 0,5, la probabilidad de ver la consulta dado un original es `p^d (1−p)^(N−d)`, con `d` su distancia de Hamming: decrece con `d`. Entre cinco QR igual de probables, el más cercano es el más probable (máxima verosimilitud), así que ningún método puede ganarle en promedio; uno puede acertar alguna consulta suelta que Hamming falla, por azar. Lo que miden las redes es cuánto se acercan a ese techo sin guardar los QR aparte y recuperando solo con su dinámica. Con daño que solo borra tinta el techo cambia (ver daño realista).

### Daño espacial y lector

**Apéndice 92.** Un bloque 11×11 cubre 121 módulos, pero blanquearlo solo altera los que antes eran negros. En el ejemplo promediado hay 65 cambios y solo nueve caen en posiciones variables que ve la red; el resto lo repone la plantilla.

Con el candidato reducido confirmado, esquinas blancas hasta 21×21 y manchas negras hasta 17×17 dieron 100 % en los casos ensayados. A 29×29 hubo fallos: 75 % en esquinas y 91 % en manchas. Esta tanda midió recuperación exacta de módulos; no ejecutó un lector óptico ni evaluó fotos con perspectiva, desenfoque o detección fallida.

La corrección QR por Reed–Solomon trabaja con **codewords de ocho bits**. Los porcentajes de los niveles no son garantías literales de tolerar cualquier porcentaje de píxeles o módulos invertidos. Fuente del fabricante: [DENSO WAVE](https://www.qrcode.com/en/about/error_correction.html).

### Daño realista

**Diapositiva 58 y apéndice 95–97. Código:** `src/tp4/damage.py`, `analysis/hopfield_qr_dano_realista.py`. Informes: [niveles moderados](../reports/ejercicio2/dano_realista_20261026/README.md) y [extremos](../reports/ejercicio2/dano_realista_extremo_20261026/README.md).

Siete daños que imitan un QR impreso gastado: desgaste (negro → blanco, nunca al revés), desgaste mayor en un borde, manchas negras y blancas irregulares, rayones que borran tinta, desgaste con mancha y una foto gastada (poco contraste, desenfoque y mancha semitransparente sobre una imagen de 8 píxeles por módulo, binarizada después). Se usaron los métodos congelados y semillas nuevas 200–219: 7.800 consultas en dos corridas. La segunda, con niveles extremos, se agregó después de ver la primera.

- **Lector solo:** no lee nada desde 5 % de módulos alterados; con desgaste, ni con 2,6 %.
- **Desaprendizaje:** 97 % o más con hasta un cuarto de los módulos alterados en los siete tipos; se rompe entre 40 y 47 %. La foto es lo que más cuesta: 87 % con 40 %.
- **Pseudoinversa completa:** aguanta más, 99 % con 44 % por desgaste.
- **Hebb:** 0 %.
- **Techo:** con daño que solo borra tinta, un negro en la consulta no puede venir de un blanco del original. Usar eso da 97 % con 49 % alterado, contra 71 % de Hamming. Hamming es el techo solo con ruido simétrico.

**Cómo contarlo:** «El problema práctico es el lector: un poco de daño ya le impide leer, y una memoria que guarda los cinco QR lo devuelve exacto».

No digas que son fotos reales: son simulaciones, y la consulta sigue llegando alineada a 37×37.

## 13. Experimentos locales nuevos que todavía no están en las diapositivas

La comparación de vecindades del 26/10 sí entró: diapositiva 15 y apéndice 68–69, explicada en [Gaussiana o uniforme](#gaussiana-o-uniforme).

Conviene conocerlos como respaldo y distinguirlos del relato publicado. No hubo que descartar resultados del compañero: los informes subidos son la referencia histórica, y la ausencia local de sus pesos no prueba que una corrida no se hiciera.

### Confirmación de Kohonen y Oja, 460 corridas del 22/10

- Treinta semillas nuevas por finalista: 4×4 rectangular mantuvo menor QE medio (2,789 frente a 2,852 y 2,849) y ARI de regiones entre semillas 0,811. Las alternativas hexagonales fueron algo más estables, sin dominar todos los criterios.
- Cien submuestras 40/50 con tres semillas: ARI medio entre semillas de la misma submuestra 0,821, pero contra referencia 0,608. **La composición del catálogo importa más de lo que sugiere repetir semillas sobre las mismas cincuenta personas.**
- Arte conservó el trío completo en **18,3 %** de corridas de submuestras; perros en **88,3 %**. Son medidas descriptivas con alineación de regiones, no validación comercial independiente.
- Oja con tasa 0,001 alcanzó alineación 0,999 en 7–13 épocas en diez semillas nuevas. Retirar los uno o tres perfiles con más seguidores cambió poco PC1; eso no estudia todos los tipos de atípicos.

Fuente: [confirmación del ejercicio 1](../reports/ejercicio1/confirmacion_20261022/README.md).

### Augmentation, 240 entrenamientos del 22/10

Se interpoló hacia vecinos del mismo tema con máximos 5/15/30 %, manteniendo 40.000 actualizaciones por modelo y evaluación sobre diez originales excluidos. La mejora media de QE fue **0,002/0,680/1,053 %**, con mejoras en **11/20, 14/20 y 12/20** particiones, respectivamente.

Ninguna variante pasó todos los criterios predeclarados. Se mantuvo el entrenamiento original. Con 30 % aumentaron tanto los tríos completos de arte como los casos con menos de tres candidatos, que pasaron de 7/60 a 14/60: no fue una solución consistente a ese problema. No es una refutación de toda augmentation, sino un resultado de esta familia y protocolo.

Fuente: [augmentation de Kohonen](../reports/ejercicio1/augmentation_20261022/README.md).

## 14. Frases de la presentación que requieren matices

Estas son observaciones para estudiar y defender; no se cambiaron las diapositivas.

| Diapositiva | Evitá interpretar literalmente | Forma más precisa de explicarlo |
| --- | --- | --- |
| 7 | «Cada variable pesa lo mismo» = conceptos equilibrados | Misma varianza por coordenada; quedan bloques correlacionados |
| 10, 28, 31 | Correlaciones o costos prueban ventas/adopciones | Son asociaciones e indicadores del catálogo; no se midieron conversiones |
| 11 | PCA 10 da exactamente las mismas regiones | ARI 0,91, dos cambios de región y 12/12 recomendaciones iguales |
| 13–18 | Todos los ARI comparan lo mismo | Distinguir asignaciones a neuronas, cinco regiones y configuración/semillas |
| 22 | Los temas se descubrieron sin información temática | Los porcentajes temáticos entran al SOM; no hay etiquetas objetivo de grupos |
| 27, 30 | El modelo eligió automáticamente el objetivo del cliente | La región objetivo se decidió interpretando perfiles y se codificó en `TARGET_REGIONS` |
| 39, 59 | La energía de actualización sincrónica siempre baja | Bajó en los casos observados; la garantía usual corresponde al caso asincrónico |
| 42 | Un control aleatorio descarta cualquier bug | Apoya la explicación por correlación junto con los otros controles |
| 42 | 0,15N garantiza esa cantidad de QR | Es una referencia bajo supuestos de patrones/criterio distintos |
| 44 | Campo exacto = 1 + crosstalk; 25 % universal | Hay corrección de diagonal y una aproximación sobre interferencia en el control |
| 43 | Todo conjunto de QR de un sitio necesariamente falla | Estos cinco comparten formato, máscara y mucho mensaje; otros diseños pueden diferir |
| 51–52 | Desaprender protege automáticamente los originales | Se excluyen como objetivos directos y después se verifica su estabilidad |
| 56 | Todos los métodos, incluida la solución de dos redes, se compararon ahí | La tabla usa consultas nuevas emparejadas; las dos redes tienen otro barrido |
| 58, apéndice 92 | Área dañada = bits invertidos; nivel H tolera cualquier 30 % de módulos | Importan color previo, ubicación, codewords y que se detecte el QR |
| 60 | Sacar compartidos resolvió por sí solo la recuperación | Redujo interferencia y mejoró subconjuntos; Hebb con cinco siguió sin originales estables |

## 15. Preguntas de defensa con respuestas breves

**¿PCA y Oja son lo mismo?** Buscan la misma primera dirección bajo esta preparación. PCA la calcula por SVD y obtiene todas las componentes; Oja adapta un vector con muestras. La implementación usa PCA como evaluación, no como señal para su actualización.

**¿Kohonen aprende dos dimensiones?** Organiza prototipos de 27 dimensiones sobre una grilla de dos dimensiones. No convierte sus pesos en vectores de dos números.

**¿Qué actualiza una neurona del SOM?** Sus 27 pesos. Se acerca al perfil presentado con una intensidad dada por tasa y vecindad en grilla.

**¿Cinco regiones es un hiperparámetro de entrenamiento del SOM?** No del entrenamiento que usamos: es un parámetro del agrupamiento posterior de sus dieciséis prototipos.

**¿Por qué no elegir siempre el mapa de menor QE?** Puede memorizar individuos, dejar muchas celdas vacías y producir regiones menos legibles/estables. QE no contiene el objetivo comercial.

**¿Por qué gaussiana y no uniforme?** Calibradas con el mismo presupuesto, las dos tuvieron casi el mismo QE en creadores no vistos (3,73). Uniforme tuvo menos error topográfico; gaussiana, regiones más repetibles entre semillas. Elegimos por regiones porque las campañas se deciden sobre ellas. No es una superioridad general.

**¿Probaron vecindad uniforme con el mismo radio 2 → 0,5?** Sí, y por debajo de radio 1 deja de mover vecinas desde la época 251: baja QE de entrenamiento pero desordena el mapa (TE 24 %) y empeora en excluidos.

**¿Por qué un índice alto de Oja no es mejor?** Porque el signo es arbitrario y PC1 describe varianza, no preferencia del cliente.

**¿Hebb entrena varias épocas?** En este código se construye `ΞᵀΞ/N` una vez. Las iteraciones posteriores son de recuperación; los intentos de desaprendizaje son entrenamiento adicional de otra variante.

**¿Por qué no guardar la diagonal?** Sin autoconexiones una neurona no se sostiene artificialmente con su propio signo. Ambos sectores simétricos sí se usan porque calculan campos de neuronas distintas.

**¿Un campo cero es imposible?** No. Puede haber cancelaciones exactas. Se mantiene el signo anterior para conservar estados bipolares.

**¿Por qué hay recuerdos inversos?** Con pesos simétricos y sin sesgos, la energía satisface `E(s)=E(−s)` y la dinámica es simétrica por signo. Reforzar `ξξᵀ` también refuerza `(-ξ)(-ξ)ᵀ`.

**¿Pseudoinversa es otra arquitectura?** No: cambia la construcción de W. La recuperación sigue siendo `sign(Ws)` con el mismo manejo de empates, ciclos y tope.

**¿Garantiza el original si está almacenado?** Idealmente deja cada original como punto fijo; no garantiza llegar a ese punto desde cualquier consulta ruidosa.

**¿Se invierte una matriz enorme?** Se pseudoinvierte una Gram de cinco por cinco y luego se construye W. La matriz final recurrente sí es de N por N.

**¿Por qué la completa puede ganar si los compartidos no identifican?** Identidad y dinámica no son lo mismo: esas coordenadas cambian Gram, campos y cuencas. En Hamming, en cambio, aportan una constante igual a todos los candidatos.

**¿Desaprendizaje es pseudoinversa aproximada?** No lo tratamos como tal. Empieza en Hebb y penaliza atractores estables encontrados; no calcula la proyección por Gram.

**¿Cincuenta intentos son cincuenta cambios de pesos?** No: los protegidos/ciclos/topes no actualizan. En el candidato hubo 20/22/20 actualizaciones efectivas.

**¿Desaprendemos también durante el uso?** No. Pesos fijos en evaluación; entrenamiento y consultas se separan.

**¿Dos redes significa Hamming obligatoriamente?** No en general. Nuestro código sí usa Hamming en el selector externo. Es distinto del comparador completo sin red.

**¿Leer un QR demuestra que se reconstruyó exactamente?** No: el lector puede corregir errores residuales. Y una recuperación exacta medida sin lector no valida detección de fotografías reales.

**¿Qué conclusión es más fuerte?** La comparación emparejada apoya que la pseudoinversa completa supera al desaprendizaje reducido en esas consultas. El punto de ventaja del desaprendizaje sobre pseudoinversa reducida no es concluyente. Hamming fue la referencia de mayor media entre los comparados ahí.

**¿Las semillas nuevas son datos nuevos?** Son nuevos órdenes/inicializaciones o nuevos daños, según el caso. No son nuevas personas ni nuevos QR independientes.

## 16. Cómo ensayar la exposición

Para cada figura, practicá esta secuencia: **qué se varió → qué mide cada eje/color → qué comparación observamos → por qué creemos que pasa → qué límite tiene esa afirmación**. No empieces por leer porcentajes sin nombrar el conjunto de consultas.

Un orden de estudio sugerido, ajustable al tiempo disponible:

1. **Datos y PCA:** recalcular a mano el z-score y el ejemplo de la diagonal. Explicar el signo y la varianza que queda fuera del biplot.
2. **SOM y Ward:** hacer una actualización, separar las dos distancias, leer matriz U y explicar por qué el objetivo del cliente aparece después.
3. **Oja y campañas:** hacer el ejemplo de actualización y defender por qué el índice no es un ranking absoluto.
4. **Hebb y dinámica:** hacer A/B/C, campo local, crosstalk, energía y detección de estabilidad/ciclo.
5. **Pseudoinversa y desaprendizaje:** seguir sus flujos y explicar por qué estabilidad y cuenca no son lo mismo.
6. **Variantes y evidencia:** distinguir máscaras, zonas, grupos, Hamming y lector; repasar las condiciones de 1.100 frente a 700 consultas.
7. **Ensayo sin mirar las notas:** contar los dos ejercicios y responder las preguntas de arriba. Volver al código solo en los puntos que no puedas justificar.

### Lista de comprobación antes de presentar

- Puedo indicar la forma de cada matriz: 50×27 datos, 16×27 SOM, 27 pesos Oja, 5×1.369 recuerdos y N×N Hopfield.
- Puedo ejecutar una actualización de SOM, Oja, recuperación Hopfield y anti-Hebb.
- Puedo explicar sin fórmulas por qué el SOM necesita vecindad y por qué Hebb falla con estos QR.
- Puedo mostrar dónde se toman decisiones humanas, se restaura información conocida y se usa un selector externo.
- No confundo QE/TE, ARI de neuronas/regiones, estabilidad/exactitud, lectura/reconstrucción ni intentos/updates.
- Sé qué porcentajes pertenecen a consultas emparejadas y cuáles vienen de otros barridos.
- Sé decir «esto no se midió» ante ventas, otros QR, daños fotográficos o variantes propuestas sin probar.

## Fuentes para profundizar sin leer todo el repositorio

- Presentación: [index.html](index.html), notas con la tecla S. El mapa de estudio usa su numeración real de 97 diapositivas.
- Núcleo matemático: [preprocessing.py](../src/tp4/preprocessing.py), [pca.py](../src/tp4/pca.py), [kohonen.py](../src/tp4/kohonen.py), [oja.py](../src/tp4/oja.py), [hopfield.py](../src/tp4/hopfield.py), [unlearning.py](../src/tp4/unlearning.py).
- Interpretación y selección: [som_regions.py](../analysis/som_regions.py), [som_campanas.py](../analysis/som_campanas.py).
- Composición de soluciones QR: [shared_modules.py](../src/tp4/shared_modules.py), [zoned_hopfield.py](../src/tp4/zoned_hopfield.py), [split_hopfield.py](../src/tp4/split_hopfield.py), [qr_bits.py](../analysis/qr_bits.py).
- Hallazgos: [PCA contra entradas originales](../reports/ejercicio1/som_pca/README.md), [campañas](../reports/ejercicio1/campanas/README.md), [pruebas completas de desaprendizaje](../reports/ejercicio2/desaprendizaje/PRUEBAS_COMPLETAS.md), [comparación emparejada](../reports/ejercicio2/desaprendizaje/PSEUDOINVERSA_VS_DESAPRENDIZAJE.md).

Se hicieron comprobaciones pequeñas del código para esta guía: conteo real de diapositivas; campos y signos de los cinco QR completos/reducidos; energía de menú y mezcla con la diagonal eliminada; reconstrucción exacta de los cinco QR desde las URL; diferencia 6/90 bits; y ejemplo A/B/C. No se repitieron los barridos completos ni se modificaron los modelos o diapositivas.

Revisión adicional del 25/10: [contraste punto por punto del ejercicio 1 con el PDF](verificacion_enunciado_ejercicio1.md).
