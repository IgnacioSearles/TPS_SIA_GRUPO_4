# Bitácora del ejercicio 2 — Reconocimiento de dígitos

Última actualización: 29 de septiembre de 2026.
Estado: análisis de datos, tasas, arquitecturas, optimizadores y repetición con cinco semillas completos.
Modelo final con momentum 0,9 entrenado y evaluado en test. Ejercicio 2 documentado; ejercicio 3 no iniciado.
Esta bitácora corresponde al ejercicio 2; no contiene resultados del ejercicio 3.

## Objetivo y protocolo común

Clasificar imágenes manuscritas de 28×28 píxeles en diez clases mediante un perceptrón
multicapa. Se usa exclusivamente `digits.csv` para aprender y seleccionar. Durante
estas fases no se leyó `digits_test.csv` ni `more_digits.csv`.

En la notación teórica, ξ^μ representa una imagen, ζ^μ el objetivo one-hot y O^μ
las diez salidas. El error reportado es:

\[
\frac{1}{2p}\sum_{\mu=1}^{p}\sum_{j=1}^{10}(O_j^\mu-\zeta_j^\mu)^2.
\]

El dígito predicho es el índice de la salida máxima; en el código, los índices
0…9 coinciden con los dígitos. Las diez sigmoides no necesariamente suman uno.

Las fases usan aproximadamente 80 % de entrenamiento (9.960 imágenes) y 20 % de
validación (2.489), con separación estratificada por etiqueta y semilla 42.
Cada red comienza desde cero; no continúa con pesos de la corrida anterior.

Criterio fijado antes de comparar: mayor accuracy de validación; en empate, menor
error de validación. Se conservan los pesos de esa época. F1 macro y recall por
clase son medidas complementarias. No se modificó el criterio luego de ver resultados.

## Fase 0 — Inspección de datos y EDA

**Pregunta:** ¿qué datos recibe la red y qué limitaciones tiene el conjunto?

Se verificaron dimensiones, etiquetas, valores faltantes, rango de intensidades,
imágenes nulas y duplicados exactos. Se ejecutó el EDA existente para generar
muestras, promedios por clase, distribución de intensidades y medidas de forma.
Las medidas del EDA no se incorporaron como nuevas entradas de la red.

| Dígito | Total | Entrenamiento | Validación |
|---|---:|---:|---:|
| 0 | 1480 | 1184 | 296 |
| 1 | 1685 | 1348 | 337 |
| 2 | 1489 | 1192 | 297 |
| 3 | 1532 | 1225 | 307 |
| 4 | 1460 | 1168 | 292 |
| 5 | 271 | 217 | 54 |
| 6 | 1479 | 1183 | 296 |
| 7 | 1566 | 1253 | 313 |
| 8 | 0 | 0 | 0 |
| 9 | 1487 | 1190 | 297 |

**Resultados:** 12.449 imágenes; todas con 784 intensidades finitas entre 0 y 1;
sin valores faltantes, imágenes completamente nulas ni duplicados exactos en la
comprobación realizada. No hay ejemplos del 8 y solo hay 271 del 5.

**Decisión:** conservar las diez salidas y documentar la falta del 8; no completar
el entrenamiento con imágenes de test. No volver a dividir los píxeles por 255.

**Límite:** validación tampoco contiene ochos. Por lo tanto, ningún resultado de
estas fases mide la capacidad de reconocer esa clase. El desbalance del 5 motiva
mirar métricas por clase, pero no demuestra por sí solo la causa de cada error.

## Fase 1 — Red de referencia

**Pregunta:** ¿la implementación puede aprender este problema con una configuración
sencilla y reproducible?

Se incorporaron `experiments/digits/baseline.py`, su JSON, pruebas específicas e
inicialización Xavier en `nn/initializers.py`. Se reutilizaron las capas, backpropagation,
SGD, entrenamiento, splitter y métricas existentes. Se agregó SciPy a requirements.txt
porque el EDA existente de fraude la necesita para correlación de Spearman.

| Elección | Justificación y alcance |
|---|---|
| 784 entradas | Una intensidad por píxel; imagen aplanada en orden fijo. |
| 10 objetivos one-hot | Categorías discretas sin imponer distancia numérica entre dígitos. |
| Una capa oculta de 32 neuronas | Referencia pequeña y fácil de analizar; 32 no es un óptimo demostrado. |
| Tanh oculta | No lineal, diferenciable y coherente con la implementación estudiada. |
| Sigmoide de salida | Compatible con objetivos 0/1; no se interpreta como probabilidades que suman uno. |
| Error cuadrático | Mantiene continuidad con teoría y código; no equivale a accuracy. |
| Xavier | Escala de pesos uniforme ±sqrt(6/(entradas+salidas)); busca evitar activaciones iniciales excesivas. |
| Términos independientes cero | Los pesos aleatorios ya rompen la simetría entre neuronas. En teoría b = −θ. |
| SGD | Optimizador ya implementado y referencia para comparar después con momentum. |
| η = 0,1 | Valor exploratorio; elegido antes de la búsqueda, no como óptimo conocido. |
| Lotes de 128 | Compromiso práctico entre frecuencia de actualización y procesamiento conjunto; 78 actualizaciones por época. |
| 50 épocas | Presupuesto inicial fijo, sin parada temprana; no equivale a convergencia. |
| Semilla 42 | Reproducibilidad, sin propiedad especial del número. |
| Sin balanceo adicional | Medir primero el comportamiento sobre la distribución original. |

**Resultados medidos:**

| Medida | Resultado |
|---|---:|
| Accuracy inicial de validación | 8,60 % |
| Accuracy final seleccionada de entrenamiento | 93,57 % |
| Accuracy final seleccionada de validación | 92,85 % |
| Error inicial de validación | 1,368009 |
| Error final seleccionado de validación | 0,065130 |
| F1 macro de validación | 0,8877 |
| Recall del 5 | 20/54 = 37,04 % |
| Mejor época | 50 |

Se observó descenso del error en ambos conjuntos y mejora de la accuracy. Las
curvas todavía mejoran al terminar. No se estableció que 50 épocas sean suficientes.
La confusión más frecuente del 5 fue con el 3: 12 de los 54 ejemplos.

Se guardaron los pesos del mejor punto y se verificó igualdad exacta de predicciones
al volver a cargarlos. Pasaron 67 pruebas del proyecto en esta fase.

## Fase 2 — Comparación controlada de tasas de aprendizaje

**Pregunta:** ¿cómo cambia el aprendizaje al variar solamente η con un presupuesto
idéntico de 50 épocas?

Se agregó `experiments/digits/learning_rates.py` y su configuración. Valores elegidos
antes de ejecutar: η ∈ {0,001; 0,01; 0,1}. El espaciado por factores de diez permite
comparar tres escalas de actualización. No constituye una búsqueda exhaustiva.

Se mantuvieron arquitectura, activaciones, error, inicialización, semillas, partición,
lotes y épocas. El generador aleatorio reinicia igual en cada corrida, por lo que
los pesos iniciales y el orden de lotes son los mismos. El script verifica que el
hash del dataset, los índices de partición y las métricas iniciales coincidan.
La corrida η=0,1 reprodujo exactamente el resumen de resultados de la fase 1.

Cada fila corresponde a la mejor época de esa corrida según el criterio previamente
fijado. En las tres fue la época 50.

| η | Mejor época | Accuracy entrenamiento | Accuracy validación | Error validación | F1 macro | Recall del 5 |
|---|---:|---:|---:|---:|---:|---:|
| 0.001 | 50 | 56.16 % | 56.53 % | 0.389190 | 0.4806 | 0.00 % |
| 0.01 | 50 | 87.72 % | 87.42 % | 0.159579 | 0.7845 | 0.00 % |
| 0.1 | 50 | 93.57 % | 92.85 % | 0.065130 | 0.8877 | 37.04 % |

**Interpretación:** η=0,1 obtuvo el mejor resultado de las tres tasas bajo estas
condiciones. Las tasas menores redujeron el error, pero avanzaron menos durante
las mismas 50 épocas. Ninguna de las dos reconoció correctamente un cinco al
momento seleccionado. La tasa 0,1 mejoró también F1 macro y recall del 5, aunque
ese recall continúa bajo.

**Decisión provisional:** mantener η=0,1 como referencia para la siguiente comparación.
No se afirma que sea la mejor tasa global: es el extremo superior del rango probado,
solo se usó una semilla y las tres corridas siguen mejorando al terminar. Un presupuesto
mayor podría modificar la comparación. Tampoco se afirma que una tasa mayor sea mejor.
No se usan tiempos de estas corridas para comparar rendimiento computacional: algunas
se ejecutaron mientras corrían pruebas, y el tiempo por época del entrenador excluye
la evaluación adicional del callback.

**Verificación:** 74 pruebas pasaron, incluidos controles de que la configuración
solo cambia la tasa y el directorio de salida, y rechazo de tasas inválidas. Los tres
modelos verificaron guardado/carga de pesos. No se evaluó test.

## Fase 2b — Extensión a 500 épocas antes de comparar arquitecturas

**Pregunta:** ¿las tasas menores estaban en desventaja por el presupuesto de 50 épocas?
¿Con más entrenamiento alguna supera a η=0,1?

Se repitieron las tres tasas desde la misma inicialización, con la misma partición,
orden de lotes, arquitectura, función de error y criterio de selección. Solo se
extendió el presupuesto a 500 épocas, sin parada temprana. Se conservaron los resultados
anteriores en otro directorio. No se continuó desde los mejores pesos guardados:
se volvió a ejecutar desde cero para mantener exactamente la trayectoria original.

**Control de comparabilidad:** se comprobó igualdad exacta entre las métricas de las
primeras 50 épocas de cada nueva corrida y su corrida anterior (se excluyó únicamente
el tiempo por época). También se verificaron los índices de partición. La auditoría
quedó guardada en `extension_audit.json`. Cada modelo volvió a verificar el guardado
y carga de sus pesos. Pasaron las 7 pruebas del módulo de comparación; la suite
completa había pasado sus 74 pruebas en la fase anterior.

**Resultados:** cada columna de accuracy representa el mejor punto seleccionado
dentro de su presupuesto, no necesariamente la última época.

| η | Accuracy validación hasta 50 | Accuracy validación hasta 500 | Época elegida (500) | Error validación | F1 macro | Recall del 5 |
|---|---:|---:|---:|---:|---:|---:|
| 0.001 | 56.53 % | 87.42 % | 499 | 0.159722 | 0.7845 | 0.00 % |
| 0.01 | 87.42 % | 92.85 % | 490 | 0.065462 | 0.8877 | 37.04 % |
| 0.1 | 92.85 % | 95.10 % | 482 | 0.041139 | 0.9388 | 77.78 % |

**Interpretación:**

- Dar más tiempo ayudó a las tres tasas. Con η=0,01 se alcanzó el 92,85 % que η=0,1
  ya había alcanzado en 50 épocas. Esto muestra que la comparación inicial estaba
  midiendo, en parte, diferencias en velocidad de aprendizaje.
- Sin embargo, al dar también más tiempo a η=0,1, esta llegó a 95,10 % y siguió siendo
  la mejor del rango probado. La mejora frente a su presupuesto anterior fue de
  aproximadamente 2,25 puntos porcentuales.
- El recall del 5 pasó de 20/54 (37,04 %) a 42/54 (77,78 %) con η=0,1. Por lo tanto,
  el bajo resultado inicial no debe atribuirse únicamente al desbalance: el tiempo
  de entrenamiento también influyó. El 8 continúa sin poder evaluarse.
- El mejor punto de η=0,1 fue la época 482: accuracy de entrenamiento 98,17 % y
  validación 95,10 %. La separación entre entrenamiento y validación aumentó frente
  a la referencia. Esto exige seguir observando generalización; por sí solo no
  demuestra un deterioro sostenido de validación.
- En la época 500, η=0,1 terminó con accuracy de validación 95,02 %. Guardamos los
  pesos de la época 482, conforme al criterio establecido. Su menor error de validación
  apareció en la época 498; no se eligió esa época porque el criterio principal es
  accuracy, no error. La pequeña fluctuación final no prueba por sí sola sobreajuste.

**Límites y decisión:** con esta semilla y hasta 500 épocas, ninguna tasa menor
superó a 0,1. No se ha demostrado un óptimo global ni que las tasas menores hayan
convergido: para 0,001 y 0,01 el error de validación todavía alcanzó su mínimo en la
época 500. Por ahora se mantiene 0,1 como referencia. En el momento de cerrar la fase 2b aún no se había iniciado la fase 3; se documenta a continuación.
Para comparar arquitecturas habrá que fijar un presupuesto común y explicitar que
la tasa óptima podría cambiar con la arquitectura.

**Reproducción desde TP3:**

```bash
python -m experiments.digits.learning_rates experiments/digits/configs/learning_rates_500.json
```

La configuración fuente escribe en `results/digits_learning_rates_500`. La ejecución
original se guardó en `outputs/ejercicio2_fase2_500` y luego se copió al proyecto.
Se conservan resumen, tabla, curvas, predicciones, pesos e índices de cada corrida,
así como la configuración, el entorno y la auditoría contra las 50 épocas.
Se actualizó el reporte del estudio para registrar `epochs_budget` con el valor real.
El test continuó reservado y no se usó `more_digits.csv`.

## Fase 3 — Comparación de arquitecturas

**Pregunta:** ¿ampliar la capa oculta o agregar otra capa mejora el reconocimiento
manteniendo comunes la tasa y el presupuesto de entrenamiento?

Se compararon `[784,32,10]`, `[784,64,10]` y `[784,64,32,10]`, con η=0,1,
500 épocas, lotes de 128, SGD, error cuadrático, tanh en todas las capas ocultas,
sigmoide en la salida e inicialización Xavier. Se mantuvieron la partición estratificada
y el criterio de selección: mayor accuracy de validación y, en empate, menor error.
Cada corrida arranca desde cero y se guardan los pesos de su época elegida.

### Ajuste del protocolo aleatorio

En las fases anteriores, un generador se usaba primero para inicializar pesos y luego
para ordenar los lotes. Cambiar el tamaño de la red consume distinta cantidad de números
aleatorios durante la inicialización y, por tanto, cambiaría también el orden de lotes.

Para aislar mejor la comparación se agregó `training.shuffle_seed=42`, un generador
independiente para ordenar los ejemplos. Las tres redes ven el mismo orden de lotes.
La semilla de inicialización sigue siendo 42, pero **los pesos iniciales no son iguales**:
sus matrices tienen diferentes dimensiones y Xavier depende del tamaño de cada capa.

Se volvió a entrenar también la red de referencia. Su accuracy fue 95,06 %, frente
al 95,10 % de la fase 2b; no son réplicas exactas porque cambió el orden de lotes.
La comparación válida dentro de esta fase usa las tres corridas con el protocolo nuevo.
Si `shuffle_seed` se omite, el entrenador del experimento conserva el comportamiento
anterior, por lo que los JSON de las fases previas mantienen su reproducibilidad.

### Resultados medidos

| Arquitectura | Parámetros | Época elegida | Accuracy entrenamiento | Accuracy validación | Error validación | F1 macro | Recall del 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 784-32-10 | 25,450 | 498 | 98.20 % | 95.06 % | 0.041014 | 0.9377 | 77.78 % |
| 784-64-10 | 50,890 | 500 | 98.65 % | 95.74 % | 0.035514 | 0.9467 | 85.19 % |
| 784-64-32-10 | 52,650 | 469 | 99.47 % | 96.46 % | 0.029934 | 0.9513 | 81.48 % |

Los parámetros cuentan todos los pesos y términos independientes. Para tamaños de
capa consecutivos a y b, una capa densa aporta a×b+b parámetros. Las tres redes
hicieron las mismas 500 épocas, pero esto no significa igual costo computacional:
las redes mayores procesan más conexiones en cada actualización.

**Interpretación:**

- Ampliar de 32 a 64 neuronas subió la accuracy de 95,06 % a 95,74 % (17 imágenes
  adicionales correctas sobre 2.489) y el recall del 5 de 42/54 a 46/54.
- Agregar una segunda capa oculta de 32 neuronas subió la accuracy a 96,46 %
  (18 aciertos adicionales respecto de la red de 64). También tuvo el mayor F1 macro.
- La red profunda no ganó en todas las clases: reconoció 44/54 cincos (81,48 %),
  frente a los 46/54 (85,19 %) de la red de 64. Esa diferencia equivale a dos imágenes;
  todavía no es evidencia suficiente de una ventaja consistente sobre esa clase.
- La red profunda alcanzó 99,47 % de accuracy de entrenamiento y 96,46 % de validación
  en la época seleccionada. Aprender casi perfectamente entrenamiento no equivale a
  resolver todas las imágenes nuevas.
- La mejor época seleccionada fue 469 para la red profunda; la última terminó en
  96,42 %. El menor error de validación apareció en la época 491. Como antes, error
  y accuracy no seleccionan necesariamente el mismo punto.

**Decisión provisional:** conservar `[784,64,32,10]` como referencia para la próxima
comparación de optimizadores, siguiendo el criterio de accuracy establecido antes de
ver los resultados. Conservar también `[784,64,10]` entre las candidatas: tuvo buen
resultado global y el mayor recall del 5. No se afirma un ganador definitivo.

**Límites:** solo se probó una semilla, una partición, una tasa y un presupuesto.
Las arquitecturas pueden necesitar tasas o duraciones distintas. Agregar profundidad
también cambia la cantidad de parámetros y la inicialización; no se puede atribuir
toda la mejora exclusivamente a la profundidad. La validación se está usando para
seleccionar, por lo que los resultados no reemplazan test. La clase 8 sigue ausente.

### Implementación y respaldo

Se agregaron `experiments/digits/architectures.py`,
`experiments/digits/configs/architectures.json` y pruebas para la configuración y el
orden común de lotes. Se extendió `baseline.py` con la semilla opcional de lotes.
La suite completa pasó **81 pruebas**. Se verificó igualdad de particiones entre
corridas, ausencia de intersección entre entrenamiento y validación, el conteo de
parámetros contra los pesos guardados y la reproducción de predicciones al recargarlos.

Desde TP3:

```bash
python -m experiments.digits.architectures experiments/digits/configs/architectures.json
```

La configuración fuente escribe en `results/digits_architectures`. La corrida original
se guardó en `outputs/ejercicio2_fase3`, luego copiada a ese directorio del proyecto.
Incluye `comparison.csv`, `study_summary.json`, `study_config.json`, `audit.json`,
`environment.json`, `architecture_comparison.png` y los archivos individuales de cada
red en `run_00`, `run_01` y `run_02` (configuración, pesos, particiones, predicciones,
historia y gráficos). Los resultados anteriores se conservaron.

## Fase 4 — SGD frente a momentum

**Pregunta:** ¿incorporar memoria de actualizaciones anteriores permite aprender más
rápido o alcanzar mejores resultados que SGD sobre la arquitectura seleccionada?

Se implementó momentum clásico en `nn/optimizers.py`, registrado como `momentum`.
Para cada peso y término independiente, se conserva una actualización anterior:

\[
\Delta w(t) = -\eta\,\frac{\partial(\text{error promedio del lote})}{\partial w}
             + \alpha\,\Delta w(t-1),
\qquad w(t+1)=w(t)+\Delta w(t).
\]

Aquí α identifica el factor de memoria de este experimento (`momentum` en el JSON).
La actualización previa comienza en cero. Con α=0 se recupera SGD. La memoria se
mantiene entre lotes y épocas, y se reinicia al construir cada optimizador nuevo.
No se implementó Nesterov ni se multiplica el gradiente por (1−α).

Mantener una dirección de gradiente constante acumularía actualizaciones cuya escala
se aproxima a η/(1−α) por ese gradiente. Esta observación no describe exactamente una
red real, donde el gradiente cambia, pero explica por qué también variamos η:
una misma tasa nominal no implica el mismo tamaño de actualización con momentum.

### Protocolo fijado antes de las corridas

- Arquitectura `[784,64,32,10]`, tanh oculta, sigmoide de salida, MSE y Xavier.
- 500 épocas, lotes de 128, semillas de inicialización y lotes 42.
- Misma partición, mismo orden de lotes y mismo modelo inicial en todas las corridas.
- SGD con η ∈ {0,01; 0,1}.
- Momentum con α ∈ {0,5; 0,9}, cruzado con las mismas dos tasas: cuatro corridas.
- Selección por mayor accuracy de validación, luego menor error. No se cambió ese criterio.
- F1 macro y recall del 5 como medidas complementarias. Durante la ejecución se agregó
  una medida descriptiva: primera época que alcanza 95 % de accuracy de validación.
  Ese umbral no se utilizó para detener ni seleccionar modelos y puede alcanzarse de
  manera transitoria; no es una medición de tiempo de ejecución.

El presupuesto es igual por configuración, no por familia: momentum tiene cuatro
configuraciones y SGD dos porque incorpora un hiperparámetro adicional. Este barrido
no demuestra superioridad general de un algoritmo ni reemplaza ajustes más extensos.

### Resultados medidos

Cada fila informa métricas de la época seleccionada; la última columna se obtiene de
la historia completa de su corrida. “No alcanzó” significa dentro de estas 500 épocas.

| Optimizador | α | η | Época elegida | Accuracy validación | F1 macro | Recall del 5 | Primera época ≥95 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| sgd | 0 | 0.01 | 500 | 93.21 % | 0.8372 | 0.00 % | No alcanzó |
| sgd | 0 | 0.1 | 469 | 96.46 % | 0.9513 | 81.48 % | 129 |
| momentum | 0.5 | 0.01 | 390 | 93.89 % | 0.8433 | 0.00 % | No alcanzó |
| momentum | 0.5 | 0.1 | 371 | 96.42 % | 0.9524 | 81.48 % | 71 |
| momentum | 0.9 | 0.01 | 497 | 96.10 % | 0.9502 | 81.48 % | 177 |
| momentum | 0.9 | 0.1 | 76 | 96.34 % | 0.9520 | 79.63 % | 21 |

**Decisión provisional:** la configuración seleccionada fue `sgd`, con
η=0.1 y α=0, en la época 469. Obtuvo
96.46 % de accuracy de validación, F1 macro
0.9513 y recall del 5 de 81.48 %.
La configuración seleccionada fue el propio SGD; el mejor momentum quedó a un
solo acierto sobre las 2.489 imágenes de validación.

**Qué aportó momentum en estas corridas:**

- Con η=0,1, SGD alcanzó por primera vez 95 % en la época 129; momentum con α=0,5
  lo hizo en la 71 y con α=0,9 en la 21. Esto demuestra menos épocas para ese umbral
  en estas corridas, no una aceleración equivalente del tiempo de ejecución.
- SGD obtuvo la mayor accuracy, 96,46 %. Momentum con α=0,5 y η=0,1 alcanzó 96,42 %:
  la diferencia es **una imagen de 2.489**, insuficiente para afirmar una ventaja
  consistente con una sola semilla. Momentum tuvo mejor F1 macro (0,9524 frente a
  0,9513) y menor error en los puntos seleccionados. Ambos reconocieron 44/54 cincos.
- Momentum con α=0,9 y η=0,1 alcanzó su mejor accuracy en la época 76 (96,34 %).
  A la época 500 bajó a 95,90 %, mientras mejoró entrenamiento. El error de validación
  pasó de 0,031934 en el punto seleccionado a 0,033012 al final. Es un patrón compatible
  con sobreajuste durante la prolongación, que muestra por qué conservamos la mejor época.
- Con η=0,01, α=0,9 mejoró sustancialmente respecto de SGD y α=0,5; no es correcto
  concluir que una tasa pequeña sea mala sin considerar el optimizador y su memoria.

Se mantienen SGD (η=0,1) y momentum (η=0,1; α=0,5 y 0,9) como candidatos para repetir
con varias semillas. La selección por accuracy conserva SGD provisionalmente, sin
descartar el interés de momentum por la velocidad de aprendizaje y las otras métricas.

**Alcance:** se usó una semilla y una partición; diferencias pequeñas pueden variar
con otras inicializaciones. La validación se reutilizó para seleccionar arquitectura
e hiperparámetros, así que sus métricas son de desarrollo y pueden ser optimistas.
No se usó test. No hay ejemplos del 8; F1 macro sigue incluyendo solo clases con soporte.
Los tiempos de las corridas no se usaron para afirmar mejoras de rendimiento computacional.

### Implementación y verificación

Se agregó `experiments/digits/optimizer_study.py` y
`experiments/digits/configs/optimizers.json`. Se probaron dos actualizaciones manuales
con gradientes de distinto signo, independencia de memoria entre parámetros,
equivalencia con SGD cuando α=0, rechazo de hiperparámetros inválidos y generación
del barrido manteniendo comunes las demás condiciones. Pasaron **90 pruebas**.

El experimento verifica el mismo hash de datos, índices de partición y métricas
iniciales. La auditoría verificó que SGD con η=0,1 reproduce exactamente el resumen y
las 500 épocas de la fase 3 (excluyendo tiempos), y que las métricas de cada modelo
corresponden a la época seleccionada. Se verificó guardado/carga de todos los modelos.
Los archivos de pesos permiten inferencia; no guardan el estado de momentum para
reanudar entrenamiento. Cada corrida de este estudio empieza desde cero.

Desde TP3:

```bash
python -m experiments.digits.optimizer_study experiments/digits/configs/optimizers.json
```

Resultados en `results/digits_optimizers`, copiados de la ejecución original en
`outputs/ejercicio2_fase4`. Incluyen tabla, resumen, configuración, auditoría,
curvas y seis subdirectorios con pesos, predicciones, particiones e historias.
Las fases anteriores se conservaron. La siguiente fase es repetir finalistas con
varias semillas y decidir el procedimiento final antes de evaluar test.

## Fase 5 — Repetición de finalistas con cinco semillas

**Pregunta:** ¿la diferencia entre los optimizadores se sostiene al cambiar la
inicialización de los pesos y el orden de presentación de los lotes?

Antes de ejecutar se fijaron las semillas 42, 7, 21, 84 y 123. Para cada una se
compararon SGD y momentum con factores 0,5 y 0,9, todos con η=0,1, arquitectura
`[784,64,32,10]`, 500 épocas y lotes de 128. Se mantuvieron Xavier, tanh, sigmoide,
MSE y la partición estratificada original (`data.split_seed=42`).

La comparación es pareada: dentro de una semilla los tres métodos tienen el mismo
modelo inicial y el mismo orden de lotes. Entre semillas cambian ambas cosas. Por
lo tanto, la variación observada no separa el efecto individual de inicialización
y orden, y tampoco mide la incertidumbre debida a cambiar la partición de datos.

Se reutilizaron las tres corridas anteriores de semilla 42 solamente después de
comprobar configuración completa (excepto directorio de salida), hash del dataset
y presencia de los archivos de respaldo. Se entrenaron doce corridas nuevas. En
`runs.csv`, `reused_from` identifica exactamente la procedencia de las reutilizadas.

Cada corrida sigue guardando la época de mayor accuracy de validación (desempate:
menor error). El criterio para seleccionar la **configuración** cambia de una corrida
a la media de accuracy sobre las cinco semillas, con media de error como desempate.
Este criterio se fijó antes de ejecutar. No se elige una configuración por su semilla
más afortunada ni se interpreta su máximo como el rendimiento esperado.

### Resumen de resultados

Media ± desviación estándar **muestral** (divisor n−1). En accuracy y recall, los
valores están expresados en porcentaje y la desviación en puntos porcentuales.
La desviación no es un intervalo de confianza ni el error estándar de la media.
F1 macro sigue promediando únicamente las nueve clases con ejemplos.

| Configuración | Accuracy media ± DE (%) | Mín.–máx. (%) | F1 macro media ± DE | Recall del 5 media ± DE (%) | Mediana época elegida | Mediana primera época ≥95 % |
|---|---:|---:|---:|---:|---:|---:|
| SGD | 96.239 ± 0.181 | 95.98–96.46 | 0.9513 ± 0.0024 | 81.85 ± 2.41 | 457 | 117 |
| Momentum 0.5 | 96.352 ± 0.151 | 96.10–96.50 | 0.9524 ± 0.0019 | 82.22 ± 1.01 | 380 | 60 |
| Momentum 0.9 | 96.264 ± 0.085 | 96.14–96.34 | 0.9510 ± 0.0007 | 81.85 ± 2.41 | 76 | 16 |

La última columna mide el primer cruce del umbral por corrida, no exige que se
mantenga por encima después y no mide tiempo de ejecución.

### Resultados individuales

| Semilla | Configuración | Época elegida | Accuracy validación | Recall del 5 |
|---|---|---:|---:|---:|
| 42 | SGD | 469 | 96.464 % | 81.48 % |
| 42 | Momentum 0.5 | 371 | 96.424 % | 81.48 % |
| 42 | Momentum 0.9 | 76 | 96.344 % | 79.63 % |
| 7 | SGD | 494 | 96.183 % | 79.63 % |
| 7 | Momentum 0.5 | 380 | 96.384 % | 81.48 % |
| 7 | Momentum 0.9 | 41 | 96.264 % | 85.19 % |
| 21 | SGD | 358 | 95.982 % | 79.63 % |
| 21 | Momentum 0.5 | 412 | 96.103 % | 81.48 % |
| 21 | Momentum 0.9 | 33 | 96.143 % | 81.48 % |
| 84 | SGD | 457 | 96.344 % | 83.33 % |
| 84 | Momentum 0.5 | 378 | 96.505 % | 83.33 % |
| 84 | Momentum 0.9 | 291 | 96.344 % | 83.33 % |
| 123 | SGD | 414 | 96.223 % | 85.19 % |
| 123 | Momentum 0.5 | 449 | 96.344 % | 83.33 % |
| 123 | Momentum 0.9 | 316 | 96.223 % | 79.63 % |

### Comparación pareada con SGD

- `momentum_lr0.1_m0.5` frente a SGD: diferencia promedio +0.112 puntos porcentuales; gana en 4 semillas, empata en 0 y pierde en 1.
- `momentum_lr0.1_m0.9` frente a SGD: diferencia promedio +0.024 puntos porcentuales; gana en 2 semillas, empata en 2 y pierde en 1.

**Selección provisional por el criterio fijado:** Momentum 0.5, con η=0.1,
obtuvo la mayor accuracy media: 96.352 %
(desviación estándar 0.151 puntos porcentuales).
La mediana de sus épocas seleccionadas fue 380; este número
es un dato para discutir el entrenamiento final, no una duración óptima garantizada.

**Interpretación de esta fase:** momentum 0,5 mejora la accuracy media en 0,112
puntos porcentuales frente a SGD y gana en cuatro de cinco semillas. Su F1 macro
medio también es ligeramente mayor y su recall medio del 5 es 82,22 %, frente a
81,85 % de los otros dos métodos. Las diferencias de calidad son pequeñas.
Momentum 0,9 presenta la menor dispersión de accuracy (0,085 puntos porcentuales)
y la menor mediana de épocas para llegar al 95 %: 16, frente a 60 con memoria 0,5
y 117 con SGD. Así, el criterio de mayor accuracy media selecciona memoria 0,5,
mientras que memoria 0,9 sigue siendo interesante si se prioriza aprender en menos épocas.
No se midió aquí el costo de cómputo necesario para alcanzar ese umbral.

**Límites de la conclusión:** cinco semillas permiten observar variabilidad, pero
no prueban superioridad general ni significación estadística. La semilla 42 ya había
participado en la selección de candidatos, y las cinco usan la misma validación sobre
la que se eligieron arquitecturas, tasas y épocas. Estos resultados siguen siendo de
desarrollo y pueden ser optimistas. El recall del 5 se mide sobre los mismos 54 ejemplos,
no sobre cinco muestras independientes de esa clase. El 8 continúa ausente. No se
ha leído el conjunto de test ni se ha entrenado todavía un modelo final con todos los datos.

### Implementación y verificación

Se agregaron `experiments/digits/seed_study.py`, `experiments/digits/configs/seeds.json`
y pruebas de la cuadrícula de semillas, agregación y rechazo de reutilización con
configuración o datos diferentes. Pasaron **97 pruebas** del proyecto.

Se verificaron 15 corridas únicas, 500 épocas por corrida, mismos índices de partición,
ninguna intersección entre entrenamiento y validación, métricas iniciales iguales
entre métodos dentro de cada semilla y concordancia de las métricas seleccionadas
con su historia. Los modelos nuevos verificaron guardado y carga de predicciones;
los reutilizados conservan esa verificación de la fase anterior.

Desde TP3:

```bash
python -m experiments.digits.seed_study experiments/digits/configs/seeds.json
```

El JSON fuente escribe en `results/digits_seeds` y permite reutilizar corridas
coincidentes de `results/digits_optimizers`. Si no existen fuentes coincidentes,
entrena desde cero. No reemplaza subdirectorios de corrida existentes: para repetir,
elegir un `output.directory` nuevo. La ejecución original se guardó en
`outputs/ejercicio2_fase5` y luego se copió al proyecto.

Archivos principales: `runs.csv` (15 corridas), `aggregate.csv` (medias, desviaciones
y rangos), `study_summary.json` (criterio y selección), `audit.json` (diferencias
pareadas y controles), `seed_comparison.png`, configuración y entorno. Cada corrida
conserva pesos, predicciones, índices, historia y gráficos. Las fases previas se conservaron.

## Decisión para la siguiente etapa — Momentum 0,9

**Decisión del equipo, indicada por el usuario después de la fase 5:** continuar
con momentum de factor 0,9, manteniendo η=0,1 y arquitectura `[784,64,32,10]`.
La elección se tomó antes de consultar test.

### Fundamento y cambio explícito de criterio

El criterio inicial de selección era mayor accuracy media de validación. Bajo ese
criterio, momentum 0,5 ganó y los resultados históricos se conservan sin modificaciones.
Para la siguiente etapa se prioriza alcanzar un buen nivel de reconocimiento en menos
épocas y la menor variabilidad observada entre semillas, aceptando una pequeña reducción
en accuracy media. Es una decisión de compromiso tomada después de analizar validación,
no una afirmación de que 0,9 haya ganado el criterio original ni una ponderación matemática
que se hubiera fijado antes del experimento.

| Evidencia de cinco semillas | Momentum 0,5 | Momentum 0,9 |
|---|---:|---:|
| Accuracy media de validación | 96,352 % | 96,264 % |
| Desviación estándar de accuracy | 0,151 puntos porcentuales | 0,085 puntos porcentuales |
| Mediana de primera época con accuracy ≥95 % | 60 | 16 |
| F1 macro medio | 0,9524 | 0,9510 |
| Recall medio del 5 | 82,22 % | 81,85 % |

Se acepta una reducción de aproximadamente **0,088 puntos porcentuales** de accuracy
media (unos 2,2 aciertos por corrida sobre 2.489 imágenes). En cambio, momentum 0,9
alcanzó por primera vez el 95 % en una mediana de 16 épocas frente a 60: **3,75 veces
menos épocas para ese umbral**, y mostró menor dispersión de accuracy entre las cinco
semillas. Esto no demuestra una aceleración equivalente del tiempo de cómputo ni una
superioridad general. Tampoco significa que su mejor resultado se alcance en 16 épocas.

La elección se apoya en métricas de validación reutilizada y solo cinco semillas.
La ausencia del 8 y el reducido número de cincos siguen siendo limitaciones. El conjunto
de test permanece reservado y no intervino en esta decisión.

### Configuración seleccionada y protocolo propuesto

- Arquitectura: 784 → 64 → 32 → 10.
- Activaciones: tanh en ambas capas ocultas y sigmoide en la salida.
- Inicialización: Xavier; objetivos one-hot; píxeles ya normalizados entre 0 y 1.
- Error: cuadrático promedio por ejemplo, sumando las diez componentes de salida.
- Optimizador: momentum clásico, factor 0,9 y tasa η=0,1.
- Tamaño de lote: 128.

**Protocolo propuesto para fijar antes de ejecutar la evaluación final:** entrenar
desde cero con todo `digits.csv`, usando la semilla original 42 tanto para pesos como
para el generador separado de lotes. Se conserva por continuidad, no por buscar la
semilla con mejor resultado. Como duración inicial del entrenamiento final se propone
**76 épocas**, la mediana de las épocas seleccionadas en las cinco corridas con
momentum 0,9 (33, 41, 76, 291 y 316). La mediana reduce la influencia de valores extremos.

Esta regla transfiere un número de recorridos del dataset, no un número idéntico de
actualizaciones: con 12.449 imágenes y lote 128 hay 98 lotes por época, frente a 78
con las 9.960 imágenes anteriores. Por eso 76 épocas no es una duración óptima
comprobada para el entrenamiento completo. Debe quedar fijada antes de mirar test;
no se elegirá una época mediante resultados de test.

**Estado al registrar la elección (antes de la fase 6):** se documentó la elección y se propuso el protocolo.
Todavía no se entrenó el modelo final ni se leyó `digits_test.csv`. La siguiente
etapa ejecutará el protocolo fijado, guardará los pesos y realizará la evaluación
independiente sin volver a ajustar los hiperparámetros a partir de test.

## Fase 6 — Entrenamiento final y evaluación independiente

**Protocolo aceptado y fijado antes de consultar test:** red 784→64→32→10, tanh en
capas ocultas, sigmoide en salida, Xavier, MSE, momentum clásico 0,9, η=0,1,
lotes de 128 y 76 épocas. Semilla de pesos 42 y generador separado de lotes con
semilla 42. Se entrenó desde cero con **todo digits.csv: 12.449 imágenes**.

La duración de 76 épocas se tomó de la mediana de las mejores épocas de desarrollo
para momentum 0,9; no se buscó la mejor época en test. Con el dataset completo se
realizaron 98 actualizaciones por época (7.448 en total). Se guardaron los pesos al
final de la época 76 y se verificó su recarga antes de abrir el test. No hubo parada
temprana, selección de checkpoint ni cambios de hiperparámetros basados en test.
La configuración y el protocolo se escribieron a disco antes del entrenamiento.

### Resultado principal: todas las imágenes de test

| Medida | Resultado |
|---|---:|
| Imágenes de test | 2.497 |
| Aciertos | 2.145 |
| Errores | 352 |
| **Accuracy de test** | **85,90 %** |
| F1 macro de test (las diez clases) | 0,8122 |
| Error cuadrático promedio de test | 0,102130 |
| Accuracy de entrenamiento final | 99,58 % |
| Recall del 5 en test | 186/223 = 83,41 % |
| Recall del 8 en test | 0/243 = 0 % |

La accuracy de entrenamiento se informa como diagnóstico de ajuste, no como
estimación de generalización. El resultado oficial de esta evaluación es **85,90 %**;
no se sustituye por el resultado sobre un subconjunto favorable.

### Resultados por dígito

| Dígito real | Cantidad en test | Aciertos | Recall | F1 |
|---|---:|---:|---:|---:|
| 0 | 245 | 243 | 99.18 % | 0.9419 |
| 1 | 283 | 280 | 98.94 % | 0.9573 |
| 2 | 258 | 239 | 92.64 % | 0.8707 |
| 3 | 252 | 252 | 100.00 % | 0.8456 |
| 4 | 245 | 238 | 97.14 % | 0.9426 |
| 5 | 223 | 186 | 83.41 % | 0.7915 |
| 6 | 239 | 230 | 96.23 % | 0.9237 |
| 7 | 257 | 243 | 94.55 % | 0.9492 |
| 8 | 243 | 0 | 0.00 % | 0.0000 |
| 9 | 252 | 234 | 92.86 % | 0.9000 |

### Interpretación de los errores y cobertura

A diferencia de la validación de desarrollo, **test sí contiene las diez clases**.
Hay 243 imágenes de 8 y no hubo ejemplos de 8 en entrenamiento. La red no predijo
8 para ninguna imagen del test. Los 243 ochos fueron errores, aproximadamente el
69,03 % de los 352 errores totales. Se confundieron principalmente con 3 (67 casos),
5 (56) y 2 (42). La matriz de confusión conserva esas imágenes y muestra el problema.

Esto afecta también la precisión de las clases que reciben ochos mal clasificados.
Por ejemplo, se reconocieron los 252 treses reales (recall 100 %), pero la precisión
del 3 fue 73,26 %: varias imágenes de otros dígitos fueron clasificadas como 3.

Como **análisis secundario de cobertura**, sobre las 2.254 imágenes cuyo dígito sí
estuvo presente en entrenamiento hubo 2.145 aciertos: accuracy 95,16 % y F1 macro
0,9502. Es una descripción posterior calculada con las mismas predicciones; no se
reentrenó ni se seleccionó un segundo modelo, y no reemplaza la métrica global.

Además, el 5 representa una proporción mayor en test (223/2.497 = 8,93 %) que en
entrenamiento (271/12.449 = 2,18 %), y su recall es menor al de otras clases conocidas.
Por ello, no se debe comparar directamente la accuracy global de desarrollo con la
de test suponiendo que ambos conjuntos tienen idéntica composición.

El F1 macro de desarrollo promediaba nueve clases con soporte y no incluía al 8.
En test sí hay soporte para el 8, cuyo F1 es cero: el promedio principal incluye
las diez clases. Se documenta esta diferencia para evitar una comparación engañosa.

Se comprobó que no hay imágenes exactamente idénticas de test dentro de entrenamiento
mediante hash de sus vectores de píxeles. Este control no descarta semejanzas visuales
ni duplicados aproximados. No se quitaron imágenes del test para mejorar las métricas.

### Implementación y controles

Se agregaron `experiments/digits/final_evaluation.py`,
`experiments/digits/configs/final.json` y pruebas de la separación de responsabilidades.
Las pruebas usan archivos sintéticos y verifican que el test se lea después de guardar
pesos, que se entrene con todas las filas y que se conserven las clases ausentes de
entrenamiento en las métricas principales. Pasaron **99 pruebas** del proyecto.

Se hizo una evaluación del modelo sobre el test real, después del entrenamiento.
Las tablas por clase y el análisis de cobertura derivan de esa misma matriz de salidas.
Se guardaron hashes de configuración, pesos, dataset de entrenamiento y dataset de test;
el archivo de pesos permaneció intacto después de evaluar. La recarga de pesos se
verificó comparando exactamente las salidas de las primeras 128 imágenes de entrenamiento.

Desde TP3:

```bash
python -m experiments.digits.final_evaluation experiments/digits/configs/final.json
```

La ejecución escribe por defecto en `results/digits_final`. Rechaza un directorio
con resultados existentes para no reemplazar esta evaluación. Reproducir el mismo
protocolo en otro directorio no constituye una nueva evaluación independiente ni
autoriza a ajustar el modelo a partir del test ya observado.

La corrida original se guardó en `outputs/ejercicio2_final` y fue copiada al proyecto.
Archivos: `final_weights.npz`, `config.json`, `protocol.json`, `history.csv`,
`summary.json`, `test_predictions.csv`, `training_curves.png`,
`test_confusion_matrix.png` y `environment.json`. Las columnas `output_0`…`output_9`
son salidas sigmoides, no probabilidades normalizadas que sumen uno.

**Cierre:** no se cambió la configuración después de observar test y no se usó
`more_digits.csv`. El ejercicio 2 queda con experimentos de desarrollo, decisión
justificada y evaluación final documentados. La ausencia de la clase 8 es una
limitación central del sistema entrenado, no un motivo para ocultar su error.

## Convenciones de métricas

- Accuracy: proporción de etiquetas correctamente predichas.
- Matriz de confusión: 10×10, filas reales, columnas predichas.
- Recall de una clase: aciertos de esa clase / cantidad real de esa clase.
- Sin ejemplos reales: recall y F1 se guardan como null (no evaluable).
- Sin predicciones de una clase: precision es null; si tiene soporte real, F1 es cero.
- F1 macro: promedio de F1 exclusivamente sobre clases con soporte real, enumeradas
  en `macro_labels`. En desarrollo son 0,1,2,3,4,5,6,7,9; en el test final se incluyen las diez clases.

La accuracy global puede ocultar dificultades en clases poco frecuentes. Además,
seleccionar modelos con esta validación hace que sus resultados sean de desarrollo:
no sustituyen una evaluación independiente en test.

## Reproducción y archivos de respaldo

Desde TP3, con requirements.txt instalado:

```bash
python -m experiments.digits.baseline experiments/digits/configs/baseline.json
python -m experiments.digits.learning_rates experiments/digits/configs/learning_rates.json
```

Los defaults escriben en `results/digits_baseline` y `results/digits_learning_rates`.
La regla general de Git ignora results/, pero esta entrega incorpora explícitamente
las capturas de resultados de las fases del ejercicio 2 para revisión del equipo.
La bitácora, los scripts y los JSON también se versionan; los experimentos se reproducen
con los comandos indicados, usando un directorio de salida nuevo cuando corresponda.
Cambiar `output.directory` para conservar distintas ejecuciones.

Se copiaron a esos directorios los resultados efectivamente obtenidos en las fases.
Durante la ejecución original se usaron los directorios de salida de esta conversación:
`outputs/ejercicio2_baseline` y `outputs/ejercicio2_fase2`. Los paths absolutos de los
JSON y resúmenes conservan esa procedencia; los JSON fuente del repositorio usan paths
relativos a TP3. Existe una copia de esta bitácora en `outputs/EJERCICIO2_BITACORA.md`.

En la fase 2, `comparison.csv` reúne las métricas, `study_summary.json` identifica el
modelo seleccionado, `study_config.json` registra el protocolo y
`learning_rate_comparison.png` muestra las curvas superpuestas. Cada subdirectorio
`run_00`, `run_01`, `run_02` contiene configuración, historia, índices, predicciones,
pesos, resumen y gráficos de su corrida. `environment.json` registra el entorno.

Hash SHA-256 de digits.csv: `11271dbc12a3a8db967f6c2708e55ae1918d69186e81d91c53c24e305fc5d3f4`.

Commit base inspeccionado: `309a4758a99d113e6d5105daafd19d34c3a0939b`.
Ese commit identifica el estado anterior al ejercicio 2. Esta entrega incorpora
los scripts, configuraciones, pruebas, documentación y resultados de las fases posteriores.

## Estado final y trabajo posterior

El ejercicio 2 cuenta con análisis de datos, comparación de tasas, arquitecturas y
optimizadores, repetición con cinco semillas, selección documentada y evaluación final.
Los resultados del test no se utilizarán para reajustar este modelo y volver a presentar
su evaluación como independiente. El ejercicio 3, con more_digits.csv, no fue iniciado.

## Texto breve para la presentación

“Comparamos tasas, arquitecturas y optimizadores y repetimos finalistas con cinco
semillas. Elegimos momentum 0,9 porque llegó antes al 95 % y tuvo menor variabilidad,
aceptando una pequeña reducción de accuracy media frente a momentum 0,5. Fijamos
76 épocas a partir del desarrollo y entrenamos con todas las imágenes disponibles.
En test obtuvimos 85,90 % de accuracy y F1 macro 0,8122. El dataset de entrenamiento
no tenía ochos, pero test contenía 243 y el modelo falló en todos. Sobre las clases
que sí había visto obtuvo 95,16 %, que reportamos solo como diagnóstico. El resultado
principal incluye las diez clases y no ajustamos el modelo después de mirar test.”
