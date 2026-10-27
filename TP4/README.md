# TP4: aprendizaje no supervisado

Base de Kohonen (SOM), Oja, preprocesamiento y experimentos. Las implementaciones de aprendizaje usan NumPy; pandas maneja tablas y matplotlib exporta gráficos.

## Ejecución

Desde la raíz `TP4`, con Python 3.10 o posterior:

```bash
python3 -m pip install -r requirements.txt
python3 analysis/eda_pca.py
python3 analysis/hopfield_qr.py
python3 main.py train --config configs/kohonen.json
python3 main.py train --config configs/kohonen_hexagonal.json
python3 main.py train --config configs/oja.json
python3 main.py experiment --config configs/kohonen_experiment.json --dry-run
python3 main.py experiment --config configs/kohonen_experiment.json
python3 main.py experiment --config configs/oja_experiment.json
python3 -m unittest discover -s tests -v
```

Se requiere `data/creadores.csv`. Las rutas del JSON son relativas al directorio desde el que se ejecuta el comando. `--dry-run` valida configuraciones y muestra el número de corridas sin entrenar ni crear directorios; no valida el contenido del CSV. Si hay corridas fallidas, se conservan los resultados de las restantes y la CLI termina con código 1. Una pausa con Ctrl+C termina con código 130 y permite reanudar.

## Organización y uso desde Python

- `src/tp4`: configuraciones, cálculo numérico, geometría, preprocesamiento, modelos y exportación separados por responsabilidad.
- `experiments`: expansión de parámetros, ejecución, aislamiento de errores y resúmenes.
- `configs`: ejemplos editables. `tests`: verificaciones con datos sintéticos. `analysis`: EDA y PCA inicial.

Los modelos reciben matrices NumPy; no conocen archivos ni gráficos. Cada `fit` inicia un entrenamiento nuevo usando un generador aleatorio local, salvo que se pase un checkpoint compatible. La CLI guarda automáticamente un único checkpoint por corrida al completar cada época.

```python
import pandas as pd
from src.tp4.preprocessing import Preprocessor, PreprocessingConfig
from src.tp4.config import KohonenConfig, OjaConfig, EarlyStopping
from src.tp4.kohonen import Kohonen
from src.tp4.oja import Oja

frame = pd.read_csv('data/creadores.csv')
prep = Preprocessor(PreprocessingConfig(pca_components=None))
x = prep.fit_transform(frame)
som = Kohonen(KohonenConfig(rows=4, cols=6, topology='hexagonal')).fit(x)
neurons = som.predict(x)                 # índices planos en orden fila-columna
coordinates = som.grid.indices[neurons] # coordenadas (fila, columna)
u = som.u_matrix()
oja = Oja(OjaConfig()).fit(x)
index = oja.transform(x)
```

`Preprocessor.state()` produce un estado JSON; `Preprocessor.from_state(state)` lo restaura sin recalcular estadísticas. Para datos nuevos se usa `transform`, conservando columnas y escalado del catálogo de referencia. No se imputan faltantes ni se eliminan constantes silenciosamente.

## Topologías y responsabilidades

`Topology` es una interfaz abstracta con dos implementaciones: `RectangularTopology` y `HexagonalTopology`. El contrato define posiciones, distancias entre neuronas, adyacencia y vértices de las celdas. Trabaja exclusivamente con NumPy, sin depender del entrenamiento ni de matplotlib.

`Grid` recibe esa interfaz, precalcula la geometría y calcula la matriz U. Kohonen usa las matrices resultantes; no contiene decisiones sobre rectangular o hexagonal. Los gráficos usan los vértices de la misma topología, sin seleccionar figuras por nombre.

Los JSON conservan los nombres originales. `create_topology` convierte el nombre en una implementación; desde Python también se puede inyectar una topología directamente:

```python
from src.tp4.topology import HexagonalTopology

som = Kohonen(
    KohonenConfig(rows=4, cols=6),
    topology=HexagonalTopology(),
).fit(x)
```

La instancia explícita tiene prioridad sobre el nombre de configuración. Para agregar una topología desde Python, implementar el contrato e inyectarla; para seleccionarla desde JSON, registrarla en la fábrica. No es necesario modificar Kohonen, Grid ni el dibujado.

Los entrenamientos separan inicialización, procesamiento de una época y actualización de una muestra. En experimentos, `configuration` valida y expande parámetros, `runner` coordina corridas y `reporting` agrupa y escribe resúmenes. La persistencia específica de cada modelo está separada del dibujado en `plotting`.

## PCA compartido, resultados e historiales

`src.tp4.pca.compute_pca` es la única implementación de SVD/PCA. La usan el preprocesamiento, el EDA, la evaluación de Oja y su exportación. Devuelve `PCAResult`, con componentes en filas, scores, autovalores, varianza explicada y rango. El signo se fija haciendo positivo el coeficiente de mayor módulo de cada componente. El cálculo centra por defecto, pero no estandariza: `center=False` se usa cuando la entrada ya está centrada. La estandarización sigue siendo responsabilidad del preprocesamiento.

El EDA separa validaciones y tablas (`analysis.eda`), representaciones (`analysis.representation`), gráficos (`analysis.plots`) y exportación (`analysis.reporting`). `analysis.eda_pca.run()` coordina estos pasos y devuelve `CatalogAnalysis`, sin cambiar los comandos ni nombres de los archivos de salida.

Los registros de `history_` son dataclasses `KohonenEpoch` u `OjaEpoch`; sus campos se consultan como atributos. `evaluate(x)` devuelve `KohonenMetrics` u `OjaMetrics`. Se conserva `metrics(x)` como adaptador a diccionario. Los experimentos usan `RunMetrics`, `ExecutionResult`, `RunRecord` y `RunFailure`; los CSV y JSON mantienen su formato plano anterior.

```python
from dataclasses import asdict

print(som.history_[0].quantization_error)
print(oja.history_[0].direction_change)
print(som.evaluate(x).topographic_error)
print(som.metrics(x)['topographic_error'])  # API compatible
history_table = pd.DataFrame([asdict(epoch) for epoch in som.history_])
```

Los diccionarios se construyen en los adaptadores de serialización. En los registros de experimentos se valida que cada corrida tenga métricas o un fallo, exclusivamente.

## Configuración

Los JSON tienen `algorithm` (`kohonen` u `oja`), `data` (por defecto `data/creadores.csv`), `output` (por defecto `results`), `preprocessing` y `model`. Las claves desconocidas se rechazan.

### Preprocesamiento

| Parámetro | Predeterminado | Significado |
| --- | --- | --- |
| `include` | `null` | Todas las columnas numéricas; una lista selecciona columnas explícitas |
| `exclude` | Precio ARS, porcentaje orgánico, engagement total, alcance argentino estimado | Columnas excluidas de la selección |
| `log_columns` | `[]` | Columnas seleccionadas a transformar con `log1p` antes de escalar |
| `pca_components` | `null` | Sin reducción; un entero aplica PCA por SVD, sin whitening |

Cada columna se centra y divide por su desvío muestral. Los identificadores quedan en las exportaciones pero no entran a los modelos. Para incluir todas las variables numéricas, configurar `exclude: []`. Para probar la representación de 10 componentes del informe inicial, configurar `pca_components: 10`.

### Parámetros compartidos de entrenamiento

| Parámetro | Kohonen | Oja |
| --- | --- | --- |
| `seed` | 0 | 0 |
| `epochs` | 500 | 1000 |
| `shuffle` | true | true |
| `learning_rate_start` | 0.3 | 0.001 |
| `learning_rate_end` | 0.01 | 0.00001 |
| `decay` | exponential | exponential |

Una época es una pasada completa por los datos. Se registra también la cantidad de actualizaciones, igual a épocas × muestras. Los decaimientos admiten `constant`, `linear` y `exponential`; el progreso recorre todas las actualizaciones previstas, desde 0 hasta 1. En modo constante se usa el valor inicial y se ignora el final. Una parada anticipada puede impedir alcanzar el valor final.

`early_stopping` tiene `enabled` (false), `min_delta` (1e-6), `patience` (20) y `min_epochs` (30). Es opcional: hay un ejemplo activado en `configs/kohonen_early_stopping.json`.

- **Kohonen:** exige mejoras acumuladas superiores a `min_delta` absoluto en error de cuantización para reiniciar paciencia. Conserva por separado el mínimo observado. Si early stopping está habilitado, restaura ese mínimo al terminar, incluso cuando termina por épocas máximas. Se registran `best_epoch` y `selected_epoch`; el historial contiene todas las épocas efectivamente ejecutadas.
- **Oja:** cuenta épocas consecutivas con cambio de dirección `1 - |coseno| <= min_delta`. Se detiene al cumplir paciencia y mínimo de épocas. Conserva la dirección de la última época. El decaimiento muy pequeño también puede causar estabilidad: detenerse no demuestra que se haya encontrado PC1; hay que revisar la alineación con PCA.

### Kohonen

| Parámetro | Predeterminado | Opciones / significado |
| --- | --- | --- |
| `rows`, `cols` | 5, 5 | Lados de la grilla; total = filas × columnas, al menos dos neuronas |
| `topology` | rectangular | `rectangular`, `hexagonal` |
| `neighborhood` | gaussian | `gaussian`, `uniform` |
| `initialization` | samples | `samples`, `uniform` |
| `radius_start` | null | Se resuelve como mitad del mayor lado |
| `radius_end` | 0.5 | Radio final positivo |
| `radius_decay` | exponential | `constant`, `linear`, `exponential` |
| `schedule_epochs` | null | Duración del decaimiento; por defecto usa `epochs`. Si se fija un valor menor, mantiene los valores del extremo durante el resto del entrenamiento |

La inicialización `samples` toma muestras sin reemplazo si hay suficientes creadores; de lo contrario, con reemplazo. `uniform` toma valores dentro del rango observado de cada variable.

La BMU minimiza distancia euclídea en el espacio de entrada. La vecindad gaussiana tiene influencia `exp(-d²/(2r²))`; la uniforme actualiza las neuronas con distancia de grilla menor o igual al radio. Los empates se resuelven por orden fila-columna.

Ambas grillas son abiertas, sin conexión entre extremos:

- Rectangular: distancia euclídea entre posiciones para entrenar; adyacencia por lado compartido (hasta cuatro vecinos).
- Hexagonal: filas impares desplazadas a la derecha (`odd-r`), distancia por pasos hexagonales calculada con coordenadas cúbicas y hasta seis vecinos.

Los radios no son numéricamente equivalentes entre ambas geometrías. Los gráficos respetan las posiciones de las celdas y muestran sus índices `(fila,columna)`.

### Oja

Se implementa una sola neurona: `y = x @ w`, `w += η*y*(x-y*w)`. El modelo centra nuevamente su entrada y conserva esa media. Se inicializa con un vector aleatorio unitario y normaliza al terminar cada época como estabilización numérica. El historial registra su norma antes de normalizar, cambio de dirección, varianza capturada, proporción de varianza y alineación absoluta con PC1. La referencia PCA se calcula una vez al preparar el entrenamiento, no una vez por época.

El signo de PC1 es arbitrario. Se conservan pesos y scores originales y se exporta una versión de los scores alineada con PC1 solamente para comparar. El índice no implica calidad universal para todas las campañas.

## Vectorización y métricas

Las distancias de todas las muestras a todas las neuronas usan `||x||² + ||w||² - 2 X Wᵀ`, con corrección de negativos por redondeo. Se vectorizan BMU, actualización simultánea de pesos de todas las neuronas, proyecciones, ocupación y métricas. La geometría precalcula distancias y adyacencias. Se mantienen los bucles de épocas y muestras porque las actualizaciones online dependen de los pesos actualizados; convertirlas a un único producto matricial modificaría el algoritmo.

Kohonen exporta error de cuantización (media de distancia euclídea a la BMU) y error topográfico (fracción de muestras cuyas dos mejores neuronas no son adyacentes). La matriz U promedia distancias de pesos a vecinos inmediatos, con la misma adyacencia del error topográfico. Oja exporta alineación absoluta con PC1, correlación absoluta de scores, varianza capturada y su proporción del total.

El error de cuantización no se debe comparar directamente entre representaciones con distintas variables, escalados o componentes. Más neuronas pueden reducir ese error y aumentar celdas vacías: revisar conjuntamente ocupación, matriz U, error topográfico y estabilidad entre semillas. Las neuronas vacías son válidas.

## Barridos y resultados

`experiment` acepta además `seeds` (lista explícita de semillas distintas) y `grid` (producto cartesiano de listas de valores). Se admiten claves anidadas de `model` y `preprocessing`:

```json
{
  "algorithm": "kohonen",
  "model": {"rows": 4, "cols": 6, "epochs": 100},
  "seeds": [0, 1, 2],
  "grid": {
    "model.topology": ["rectangular", "hexagonal"],
    "model.learning_rate_start": [0.1, 0.3],
    "model.early_stopping.enabled": [false, true],
    "preprocessing.pca_components": [null, 10]
  }
}
```

Este ejemplo contiene 48 corridas; `--dry-run` permite revisar la cantidad. Se validan todos los parámetros antes de iniciar. Cada corrida usa su propia semilla y no depende del orden del barrido. Se conserva la configuración pedida y la configuración efectiva con valores predeterminados resueltos.

Cada ejecución crea un directorio dentro de `results/` (o del `output` del JSON), ignorado por Git, con el nombre del archivo de configuración y la fecha y hora local: `results/kohonen_hexagonal_2026-10-21_00-21-56`. Si dos ejecuciones empiezan en el mismo segundo, la segunda agrega `_2`. Desde Python, `run(config, name=...)` elige el nombre; por defecto es el algoritmo.

- Por corrida: configuración, parámetros de preprocesamiento, `model.npz`, historial, métricas y `learning_curves.png`. Se guarda además un único `checkpoint.npz` reemplazable y, al completar la corrida, un `record.json` que permite omitirla al reanudar. Kohonen agrega asignaciones con ID/nombre, geometría, ocupación y matriz U con PNG. Oja agrega proyecciones originales/alineadas y referencia PCA. En caso de fallo se guarda el error y su traceback.
- `runs.csv`: estado y métricas individuales. `summary.csv`: intentos, éxitos, media y desvío muestral por combinación, excluyendo la semilla del agrupamiento. Con un solo éxito, el desvío queda vacío; no se inventa estabilidad. `experiment.json` vincula configuraciones y grupos y registra estado (`running`, `paused`, `completed`), cantidad planificada y pendientes. `plan.json`, creado antes de entrenar, contiene todas las configuraciones efectivas, rutas absolutas e identificaciones de los datos y del código para reanudar.

Las corridas son secuenciales. No se selecciona automáticamente un ganador ni se comparan métricas de distintas representaciones como si tuvieran la misma escala. Los archivos contienen datos derivados del catálogo y permanecen fuera de Git.

## Pausar y reanudar

Durante `train` o `experiment`, presionar **Ctrl+C** solicita terminar la época actual, guardar el último checkpoint y pausar el experimento. Se imprime el comando para continuar:

```bash
python3 main.py resume --experiment results/<directorio-del-experimento>
python3 main.py resume --experiment results/<directorio-del-experimento> --dry-run
```

Un segundo Ctrl+C corta inmediatamente: se recupera desde el último checkpoint completo, repitiendo la época incompleta. No se generan copias históricas de los pesos: hay **un solo `checkpoint.npz` por corrida**, que permanece también cuando termina. Incluye pesos actuales, mejores pesos de Kohonen, época completada, estado aleatorio, contadores de early stopping e historial de mediciones. El preprocesamiento se guarda aparte y se restaura sin recalcularlo.

El checkpoint se escribe en un archivo temporal y reemplaza atómicamente al anterior. Una escritura fallida conserva el checkpoint previo. No se usa pickle. En RAM no se acumula una trayectoria de pesos; el historial conserva solo registros numéricos por época, como antes. Su memoria crece con la cantidad de épocas, pero cada registro es pequeño. En disco se reescribe un checkpoint comprimido por época: hay costo de E/S, además del entrenamiento.

La reanudación omite las corridas ya terminadas y continúa las incompletas desde la siguiente época. Si el corte ocurrió durante la exportación, restaura la última época y vuelve a exportar. No duplica mediciones ni cambia la duración original del decaimiento. Las corridas fallidas se conservan por defecto; para volver a intentarlas:

```bash
python3 main.py resume --experiment results/<directorio-del-experimento> --retry-failed
```

El comando usa el plan guardado; no acepta otra configuración. Comprueba que el CSV, el código y las versiones de Python/NumPy no hayan cambiado. Los datos transformados también se verifican contra el checkpoint. Las rutas guardadas son absolutas: se puede reanudar desde otro directorio usando una ruta correcta al programa, pero los datos deben seguir en la ubicación guardada. Los experimentos creados antes de esta funcionalidad no tienen el plan ni los checkpoints necesarios para reanudar el entrenamiento.

Los pesos de `model.npz` son el resultado final seleccionado; el checkpoint es el estado del entrenamiento, que puede diferir si Kohonen restaura su mejor época. Para uso directo en Python:

```python
from src.tp4.checkpoint import load_checkpoint, save_checkpoint

model = Oja(OjaConfig(epochs=1000))
model.fit(x, on_epoch=lambda state: save_checkpoint('checkpoint.npz', state))
# Después de interrumpir: misma configuración y misma matriz transformada.
model = Oja(OjaConfig(epochs=1000))
model.fit(x, checkpoint=load_checkpoint('checkpoint.npz'))
```

El callback guarda al finalizar cada época; la captura controlada de Ctrl+C pertenece a la CLI. `history.csv` y las curvas se exportan al finalizar o pausar; durante el entrenamiento las mediciones de cada época ya están persistidas dentro del checkpoint.

## Qué mirar por época en aprendizaje no supervisado

Tiene sentido registrar curvas de aprendizaje, aunque no hay una accuracy natural porque el catálogo no tiene etiquetas de grupos correctos.

| Modelo | Medición | Qué aporta |
| --- | --- | --- |
| Kohonen | Error de cuantización | Distancia media a la neurona ganadora; menor indica mejor representación de los datos |
| Kohonen | Error topográfico | Fracción de muestras cuyas dos mejores neuronas no son vecinas; menor indica mejor preservación de la organización local |
| Kohonen | Cambio de pesos | Estabilidad del mapa; un valor pequeño también puede deberse a una tasa muy baja |
| Oja | Alineación absoluta con PC1 | Cercanía a la solución de referencia; 1 significa misma dirección, independientemente del signo |
| Oja | Proporción de varianza capturada | Cuánto resume el índice; debería acercarse a la proporción de PC1, que no necesariamente es 100% |
| Oja | Cambio de dirección | Convergencia del vector; complementa la comparación con PC1 |

Se genera `learning_curves.png` con tres paneles por modelo. Las curvas pueden oscilar y las métricas de Kohonen pueden competir entre sí; no se exige una mejora monótona. Menor error de cuantización con más neuronas tampoco prueba que el mapa sea mejor para marketing. Comparar siempre semillas y representaciones equivalentes. Estas métricas evalúan el aprendizaje y la estructura; para afirmar mejoras en ventas, consultas o adopciones se necesitarían resultados reales de campañas.

## Alcance

Esta base sirve para entrenar y comparar implementaciones; falta realizar el estudio de hiperparámetros y la interpretación comercial final. El PCA inicial y sus decisiones se explican en [el informe](reports/ejercicio1/README.md). Los ejemplos cortos verifican funcionamiento, no convergencia definitiva ni superioridad de una topología.

## Batería amplia de Kohonen

Hay 2.600 corridas exploratorias preparadas en `configs/kohonen_suite/`, más ocho de prueba rápida. Cubren geometría, topología, aprendizaje, vecindad, representaciones, duración, early stopping, orden y combinaciones conjuntas. No se ejecutan automáticamente.

```bash
python3 -m experiments.kohonen_suite
python3 main.py experiment --config configs/kohonen_suite/00_smoke.json
python3 main.py experiment --config configs/kohonen_suite/01_geometry.json
python3 -m analysis.kohonen_experiments
```

El [protocolo de experimentos](docs/experimentos_kohonen.md) detalla presupuesto, casos acoplados, métricas, reanudación y confirmación con semillas nuevas. El comparador agrega ocupación, estabilidad ARI y cohesión en un espacio común; no ordena globalmente errores de distintas representaciones.

## Bitácora y evidencia publicada

La [bitácora del ejercicio 1](reports/ejercicio1/bitacora/README.md) registra experimentos ejecutados, decisiones y pendientes. Incluye un inventario CSV de estados y enlaza los resultados agregados de las 300 corridas de geometría; los datos y checkpoints permanecen en `results/`, fuera de Git.

## Desaprendizaje de Hopfield

Implementación y experimento reproducible de desaprendizaje antihebbiano sobre una única red con los cinco QR completos. Ver [protocolo, métricas y confirmación](docs/experimentos_desaprendizaje.md). Las [pruebas completas](reports/ejercicio2/desaprendizaje/PRUEBAS_COMPLETAS.md) comparan ambas representaciones, confirmación con ruido nuevo y daño espacial. Ver también la [bitácora del ejercicio 2](reports/ejercicio2/bitacora/README.md), la [comparación con pseudoinversa](reports/ejercicio2/desaprendizaje/PSEUDOINVERSA_VS_DESAPRENDIZAJE.md) y la [evidencia liviana publicada](reports/ejercicio2/desaprendizaje/evidencia/README.md). No se publican pesos ni checkpoints.

```bash
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_smoke.json
```

## Comparación de vecindades de Kohonen (26/10)

Las nuevas configuraciones individuales son `configs/kohonen_uniform_matched.json` (radio 2 → 0,5) y `configs/kohonen_uniform_neighbors.json` (2 → 1). Conservan el modelo gaussiano existente como referencia.

- [Repetición de finalistas, submuestras y augmentation](reports/ejercicio1/uniforme_20261026/README.md): 1.260 modelos nuevos frente a controles gaussianos verificados.
- [Versus con calibración equilibrada y confirmación anidada](reports/ejercicio1/versus_vecindades_20261026/README.md): 1.800 calibraciones y 90 confirmaciones; incluye radios constantes.

Desde TP4, con directorios de salida nuevos:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m analysis.kohonen_uniform_experiments --output /tmp/uniformes --workers 6
OPENBLAS_NUM_THREADS=1 python3 -m analysis.summarize_kohonen_uniform /tmp/uniformes
OPENBLAS_NUM_THREADS=1 python3 -m analysis.kohonen_neighborhood_versus --output /tmp/versus --workers 6
OPENBLAS_NUM_THREADS=1 python3 -m analysis.summarize_neighborhood_versus /tmp/versus
```

El primer experimento verifica y reutiliza las métricas gaussianas previas; el segundo ejecuta su propia calibración. Ambos permiten `--resume` y conservan solo métricas y asignaciones, sin pesos. La presentación no se modifica y sus hashes se comprueban. Después se incorporaron a la presentación: diapositiva 15 y apéndice 68–69 ([guía](presentation/experimentos_kohonen_vecindades.md)).
