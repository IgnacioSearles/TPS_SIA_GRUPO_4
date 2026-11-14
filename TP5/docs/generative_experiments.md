# Tres experimentos generativos nuevos

Esta suite vuelve al objetivo del ejercicio 2: reconstruir y generar muestras
plausibles de un dataset nuevo. No busca notas faltantes ni reabre la línea
cerrada de `ej2_pitch_*`. El VAE sigue siendo el MLP de NumPy con posterior
gaussiano y decoder libre; no se incorpora un sintetizador al decoder ni se
entregan al modelo parámetros geométricos o musicales.

## Datos, comparación y presupuesto

Cada dataset tiene **1.024 ejemplos de entrenamiento y 256 de prueba**. Los datos
se generan con `data_seed=20261113`, independiente de la semilla del modelo.
Se usan las semillas 7,17,27. Las imágenes de prueba no se usan para entrenar ni
para actualizar parámetros; se excluyen duplicados exactos entre ambos conjuntos.

Las tres tandas suman **18 modelos**. No requieren descargas ni nuevas dependencias.
`main.py` conserva las corridas terminadas y `--evaluate-only` vuelve a evaluar sin
entrenar. No usar `--force` al retomar; una corrida que no llegó a guardar resultados
debe comenzar desde cero porque no hay checkpoints por época.

## 1. Generar figuras geométricas

**Pregunta:** ¿el VAE genera nuevas imágenes reconocibles de la misma familia?

- Elipses y rectángulos centrados de 32×32, con ancho, alto y rotación variables.
- Antialias 4x y escala de grises en [0,1]. Las etiquetas y parámetros se guardan
  sólo como metadata para auditar el dataset; el encoder recibe 1.024 píxeles.
- Capas `[128,32]`, latente 4, ReLU/He, float32, BCE, beta KL=1 con calentamiento
  de 50 épocas. Adam 0,001, batch 64, 300 épocas.
- Tres corridas. El gráfico latente muestra PCA porque el latente tiene 4 dimensiones;
  no se supone que sus ejes correspondan a ancho, alto o rotación.

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_new_1_*.json --jobs 3 --threads 1
```

Inspeccionar `generated.png`, `generated_nearest.png`, `test_reconstruction.png`,
`interpolation.png` y `latent.png`. Una imagen alejada del entrenamiento también
podría ser un artefacto: la distancia no certifica novedad válida.

## 2. AE contra VAE y peso de KL

**Pregunta:** ¿cómo cambia la reconstrucción y la generación al regularizar el latente?

Se reutiliza el dataset exacto y el presupuesto del experimento 1. Se agregan:

- AE determinista: tres corridas, sin posterior ni KL.
- VAE con beta **0, 0,1 y 4**: nueve corridas. El VAE beta=1 ya lo aporta la
  primera tanda, por lo que no se entrena otra vez.

El VAE beta=0 sigue muestreando su posterior y aprendiendo una log-varianza: no
es idéntico al AE determinista. Se mantienen iguales las capas ocultas y el
latente; el VAE tiene la cabeza adicional de log-varianza. El calentamiento KL
es igual entre variantes VAE. Se usan las mismas coordenadas N(0,I) para generar
en todos los modelos; en el AE son una referencia común, no un prior aprendido.

```bash
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_new_2_*.json --jobs 3 --threads 1 &&
python -m src.analysis.generative_suite --shapes
```

Este informe necesita que haya terminado también la primera tanda. Queda en
`results/generative_shapes_report`: métricas por corrida, medias y desvíos entre
semillas, y gráficos. No se elige automáticamente un ganador ni se afirma que
el VAE siempre debe superar al AE en todas las métricas.

## 3. Timbres nuevos con frecuencia fija

**Pregunta:** ¿puede el VAE generar combinaciones de timbre nuevas sin variar la nota?

- Audio sintético mono de un segundo a 16 kHz y fundamental fija de **220 Hz**.
- Seis amplitudes de armónicos variables y envolventes continuas de ataque,
  liberación y duración. La fundamental está presente en todos los originales.
- El sintetizador crea el dataset; el decoder sigue generando **espectrogramas
  libres**, sin imponer relaciones armónicas a su salida.
- Códec mel de 64 bandas, FFT 1024 y salto 256. Latente 4, capas `[128,32]`,
  MSE, beta=0,5 con calentamiento 50, Adam 0,001 y 200 épocas. Tres corridas.

```bash
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_new_3_*.json --jobs 3 --threads 1 &&
python -m src.analysis.generative_suite --timbres
```

Se exportan muestras en `audio/generative/`, recorridos en `audio/interpolation/`
y ejemplos de prueba en `audio/test/`: `__raw` es el original, `__codec` el original
pasado por el códec y `__model` la reconstrucción. Escuchar los tres permite
distinguir pérdidas del códec y del modelo.

La afinación de las muestras y del códec se mide como diagnóstico. Acertar 220 Hz
no certifica que un timbre sea bueno, nuevo o armónicamente limpio. Esta prueba
tampoco demuestra generalización a otras alturas ni aprendizaje de una regla
musical universal.

## Ejecutar las tres tandas de una vez

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_new_*.json --jobs 3 --threads 1 &&
python -m src.analysis.generative_suite
```

Esto ejecuta 18 modelos en total y guarda la comparación en
`results/generative_all_report`. Para ejecutar en serie, quitar `--jobs 3 --threads 1`.
No se garantiza una duración: medir primero la tanda 1 permite estimarla en la
máquina real. Los comandos no ejecutan ninguna configuración de notas omitidas.

## Evaluaciones y límites

- `generative_manifest.json`: parámetros y etiquetas de entrenamiento/prueba,
  hashes de los datos, semillas y coordenadas de generación reproducibles.
- `test_reconstruction.csv/png`: MSE de cada imagen o espectrograma de prueba.
- `generated.csv/npz/png`: muestras desde N(0,I), coordenadas y distancia MSE al
  ejemplo de entrenamiento más próximo. No se declara «válida» una muestra sólo
  por no coincidir con el entrenamiento.
- `generated_nearest.png`: muestra generada junto a su vecino más próximo para
  inspeccionar posibles copias y artefactos.
- Diversidad: MSE media entre pares distintos de muestras generadas. Puede ser
  alta por ruido o baja por colapso; no es un puntaje de calidad.
- Las métricas de formas y audio tienen unidades distintas y se comparan en
  paneles separados, no en un ranking común.

Las comparaciones verifican que datos, arquitectura oculta, presupuesto y conjunto
de coordenadas no hayan cambiado entre modelos de una misma familia. La
plausibilidad se evalúa con las imágenes y audios, además de esas métricas. Los
parámetros descriptivos del dataset no se usan para supervisar el encoder.

## Control adicional: distribución ajustada del AE

Una mala generación del AE desde N(0,I) no demuestra que el AE sea incapaz de
generar: su entrenamiento no lo obliga a colocar códigos en esa distribución.
Este control reutiliza los pesos existentes, sin cambiar las configuraciones ni
los resultados previos de las corridas.

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.latent_sampling
```

Requiere las 15 corridas de figuras terminadas. Compara AE con normal estándar,
AE con gaussiana ajustada y VAE beta=1 con normal estándar, en las semillas
7,17,27. Para el ajuste se codifican exclusivamente las 1.024 imágenes de
entrenamiento. Se calcula la media y covarianza completa muestral (N-1); una
raíz simétrica por descomposición espectral permite covarianzas singulares.
Solo se recortan autovalores negativos por redondeo, sin ajustar parámetros en
prueba. Se generan 64 muestras con la misma matriz normal base de semilla 1234.

Las salidas quedan en `results/generative_latent_sampling_report`: CSV por
muestra/corrida, medias y desvíos entre semillas, mosaicos completos y vecinos
de entrenamiento. Las comparaciones compactas muestran los índices prefijados
0,8,...,56. También se guardan la gaussiana, coordenadas, imágenes y hashes de
pesos/configuraciones para auditar el origen. Se comprueba que las generaciones
normales coincidan con las guardadas anteriormente. No se reentrena ni se
sobrescriben los pesos o las evaluaciones originales.

Una única gaussiana puede cubrir regiones poco pobladas del latente y producir
artefactos. Diversidad y distancia al entrenamiento son diagnósticos, no
certificados de calidad o novedad. Esta distribución tampoco iguala los
objetivos de entrenamiento del AE y del VAE. El ajuste posterior de densidad
se inspira en [From Variational to Deterministic Autoencoders](https://arxiv.org/abs/1903.12436);
este control simple no reproduce todos los métodos ni resultados de ese trabajo.
