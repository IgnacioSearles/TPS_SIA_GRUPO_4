# Notas omitidas: interpolación musical con el VAE

> **Estudio cerrado (13/11/2026).** Esta guía conserva el protocolo histórico.
> Ver [resultados y decisión de cierre](../reports/ej2_notas_omitidas/cierre.md).
> No se propone continuar las corridas finales.

Este experimento pregunta si la interpolación entre notas vecinas produce una
altura ausente del entrenamiento. No completa progresiones ni descubre huecos
de forma autónoma. El VAE, su pérdida, backpropagation y el entrenador no cambian.
Se amplían los datos y las evaluaciones, reutilizando el códec mel/Griffin-Lim.

## Versión ampliada: 1.000 sonidos

Las configuraciones `ej2_pitch_expanded_synthetic_complete` y
`ej2_pitch_expanded_synthetic_omitted` agregan 40 envolventes por altura MIDI
48–72: cuatro ataques (10, 30, 70, 120 ms), cinco liberaciones (50, 150, 300, 450,
600 ms) y dos duraciones efectivas (80 % y 100 % del archivo). El timbre sigue
siendo el mismo conjunto de seis armónicos. Cada combinación se sintetiza
directamente en su altura; no se transpone ni se duplica una referencia omitida.

El control usa 1.000 sonidos y la condición con omisiones usa 880. Se excluyen
las 40 variantes de cada una de las alturas 54, 60 y 66: 120 sonidos reservados.
El manifiesto registra la variante y sus parámetros de envolvente.

Con batch 64 y 200 épocas hay 2.800 actualizaciones en la condición con omisiones
y 3.200 en el control, frente a 200 en cada corrida pequeña. No se aumenta además
el número de épocas ni se cambian la arquitectura, KL u optimizador.

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5
source .venv/bin/activate
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_pitch_expanded_synthetic_*.json
python -m src.analysis.pitch_study \
  results/ej2_pitch_expanded_synthetic_complete \
  results/ej2_pitch_expanded_synthetic_omitted \
  --output results/pitch_expanded_report
```

Son seis modelos (dos condiciones por tres semillas). Sus configuraciones y
directorios son nuevos: no modifican ni reanudan las corridas pequeñas que ya
estén ejecutándose. NSynth conserva su protocolo original de grabaciones reales;
esta ampliación corresponde al control sintético y no se presenta como un
aumento artificial de las grabaciones de NSynth.

Para evaluar se fijan ANTES de entrenar tres envolventes: `v00`, `v20`, `v39`.
Los vecinos y el objetivo de cada caso comparten esa misma envolvente. Se
recorren las once fracciones para las tres notas y las tres envolventes; no se
seleccionan después los casos que suenan mejor. Se calibra el códec sobre todas
las 25 alturas de esas tres variantes (75 referencias). Cada variante debe
alcanzar al menos 90 % de aciertos de calibración.

El informe conserva 27 predicciones centrales del VAE por condición. Para
evidencia favorable se exige al menos 8/9 aciertos **en cada variante** y superar
ambos baselines en la tasa total. La evaluación y sus CSV identifican la variante;
los audios usan directorios como `audio/pitch/60_v20/`. Las restantes 37 envolventes
contribuyen al entrenamiento, pero no se evalúan en este protocolo fijo.

No combinar el estudio pequeño y el ampliado en un mismo agregado: son datasets
distintos; el analizador rechaza esa mezcla. Se pueden comparar después sus
informes separados, sin tratar sus cantidades de ejemplos como equivalentes.

## Ejecutar el estudio pequeño original (referencia)

Desde `TP5`, usar el entorno virtual y las dependencias de `requirements.txt`:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest

# Primero el control sintético: seis modelos (dos condiciones por tres semillas).
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_pitch_synthetic_*.json

# NSynth no está incluido en Git. Descargar una vez (aprox. 349 MB).
python scripts/download_nsynth.py
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_pitch_nsynth_*.json

# Informe conjunto; admite también solo los dos experimentos sintéticos.
python -m src.analysis.pitch_study \
  results/ej2_pitch_synthetic_complete results/ej2_pitch_synthetic_omitted \
  results/ej2_pitch_nsynth_complete results/ej2_pitch_nsynth_omitted \
  --output results/pitch_study_report
```

Los comandos dejan resultados en directorios nuevos `results/ej2_pitch_*` y no
modifican los experimentos anteriores. Se reutilizan las corridas terminadas.
Para regenerar audio y métricas sin entrenar:

```bash
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_pitch_synthetic_*.json --evaluate-only
```

No cambiar el dataset, las semillas ni los hiperparámetros dentro de una carpeta
con resultados: usar otro nombre de experimento. La evaluación comprueba el
manifiesto previo para evitar mezclar referencias distintas. `--evaluate-only`
permite cambiar las iteraciones de inversión del códec y recalibrar las medidas;
usar los mismos parámetros en todas las corridas que se agregan en un informe.

## Protocolo fijo

Las cuatro configuraciones conservan los parámetros de `ej2_vae_nsynth`: latente
16, capas `[512,128]` y decoder espejo, ReLU, He, precisión float32, pérdida MSE,
KL 0,5 con calentamiento de 50 épocas, Adam 0,001, 200 épocas, batch 64 y el mismo
códec (1 s, 16 kHz, FFT 1024, salto 256, 128 bandas mel). Semillas 7, 17 y 27.

**Sintético:** MIDI 48–72, seis armónicos con amplitud `1/k`, fases cero, ataque
y liberación lineales de 50 ms, normalización de pico. Se reservan 54, 60 y 66.
No hay ruido, cambios de afinación ni aumentos derivados de los objetivos.

**NSynth:** mayor tramo cromático consecutivo de un instrumento exacto y una
velocidad dentro de 48–72. Se desempata por identificador de instrumento,
velocidad numérica y comienzo del tramo, ascendentes. Se exige al menos 13 notas;
no se mezclan grupos para completar huecos. Los tres objetivos ocupan las
posiciones `floor((N-1)*k/4)`, para `k=1,2,3`. Falta de archivo o cobertura genera
un diagnóstico antes de entrenar. La elección no depende de calidad acústica ni
de resultados: el control del estimador puede volver inconcluso un grupo ruidoso.

Cada dataset se entrena con y sin sus tres objetivos. El manifiesto contiene
fuente, pitch MIDI, etiqueta, instrumento, velocidad, hash de la forma de onda,
semilla y pertenencia a entrenamiento o referencia. Las muestras reservadas
quedan en metadatos de evaluación: el Trainer recibe solo la matriz filtrada.

El encoder recibe exclusivamente los vecinos `p-1` y `p+1`. Se interpolan sus
medias posteriores en 11 puntos uniformes. El punto 5 (fracción 0,5) es la
predicción principal: no se busca el mejor punto después de escuchar el objetivo.
Los dos baselines usan la media de los espectrogramas mel normalizados y el
vecino inferior. El control completo agrega reconstrucción del objetivo conocido.
Las referencias reservadas solo se codifican para medir el códec y mostrar el
objetivo; no se pasan al encoder del modelo con omisiones.

## Configuración e interfaces

```json
{
  "dataset": {
    "type": "synthetic_notes",
    "pitch_range": [48, 72],
    "pitch_study": {"omit_targets": true, "targets": [54, 60, 66]},
    "spectrogram": {"mel_bands": 128}
  },
  "evaluation": {"pitch_interpolation": {}}
}
```

Este fragmento muestra las opciones nuevas; usar los JSON completos provistos
para reproducir el protocolo. `pitch_study.targets` omitido elige cuartiles del
tramo. `omit_targets: false` crea el control conocido. `exclude_pitches` permite
exclusiones explícitas adicionales y se aplica antes de selección aleatoria;
no puede eliminar vecinos necesarios. En NSynth se agregan `instrument`
(identificador como `keyboard_acoustic_001`) y `velocity` (entero 0–127), también
disponibles fuera del estudio. Todas estas opciones participan en la caché.
El estudio no admite `subset` ni `max_per_family`, para conservar su integridad.

## Métricas y archivos

La autocorrelación normalizada usa ventanas de 4096 muestras, salto 1024 y rango
80–1200 Hz. Se descartan ventanas con RMS menor a `1e-4`; se aceptan picos ≥ 0,6.
Se elige el primer pico a no más de 0,02 del máximo y se interpola parabólicamente.
Se requieren tres ventanas válidas; la frecuencia final es su mediana. No se usa
la altura objetivo para elegir el período ni se corrigen errores de octava.

Un acierto exige error absoluto ≤ 50 cents. Silencios, sonidos sin estimación y
errores de octava cuentan como fallos. Las mezclas polifónicas pueden presentar
un período común engañoso: este estimador no certifica monofonía ni calidad sonora.

Por corrida se exportan:

- `pitch_manifest.json`: datos, partición y parámetros del protocolo.
- `pitch_calibration.csv`: afinación de toda la escala original y pasada por el códec.
- `pitch_interpolation.csv`: semilla, objetivo, método, fracción, Hz, cents, ventanas
  válidas, validez y acierto; conserva los 11 puntos y todos los fallos.
- `pitch_summary.json`, `pitch_trajectories.png` y `pitch_spectrograms.png`.
- `audio/pitch/references/`: originales y referencias procesadas por el códec.
- `audio/pitch/<MIDI>/`: once pasos, punto medio, secuencia y baselines; en los
  controles completos también la reconstrucción del objetivo conocido.

El informe agregado genera `midpoints.csv`, `summary.json` y `comparison.png`,
incluyendo resultados de cada semilla. No mezcla datasets, códecs o modelos
distintos dentro de una comparación. Rechaza duplicados y tablas incompletas;
si faltan semillas o condiciones informa `incomplete`, sin declarar éxito.

Si menos del 90 % de las referencias completas procesadas por el códec acierta,
la conclusión es `inconclusive_codec`. Con las seis corridas de un dataset
completas y calibradas, la evidencia es favorable si al menos 8 de las 9
predicciones omitidas aciertan y su tasa supera estrictamente ambos baselines.
No alcanzar el criterio produce `not_favorable`: sigue siendo un resultado válido.
Un dataset con un solo ejemplo por nota es un control exploratorio pequeño;
estas nueve predicciones no son nueve datasets independientes.

La escucha y la inspección de espectrogramas complementan la afinación. Un tono
correcto puede tener timbre pobre; un resultado negativo puede revelar limitaciones
del códec o del modelo. No ajustar parámetros usando los objetivos reservados:
un estudio posterior debe elegir nuevos objetivos de evaluación.
