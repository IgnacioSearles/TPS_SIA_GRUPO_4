# CNN sobre NSynth: reconstrucción, generación y denoising

Este estudio compara redes densas y convolucionales sobre sonidos reales de NSynth. Reutiliza el espectrograma mel y los modelos existentes. No intenta encontrar notas omitidas ni modifica las pérdidas: AE con MSE y VAE con MSE más β KL.

Los tres experimentos comparten la partición y aprovechan doce entrenamientos: VAE denso, VAE convolucional, AE denoising denso y AE denoising convolucional, cada uno con semillas 7, 17 y 27. Los dos VAE sirven tanto para reconstrucción como para generación libre. El protocolo está en `results/nsynth_cnn_study/protocol.json`.

## Datos y separación

La selección coincide con `configs/ej2_vae_nsynth.json`: MIDI 48–72, hasta 150 notas por familia, semilla de selección 0. El códec usa un segundo a 16 kHz, FFT 1024, salto 256 y 128 bandas mel. Cada ejemplo contiene 128 × 64 = 8192 valores y el latente tiene 16 dimensiones.

La selección disponible tiene **1108 notas, 53 instrumentos exactos y 10 familias**. Se reserva aproximadamente el 20% de los instrumentos de cada familia. El rango se fija mediante SHA256 de semilla 20261115, familia e identificador completo del instrumento. El número reservado redondea al entero más próximo y exige al menos un instrumento de cada lado. Todas las notas y velocidades de un mismo instrumento quedan en la misma partición. Una familia con menos de dos instrumentos detendría la preparación; no hay sustitución automática por una división de notas.

| Familia | Instrumentos train / test | Notas train / test |
|---|---:|---:|
| bass | 8 / 2 | 124 / 26 |
| brass | 4 / 1 | 100 / 24 |
| flute | 1 / 1 | 41 / 6 |
| guitar | 6 / 1 | 139 / 11 |
| keyboard | 6 / 2 | 114 / 36 |
| mallet | 2 / 1 | 41 / 12 |
| organ | 5 / 1 | 117 / 33 |
| reed | 3 / 1 | 84 / 33 |
| string | 5 / 1 | 102 / 9 |
| vocal | 1 / 1 | 22 / 34 |
| **Total** | **41 / 12** | **884 / 224** |

El porcentaje objetivo se aplica a instrumentos, no a notas. Flauta y voz tienen dos instrumentos y requieren una separación 1/1. La distribución por familia queda desbalanceada, de modo que deben leerse tanto resultados agregados como resultados por familia y semilla.

`src/data/nsynth_cnn_study.py` prepara `results/nsynth_cnn_study/data/manifest.json` con etiquetas, instrumento, familia, fuente, altura, velocidad, partición y hash del espectrograma de cada nota. Los hashes usan float32 little-endian en orden C. `train_settings.json` usa `dataset.subset`, ya soportado por la fábrica. Todos los modelos reciben exactamente estas 884 etiquetas; los 224 ejemplos de prueba se cargan para evaluar.

Los modelos NSynth anteriores entrenaron con la selección completa, incluidos los nuevos instrumentos reservados. Por eso no se reutilizan sus pesos para presentar una comparación de generalización sobre este test.

## Los tres experimentos

1. **Reconstrucción:** comparar la salida de los VAE densos y convolucionales con los 224 espectrogramas de instrumentos reservados. Se reporta MSE por nota, familia y semilla. La escucha distingue original, original pasado por el códec y reconstrucción para separar defectos del códec de defectos del modelo.
2. **Generación libre:** decodificar 64 vectores de una normal estándar por semilla, compartidos entre los dos VAE. Se inspeccionan diversidad y distancias a los conjuntos reales. Una distancia grande no demuestra novedad y una distancia pequeña no garantiza buena calidad. Las primeras seis generaciones se fijan como ejemplos audibles antes de observar los resultados.
3. **Denoising:** entrenar AE densos y convolucionales con ruido gaussiano de desviación 0,1 sobre los valores mel normalizados, recortados a [0,1]. La salida objetivo sigue siendo el mel limpio. En prueba se evalúan niveles 0, 0,05, 0,1 y 0,2, comparando con conservar la entrada contaminada. El nivel cero permite detectar si el modelo deteriora entradas limpias.

Las condiciones usan latente 16, capas densas [512,128], ReLU, float32, Adam 0,001, batch 64 y 200 épocas. El VAE usa β = 0,5 y calentamiento de 50 épocas. Las CNN agregan canales [8,16], kernels 3×3 y stride 2 en encoder. El decoder usa aumento de resolución por repetición y convoluciones. No se cambian ni MSE ni KL.

En denoising, la contaminación de prueba se comparte entre denso y CNN por semilla mediante `SeedSequence([seed,20261115,2])`. Los latentes de generación usan `SeedSequence([seed,20261115,1])`. Las variantes y las notas de una misma corrida no cuentan como entrenamientos independientes.

## Ejemplos fijados

Las seis referencias audibles son la primera etiqueta ordenada de prueba en cada una de las primeras seis familias en orden alfabético:

- `bass_electronic_027-048-025`
- `brass_acoustic_046-053-025`
- `flute_acoustic_002-067-025`
- `guitar_acoustic_030-048-025`
- `keyboard_acoustic_004-050-025`
- `mallet_acoustic_056-048-127`

La elección permite representar seis familias sin seleccionar los mejores resultados. **No sustituye las métricas de las 224 notas.** El manifiesto conserva además tres ejemplos fijos por familia y partición. `example_originals.npz` contiene los seis audios originales normalizados y recortados a un segundo; su codificación coincide con los espectrogramas de la caché dentro de tolerancia 1e-7.

## Límites de interpretación

Reservar instrumentos completos mide generalización a instrumentos no vistos dentro de familias conocidas. Es más exigente que reservar notas aleatorias del mismo instrumento. Las alturas de prueba pueden estar presentes en otros instrumentos de entrenamiento: esto no es una evaluación de alturas ausentes.

El ruido se agrega **en el espectrograma mel normalizado**. No representa una grabación con ruido acústico, reverberación o interferencias reales, y su nivel no equivale directamente a una relación señal/ruido en dB. Una mejora permite afirmar que el modelo elimina esta corrupción artificial de la representación. Para hablar de limpieza de grabaciones reales haría falta un estudio distinto con contaminación sobre la onda.

El códec recupera fase por Griffin-Lim y puede introducir defectos audibles aun con un espectrograma original. Las comparaciones deben incluir ese control. Las métricas de cercanía y MSE no reemplazan la escucha ni demuestran estructura musical. El resultado favorable requiere consistencia entre semillas, no elegir una única imagen o sonido atractivo.

## Ejecución y reanudación

Desde WSL:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 && source .venv/bin/activate && bash scripts/run_nsynth_cnn_study.sh 6
```

El ejecutor usa seis procesos, cada uno con un hilo numérico. Un lock evita lanzar dos ejecutores del mismo estudio. Guarda checkpoint cada cinco épocas, reutiliza corridas terminadas y puede reanudarse con el mismo comando después de una interrupción. No lanzar otra copia si ya está activo.

Para preparar datos y configuraciones sin entrenar:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 && source .venv/bin/activate && OPENBLAS_NUM_THREADS=1 python -m src.analysis.nsynth_cnn_protocol --prepare
```

Para evaluar las corridas terminadas sin entrenar:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 && source .venv/bin/activate && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.analysis.nsynth_cnn_evaluation --jobs 2
```

Resultados y gráficos: `reports/nsynth_cnn_study/`. Configuraciones: `configs/nsynth_cnn/`. Referencias, manifiesto y protocolo: `results/nsynth_cnn_study/`. El informe parcial debe señalar las corridas pendientes y conservar los resultados negativos.

## Duración equiparada al repositorio

Las doce corridas usan 200 épocas, igual que ej2_vae_nsynth, ej2_ae_nsynth, los controles de muestreo NSynth y las comparaciones de audio/notas omitidas. Se corrigió el límite inicial de100 a200 mediante una migración auditada de checkpoints: solo cambió el máximo de épocas en la firma de configuración; pesos, momentos de Adam, historia y estados RNG se conservaron. Hay copias previas y verificación en results/nsynth_cnn_study/epoch_extension_audit.json. Esto equipara el presupuesto, pero no constituye una garantía matemática de convergencia.
