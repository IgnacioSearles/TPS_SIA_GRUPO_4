# VAE convolucional de audio: comparación fijada

## Pregunta

¿Aprovechar la estructura local del espectrograma mejora la reconstrucción y la
interpolación de alturas ausentes respecto del VAE denso?

Esta etapa reabre **solo esta comparación de audio**, autorizada el 15/11/2026.
El estudio de figuras y las presentaciones siguen intactos. No se inicia una
búsqueda de hiperparámetros ni se modifica la pérdida de AE/VAE.

## Diseño

Cuatro configuraciones en `configs/audio_conv/`, cada una con semillas 7, 17 y 27:
MLP completo, MLP con omisiones, CNN completo y CNN con omisiones. Son doce
entrenamientos nuevos e independientes. No reutilizamos pesos de otra etapa ni
tocamos resultados anteriores. Las corridas terminadas de esta etapa se reutilizan.

- Dataset sintético MIDI 48–72, cuarenta envolventes por semitono, microtonos cada
  cuarto de semitono y diez variantes por microtono. Timbre de seis armónicos 1/k.
- Mismas configuraciones de validación de la etapa anterior: objetivos 51, 57 y 63;
  variantes v00, v20 y v39. La exclusión retira **todas** las variantes y microtonos
  a distancia <=0,75 semitonos de cada objetivo, antes de formar minibatches.
- Las referencias finales 55, 61 y 67 y su entorno se excluyen de ambos modos,
  no se calibran ni evalúan. El control llamado completo incluye los objetivos de
  validación, no estas referencias finales.
- 1.420 muestras en completo y 1.120 con omisiones. Cada arquitectura recibe
  exactamente los mismos datos en el mismo modo; la auditoría registra sus hashes.
- Códec existente: 1 s, 16 kHz, FFT 1024, salto 256, 128 bandas mel × 64 tiempos,
  70 dB, inversión mel 200 iteraciones y Griffin–Lim 60. No se cambia representación.
- Ambos: latente 16, ocultas [512,128], ReLU, He, float32, salida sigmoide,
  MSE + KL con beta 0,5 y calentamiento de 50 épocas, Adam 0,001, batch 64,
  200 épocas. Sin pérdida auxiliar de interpolación ni etiquetas de altura.

Se mantienen las mismas fórmulas protegidas de `losses.py`, `autoencoder.py`
y `variational.py`. No fue necesario cambiar la implementación del modelo:
la arquitectura espacial existente ya admite rectángulos de un canal.

## Arquitectura CNN

El espectrograma conserva el orden **frecuencia × tiempo**, sin reinterpretar
filas como canales:

```
128×64×1 -> Conv3 stride2 -> 64×32×8 -> Conv3 stride2 -> 32×16×16
-> flatten 8192 -> 512 -> 128 -> mu/logvar (16+16) -> z (16)
z -> 128 -> 512 -> 8192 -> 32×16×16
-> ampliar ×2 -> Conv3 -> 64×32×8 -> ampliar ×2 -> Conv3 -> 128×64×1
-> sigmoide
```

Hay convoluciones en **encoder y decoder**. Sin pooling, convoluciones transpuestas,
refinamientos adicionales, transferencias desde RGB ni reconstrucción paramétrica.
El modelo aprende a producir todas las bandas; no le damos la serie armónica.

Los canales 8/16 dejan 8.192 características antes de las capas densas, igual
que la entrada del MLP. Los tamaños finales son 8.535.328 y 8.537.809 parámetros:
CNN agrega 2.481 (0,029 %). Se fijó esta configuración antes de observar resultados.
Igual cantidad de parámetros no significa igual costo ni idéntica capacidad efectiva.

## Evaluación predefinida

1. Calibrar originales y originales pasados por el códec; menos de 90 % de
   aciertos por variante vuelve inconclusa la evaluación de afinación.
2. En controles completos, reconstruir directamente los objetivos conocidos.
   Esto distingue problemas de síntesis de los específicos de interpolación.
3. En ambos modos, codificar solo los vecinos enteros inmediatos. Decodificar
   once fracciones uniformes. La predicción principal es siempre el punto 0,5.
4. Comparar con mezcla mel al 50 % y vecino inferior. El objetivo reservado
   aparece únicamente después de generar, al medir.
5. Conservar todos los fallos: no evaluable cuenta como fallo; error de altura
   <=50 cents, sin corregir octavas. También medir MSE espectral y si el objetivo
   es el más próximo de {inferior,objetivo,superior}. Reportar la conjunción de
   ambas propiedades y las tablas por semilla, sin elegir la mejor semilla.

El criterio previo de altura se conserva: >=8/9 por cada variante y tasa mayor
que ambas referencias sin aprendizaje. Se necesitan las doce corridas y calibración
fiable para declarar un resultado favorable; un informe parcial no elige ganador.
Ese criterio no certifica espectro, novedad ni calidad auditiva. Las 27 predicciones
por arquitectura/modo contienen variantes relacionadas; no son 27 entrenamientos.

La comparación CNN/MLP se informa por semilla y total. Se revisan reconstrucciones
de los controles, trayectorias completas y WAV. No se concluye éxito por una mejora
solo en la estimación de fundamental ni por la MSE de entrenamiento. Un resultado
negativo produce un informe válido y no dispara ajustes automáticos.

## Ejecutar y continuar tras una interrupción

El ejecutor de ambas fases, con auditoría e informes automáticos, es:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 && source .venv/bin/activate && bash scripts/run_audio_convolution.sh 3
```

El bloqueo evita iniciar otra copia si ya está activo. El lanzamiento inicial de
esta etapa se hizo en segundo plano; su salida queda en
`results/audio_convolution/run.log`. Para verla: `tail -f results/audio_convolution/run.log`.
No hace falta volver a iniciar el comando mientras esa ejecución siga activa.

Desde WSL, activar el entorno y verificar el protocolo:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 && source .venv/bin/activate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.analysis.audio_convolution --audit
```

El control completo se ejecuta primero; luego las omisiones. Esto permite revisar
la síntesis conocida antes de interpretar alturas no vistas. La comparación
conserva todas las corridas, aun cuando una rama dé resultados negativos.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u main.py \
  'configs/audio_conv/*_complete.json' --jobs 3 --threads 1 --checkpoint-every 10
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u main.py \
  'configs/audio_conv/*_omitted.json' --jobs 3 --threads 1 --checkpoint-every 10
python -m src.analysis.audio_convolution
```

Para continuar, repetir esos comandos **sin `--force`**: se omiten las corridas
terminadas y se reanudan los checkpoints. Se conserva Adam, RNG y época; podrían
repetirse hasta nueve épocas desde el último checkpoint. Al terminar el entrenamiento
y antes del resumen también puede reanudarse para repetir solo la evaluación final.

Solo reevaluar pesos guardados:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u main.py \
  'configs/audio_conv/*.json' --evaluate-only --jobs 3 --threads 1
python -m src.analysis.audio_convolution
```

El informe en `reports/ej2_audio_convolution/` contiene `protocol.json`,
`summary.json`, `informe.md`, `comparison.png`, `comparison.csv`, `per_seed.csv`,
`runs.csv` y `trajectories.csv`. Los WAV y gráficos individuales quedan en
`results/ej2_audio_conv_*/base__seed=*/`.

Las cuatro configs de `configs/audio_conv/smoke/` son pruebas técnicas separadas
de dos épocas y pocas notas. No forman parte de los resultados del experimento.

## Límites

Los filtros locales podrían ayudar con patrones espectrales; no garantizan una
coordenada latente lineal en altura. Desplazar un kernel en mel tampoco equivale
exactamente a transponer una nota. El códec sigue siendo un límite compartido.
Esta prueba aborda sonidos sintéticos: no se extrapola un eventual éxito a NSynth.
Si mejora, se propondría una etapa independiente con instrumentos reales; no se
lanza esa etapa ni se abren las referencias finales automáticamente.
