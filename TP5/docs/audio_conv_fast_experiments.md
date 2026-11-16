# Experimentos rápidos para interpretar la CNN de sonido

La comparación principal de doce entrenamientos continúa sin cambiar sus configuraciones. Corte inicial de esta tanda: seis controles completos terminados; las tres CNN con notas omitidas estaban entre las épocas150 y160. Las notas de validación son51,57,63 y las variantes de envolvente v00,v20,v39. Las notas55,61,67 permanecen reservadas, sin evaluar.

En controles completos, el denso acierta altura16/27 y altura junto con criterio espectral8/27. La CNN obtiene15/27 y11/27, respectivamente. La MSE de reconstrucción directa es0.000279 en el denso y0.000268 en CNN. Estas diferencias son pequeñas, y las27 combinaciones no representan27 semillas independientes: solo hay tres entrenamientos por arquitectura. La prueba decisiva de alturas omitidas todavía no estaba completa al iniciar esta tanda.

## Mezcla antes o después del decoder

`src/analysis/audio_conv_path_diagnostic.py` usa los seis modelos completos congelados. Compara decodificar la mezcla de medias posteriores, mezclar las dos reconstrucciones de vecinos y mezclar los espectrogramas de entrada. Ningún método utiliza el objetivo para generar. El objetivo participa únicamente en las mediciones posteriores.

La altura y el criterio conjunto se evalúan en el punto medio predefinido. También se exportan distancias espectrales de las fracciones0.25 y0.75 al objetivo central, como diagnóstico. No se elige el mejor punto ni se interpreta que esos puntos deban coincidir con la nota central. El propósito es determinar qué aporta la no linealidad del decoder sobre una mezcla de dos notas, antes de aumentar el tamaño de las redes.

Comando reproducible:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m src.analysis.audio_conv_path_diagnostic --jobs 2
```

Resultados, WAV, CSV por semilla, gráfico e informe: `reports/audio_conv_path_diagnostic/`.

## Piloto de regularización KL

Se congelaron copias de parámetros de las CNN con notas omitidas: semilla7 en época160, semilla17 en época150 y semilla27 en época160. El estudio original sigue escribiendo sus propios checkpoints, que no se modificaron. Las tres ramas de cada semilla comparten exactamente el mismo padre congelado, verificado mediante SHA256.

Las ramas usan beta0.05,0.5 y1.0. Cada una hace20 épocas adicionales, Adam0.0003, batch64, sin warmup, con el resto de arquitectura y datos idéntico. Son nueve corridas con seis trabajadores y un hilo BLAS por trabajador. No se cambia la fórmula MSE+betaKL, no se agrega supervisión de altura ni se cambia el códec mel.

El beta0.5 es un control con el mismo reinicio de Adam y de los generadores aleatorios. El resultado debe compararse con este control, no directamente con el entrenamiento principal de200 épocas. La diferencia de épocas padre entre semillas impide describir la tanda como un único presupuesto común, pero no altera la comparación pareada de beta dentro de una semilla. Este es un piloto exploratorio de continuaciones cortas, no una comparación de beta desde cero.

Beta bajo explora si una menor penalización KL permite representar mejor el espectro. Beta alto explora si una presión mayor hacia la distribución normal mejora la regularidad latente. Ninguna garantiza una trayectoria musicalmente correcta. Las tres notas de validación ya guiaron búsquedas previas y no deben presentarse como un test final nuevo.

Para continuar después de un reinicio:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5
bash scripts/run_audio_conv_beta_pilot.sh 6
```

El lock evita duplicar ejecutores. Las corridas terminadas se reutilizan y los checkpoints guardan Adam y RNG cada cinco épocas. El inicio de cada rama desde sus pesos padre es una fase nueva de optimización, mientras que reanudar un checkpoint de esa rama sí restaura su estado completo. Mantener las copias de `results/audio_conv_beta_pilot/parents/`: los checkpoints del estudio principal pueden avanzar y no reconstruyen necesariamente esos padres.

```bash
tail -f results/audio_conv_beta_pilot/run.log
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.analysis.audio_conv_beta
```

Informe del piloto: `reports/audio_conv_beta_pilot/informe.md`. El ejecutor lo actualiza al terminar las nueve corridas. Un resultado negativo se conserva como resultado válido y no dispara más búsquedas automáticamente.

## Cómo tomar una decisión

Primero completar el contraste CNN/denso en las notas omitidas. Si ambas reconstruyen bien pero interpolan mal, no se justifica ampliar la red solo por mejorar su reconstrucción. El diagnóstico de mezcla indica si el decoder aporta estructura útil o se limita a mezclar vecinos. El piloto beta indica sensibilidad a la regularización, sin probar por sí solo generalización a otras notas o timbres. No se abren las notas reservadas ni se elige un ganador automáticamente después de estos pilotos. La presentación no se modifica durante esta tanda.

## Estado posterior

El usuario canceló las tres ramas beta para priorizar NSynth. Se detuvieron sus nueve corridas y se conservaron los checkpoints. No se presentan como una comparación concluida. Ver reports/audio_conv_beta_pilot/cancelled.json.
