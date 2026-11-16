# Ablación del calentamiento KL en NSynth

Comparamos beta0.5 desde la primera época con una rampa lineal hasta beta0.5 durante50épocas. Todos los entrenamientos duran200épocas, desde cero, con las mismas semillas7,17,27, mismo split de instrumentos, mismo Adam, batch y arquitectura. Se prueban VAE denso y convolucional; los AE de denoising no tienen KL y no participan en esta ablación.

Se reutilizan los seis controles actuales de `configs/nsynth_cnn/vae_{mlp,conv}.json`. Se agregan solo seis corridas nuevas, en `configs/nsynth_warmup/`. Para cada arquitectura y semilla se verificó que la inicialización de pesos y los estados RNG de muestreo, corrupción y orden de lotes fueran idénticos. La única diferencia de hiperparámetros es `model.kl.warmup_epochs`,0o50. No es una continuación de las ramas beta canceladas.

Se comparan reconstrucción en instrumentos reservados, KL posterior, dimensiones latentes activas y generación con códigos normales estándar pareados. No se eligen ejemplos según el resultado. La pérdida total antes de la época50 tiene distinto peso KL entre condiciones y no es una comparación directa de un objetivo fijo. Reconstrucción y KL se interpretan separadamente.

Dos trabajadores adicionales con prioridad reducida (`nice10`) preservan prioridad del estudio principal. No se duplican los controles.

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5
bash scripts/run_nsynth_warmup_study.sh 2
```

El lock evita duplicados; checkpoints completos cada cinco épocas permiten reanudar después de una interrupción. El log es `results/nsynth_warmup_study/run.log`. Informe: `reports/nsynth_warmup/informe.md`.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.analysis.nsynth_warmup_report
```

Los estados de inicialización auditados están en `results/nsynth_warmup_study/protocol.json`. Un resultado favorable en reconstrucción puede acompañarse de generación peor, y viceversa. El análisis no presupone que el calentamiento mejore el resultado. Las distancias a vecinos y las dimensiones activas tampoco prueban calidad auditiva o novedad.
