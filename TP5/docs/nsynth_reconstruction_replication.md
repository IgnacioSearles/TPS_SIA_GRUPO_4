# Réplicas adicionales de reconstrucción VAE en NSynth

Esta tanda responde a la comparación de reconstrucción entre VAE denso y CNN. Mantiene datos, arquitectura por condición, pérdida MSE+betaKL, beta0.5, warmup50, latente16, capas512/128, precisiónfloat32, Adam0.001, batch64 y200épocas. Solo agrega semillas.

Se reutilizan las semillas7,17,27 y las réplicas warmup50 de37,47,57 que ya estaban terminadas, ejecutándose o en cola. No se duplican. Las únicas seis corridas nuevas corresponden a67,77,87 para denso y CNN. La auditoría compara configuraciones y firmas de checkpoints contra el protocolo original y se guarda en `results/nsynth_reconstruction_replication/protocol.json`; el avance observado se actualiza en `latest_reused_runs.json`.

## Ejecución y reanudación

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5
bash scripts/run_nsynth_reconstruction_replication.sh 3
```

Usa tres procesos y un hilo BLAS por proceso, lock contra duplicados y checkpoints completos cada cinco épocas. Después de cada corrida actualiza la comparación. Cuando terminan las seis nuevas, espera a las réplicas warmup50 existentes que falten y refresca el informe sin reentrenarlas. Si el ejecutor de esas réplicas fue interrumpido, hay que reanudarlo por separado con su comando original. Este ejecutor nunca ocupa sus carpetas ni programa esas semillas.

El log está en `results/nsynth_reconstruction_replication/run.log` y el PID de lanzamiento en `launcher.json`. No sobrescribe el `summary.csv` compartido del estudio original ni modifica los informes de tres semillas o de la ablación de warmup.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.analysis.nsynth_reconstruction_replication --report-only
```

## Evaluación

La comparación tiene18modelos esperados, nueve semillas por arquitectura y224notas de instrumentos reservados por modelo. Reutiliza el CSV de reconstrucción anterior cuando pesos, etiquetas y manifiesto coinciden. Si no existe ese CSV, calcula solo reconstrucción por lotes: no exporta WAV, no ejecuta PGHI y no realiza generación libre. Las métricas mel no dependen de la recuperación de fase del audio.

`reports/nsynth_reconstruction_replication/informe.md` conserva explícitamente los modelos y pares pendientes. `paired_seeds.csv` contiene únicamente pares completos, y define delta como MSE de CNN menos MSE del denso. Se exportan `per_note.csv`, `per_family.csv`, `runs.csv`, `summary.json` y un gráfico por semilla. Un caso inválido se conserva como fallo; no se elimina para reducir el error medio.

Solo habrá nueve entrenamientos independientes por arquitectura al terminar el plan, no9×224entrenamientos. El conjunto de prueba es el mismo del estudio anterior: estas réplicas miden estabilidad ante semillas y no equivalen a una validación externa. Una MSE menor no prueba calidad auditiva ni novedad. No se declara significación estadística ni un ganador mientras falten pares.
