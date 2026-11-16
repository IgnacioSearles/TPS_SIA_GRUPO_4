# Informe de warmup: tanda original y ampliación

`src.analysis.nsynth_warmup_report.report(root)` evalúa el plan completo de24VAE: dos arquitecturas, warmup0/50 y seis semillas7,17,27,37,47,57. Todos usan200épocas y beta final0.5. Los12modelos de las semillas7,17,27 forman la tanda original. Los12modelos de37,47,57 amplían la comparación; no reemplazan ni reentrenan los originales.

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.analysis.nsynth_warmup_report
```

El reporte queda incompleto hasta disponer de24modelos terminados y evaluados. Enumera los pendientes y conserva los denominadores. Un agregado de las tres primeras semillas nunca se presenta como resultado de seis. La ampliación mantiene el mismo conjunto de instrumentos de prueba: estudia estabilidad frente a la inicialización, pero no constituye un test externo nuevo.

## Archivos

- `reports/nsynth_warmup/per_seed.csv`: todas las condiciones y semillas terminadas.
- `comparison.csv`: agregado de los modelos actualmente disponibles, con número de semillas por métrica.
- `comparison_original_three.csv`: agregado separado de7,17,27.
- `comparison_extension_three.csv`: agregado separado de37,47,57, cuando existan resultados.
- `paired_differences.csv`: diferencias warmup0 menos warmup50, por arquitectura y semilla.
- `paired_descriptions.csv`: media, mediana, rango y número de diferencias negativas/positivas/cero, separados por tanda original, ampliación y plan de seis semillas. Cada fila informa pares disponibles y esperados.
- `latent_dimensions.csv`: varianza de medias posteriores y KL por dimensión, con umbral activo estricto de0.01.
- `summary.json` e `informe.md`: estado, métricas y límites del análisis.

No se calculan valores p ni se declara significación estadística. Con seis semillas sigue habiendo una muestra pequeña de entrenamientos. En MSE held-out una diferencia negativa favorece warmup0 para esa semilla, pero las diferencias de KL, dimensiones activas o diversidad no se etiquetan automáticamente como mejores o peores.

## Compatibilidad y recursos

El reporte exige configuraciones pareadas idénticas salvo warmup,200épocas registradas, mismos códigos generativos y mismas referencias. Reutiliza las estadísticas latentes cacheadas por pesos, manifiesto y selección; no vuelve a entrenar, decodificar WAV ni ejecutar Griffin-Lim/PGHI. Las métricas aquí analizadas son de espectrogramas mel y posterior latente y no dependen de la inicialización de fase al exportar audio. La escucha debe usar una fase homogénea y registrar su método aparte. Los WAV históricos no se reinterpretan como PGHI.

Los gráficos muestran todas las semillas disponibles. Las curvas separan reconstrucción y KL, y solo comparan pérdida total desde la época50, cuando ambas condiciones ya usan beta0.5. Las variantes y las224notas de prueba de un modelo no cuentan como224entrenamientos independientes.
