# Control de generación del AE con distribución ajustada

Se reutilizaron los modelos de las semillas 7,17,27, sin reentrenar. Para cada
AE se ajustó una gaussiana de covarianza completa a sus 1.024 códigos de
entrenamiento. Las referencias de prueba no intervienen en el ajuste. Se
generaron 64 muestras por método y semilla (192 por método), usando una misma
matriz normal base de semilla 1234.

| Método | Diversidad MSE, media ± desvío | Distancia MSE al entrenamiento, media ± desvío |
|---|---:|---:|
| AE, normal estándar | 0,10436 ± 0,00771 | 0,03670 ± 0,01006 |
| AE, gaussiana ajustada | 0,10420 ± 0,00408 | 0,00767 ± 0,00089 |
| VAE β=1, normal estándar | 0,10178 ± 0,00843 | 0,00822 ± 0,00098 |

Los desvíos son entre las tres semillas, no entre 192 observaciones independientes.
La distancia baja aproximadamente un 79 % para el AE ajustado respecto de la
normal estándar. La diversidad media permanece prácticamente igual. Ninguna de
estas métricas certifica por sí sola calidad, novedad o ausencia de memorización.

## Inspección e interpretación

Las comparaciones compactas usan los índices prefijados 0,8,...,56, iguales para
todos los métodos. Además se revisaron los tres mosaicos completos del AE
ajustado: 192 imágenes. Se observan muchas elipses y rectángulos reconocibles y
una reducción clara del ruido de fondo respecto del AE con normal estándar.
Persisten casos pequeños, deformados, con bordes irregulares o ruido, por
ejemplo la muestra 1 de la semilla 7. No se hizo una clasificación cuantitativa
de validez ni una evaluación humana ciega.

Este control corrige la interpretación de la comparación inicial: los artefactos
del AE desde N(0,I) no demuestran incapacidad de generación. El AE produce
figuras plausibles cuando se adapta la distribución de muestreo a sus códigos.
El VAE permite usar un prior definido antes de entrenar gracias a la
regularización KL; en este experimento, eso no demuestra una superioridad
universal de su calidad generativa frente al AE con densidad ajustada.

Tampoco puede afirmarse que el AE ajustado sea superior al VAE por su menor
distancia media al entrenamiento. Esa cercanía puede reflejar plausibilidad,
menor novedad o ambos. Los resultados de reconstrucción no cambian: los pesos
son exactamente los mismos. Una gaussiana única aún puede atravesar regiones
del latente con pocos códigos.

## Reproducir

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.latent_sampling
```

Salidas: `results/generative_latent_sampling_report/`. Incluyen CSV por muestra
y corrida, resumen, mosaicos completos, comparaciones con vecinos de
entrenamiento, coordenadas, media/covarianza y hashes de los seis modelos fuente.
La validación previa exige las 15 corridas originales de figuras terminadas y
compatibles. Las generaciones de AE/VAE desde la normal estándar se contrastan
contra las ya guardadas; las evaluaciones anteriores y los modelos no se
sobrescriben.

No se realizó escucha subjetiva del audio en este control. Su afinación y los
diagnósticos espectrales previos no reemplazan esa revisión.
