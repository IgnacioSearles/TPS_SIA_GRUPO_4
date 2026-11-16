# NSynth: arquitectura densa frente a CNN
Estado: complete, 12/12 entrenamientos evaluados.

## Reconstrucción y denoising en instrumentos reservados

| Condición | Tarea | Método | Ruido mel σ | Casos | MSE | Inválidos |
|---|---|---|---:|---:|---:|---:|
| dae_conv | denoising | input | 0.0 | 672 | 0.000000 | 0 |
| dae_conv | denoising | input | 0.05 | 672 | 0.001927 | 0 |
| dae_conv | denoising | input | 0.1 | 672 | 0.007413 | 0 |
| dae_conv | denoising | input | 0.2 | 672 | 0.027086 | 0 |
| dae_conv | denoising | model | 0.0 | 672 | 0.015674 | 0 |
| dae_conv | denoising | model | 0.05 | 672 | 0.015460 | 0 |
| dae_conv | denoising | model | 0.1 | 672 | 0.015362 | 0 |
| dae_conv | denoising | model | 0.2 | 672 | 0.016126 | 0 |
| dae_mlp | denoising | input | 0.0 | 672 | 0.000000 | 0 |
| dae_mlp | denoising | input | 0.05 | 672 | 0.001927 | 0 |
| dae_mlp | denoising | input | 0.1 | 672 | 0.007413 | 0 |
| dae_mlp | denoising | input | 0.2 | 672 | 0.027086 | 0 |
| dae_mlp | denoising | model | 0.0 | 672 | 0.016263 | 0 |
| dae_mlp | denoising | model | 0.05 | 672 | 0.016156 | 0 |
| dae_mlp | denoising | model | 0.1 | 672 | 0.016585 | 0 |
| dae_mlp | denoising | model | 0.2 | 672 | 0.020476 | 0 |
| vae_conv | reconstruction | model | 0.0 | 672 | 0.014828 | 0 |
| vae_mlp | reconstruction | model | 0.0 | 672 | 0.015698 | 0 |

## Generación libre del VAE

| Condición | Semilla | MSE al vecino train | MSE al vecino test | Diversidad MSE entre pares | Vecinos train distintos /64 |
|---|---:|---:|---:|---:|---:|
| vae_mlp | 7 | 0.015890 | 0.022274 | 0.035999 | 59 |
| vae_mlp | 17 | 0.015989 | 0.021271 | 0.034268 | 60 |
| vae_mlp | 27 | 0.016528 | 0.023967 | 0.040837 | 62 |
| vae_conv | 7 | 0.014131 | 0.021289 | 0.037837 | 62 |
| vae_conv | 17 | 0.015666 | 0.021633 | 0.033664 | 61 |
| vae_conv | 27 | 0.013839 | 0.020204 | 0.034454 | 60 |

La separación de instrumentos evita que notas del mismo instrumento estén a la vez en train y test. Las métricas se exportan por nota, familia y semilla. Las notas de un instrumento no son entrenamientos independientes.
El denoising usa ruido gaussiano σ0.1 en espectrogramas mel durante entrenamiento. Evaluamos σ0,0.05,0.1,0.2 con entradas idénticas entre arquitecturas. Esto no equivale a entrenar o evaluar eliminación de ruido acústico de un micrófono.
Generamos64 códigos normales estándar por semilla, idénticos entre VAE. Los primeros6 fijados se exportan a WAV. Las distancias a entrenamiento y prueba, su diversidad y los vecinos distintos describen las muestras, pero no certifican calidad, novedad ni ausencia de memorización. Los conjuntos train y test tienen tamaños distintos: las distancias mínimas no son directamente comparables como evidencia de generalización. La MSE global pondera cada nota por igual, no cada familia. El CSV por familia permite inspeccionar ese desequilibrio.
Los espectrogramas comparativos usan semilla7 y ejemplos prefijados por familia, sin elegir los mejores. El códec mel introduce degradación incluso sin red. Los WAV de referencia deben distinguir original y referencia pasada por el códec.
No se modificó ninguna fórmula de pérdida: VAE MSE+betaKL, AE MSE. No se evalúan aquí notas omitidas.
