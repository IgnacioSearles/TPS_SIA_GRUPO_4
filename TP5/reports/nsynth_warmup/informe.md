# NSynth: ablación de warmup KL
Estado: incomplete, 12/24 modelos entrenados y 12/24 evaluados.

Dos arquitecturas, warmup 0 o 50, 200 épocas desde cero y beta final 0,5. La tanda original usa semillas 7, 17, 27 y 12 modelos; su ampliación agrega semillas 37, 47, 57 y 12 modelos. Los seis controles warmup 50 originales se reutilizan del estudio NSynth principal. El plan completo tiene seis semillas por condición.

Tanda original: 12/12 entrenados. Ampliación: 0/12 entrenados. Mientras falten semillas nuevas, el agregado disponible sigue reflejando las semillas terminadas; no se presenta como un resultado de seis semillas.

| Arquitectura | Warmup | Semilla | MSE test | KL final train | Recon final train | Loss final train | KL posterior media | Dims activas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp | 0 | 7 | 0.015494 | 29.325009 | 18.271324 | 32.933828 | 29.072281 | 16 |
| mlp | 0 | 17 | 0.014626 | 29.044504 | 17.794257 | 32.316509 | 28.896748 | 16 |
| mlp | 0 | 27 | 0.015661 | 28.818381 | 18.869888 | 33.279078 | 29.134586 | 16 |
| mlp | 50 | 7 | 0.015828 | 29.154017 | 19.848297 | 34.425306 | 30.111655 | 16 |
| mlp | 50 | 17 | 0.014750 | 30.045884 | 18.063629 | 33.086570 | 29.088094 | 16 |
| mlp | 50 | 27 | 0.016515 | 29.261203 | 18.458631 | 33.089232 | 29.346084 | 16 |
| conv | 0 | 7 | 0.014776 | 25.398326 | 14.919059 | 27.618222 | 24.555498 | 16 |
| conv | 0 | 17 | 0.014611 | 26.346813 | 18.359512 | 31.532919 | 26.535944 | 16 |
| conv | 0 | 27 | 0.014915 | 25.446229 | 14.364067 | 27.087181 | 24.690656 | 16 |
| conv | 50 | 7 | 0.014692 | 26.123477 | 15.040895 | 28.102633 | 25.812640 | 16 |
| conv | 50 | 17 | 0.014801 | 27.078634 | 17.801237 | 31.340553 | 27.130468 | 16 |
| conv | 50 | 27 | 0.014991 | 25.672064 | 14.448698 | 27.284730 | 25.657456 | 16 |

## Diferencias pareadas de MSE held-out

Δ = MSE con warmup 0 menos MSE con warmup 50. Un valor negativo favorece warmup 0 en esa semilla. Se muestran por separado las semillas originales, las nuevas y todas las disponibles. No se calcula significación estadística.

| Grupo | Arquitectura | Pares disponibles/esperados | Δ media | Δ mínima | Δ máxima | Δ negativas | Δ positivas |
|---|---|---:|---:|---:|---:|---:|---:|
| Original: 7, 17, 27 | mlp | 3/3 | -0.000437 | -0.000854 | -0.000124 | 3 | 0 |
| Original: 7, 17, 27 | conv | 3/3 | -0.000061 | -0.000190 | 0.000083 | 2 | 1 |
| Plan completo: seis semillas | mlp | 3/6 | -0.000437 | -0.000854 | -0.000124 | 3 | 0 |
| Plan completo: seis semillas | conv | 3/6 | -0.000061 | -0.000190 | 0.000083 | 2 | 1 |

## Modelos pendientes

- vae_mlp_no_warmup, semilla 37: entrenamiento pendiente.
- vae_mlp_no_warmup, semilla 47: entrenamiento pendiente.
- vae_mlp_no_warmup, semilla 57: entrenamiento pendiente.
- vae_mlp, semilla 37: entrenamiento pendiente.
- vae_mlp, semilla 47: entrenamiento pendiente.
- vae_mlp, semilla 57: entrenamiento pendiente.
- vae_conv_no_warmup, semilla 37: entrenamiento pendiente.
- vae_conv_no_warmup, semilla 47: entrenamiento pendiente.
- vae_conv_no_warmup, semilla 57: entrenamiento pendiente.
- vae_conv, semilla 37: entrenamiento pendiente.
- vae_conv, semilla 47: entrenamiento pendiente.
- vae_conv, semilla 57: entrenamiento pendiente.

## Generación libre, mismos códigos normales estándar

| Arquitectura | Warmup | Semilla | MSE vecino train | MSE vecino test | MSE entre pares | Vecinos train distintos /64 |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 0 | 7 | 0.015033 | 0.021251 | 0.035603 | 58 |
| mlp | 0 | 17 | 0.015381 | 0.021177 | 0.033328 | 61 |
| mlp | 0 | 27 | 0.017100 | 0.022668 | 0.041629 | 61 |
| mlp | 50 | 7 | 0.015890 | 0.022274 | 0.035999 | 59 |
| mlp | 50 | 17 | 0.015989 | 0.021271 | 0.034268 | 60 |
| mlp | 50 | 27 | 0.016528 | 0.023967 | 0.040837 | 62 |
| conv | 0 | 7 | 0.014236 | 0.021060 | 0.037060 | 60 |
| conv | 0 | 17 | 0.014101 | 0.020185 | 0.033404 | 62 |
| conv | 0 | 27 | 0.012590 | 0.018938 | 0.033472 | 62 |
| conv | 50 | 7 | 0.014131 | 0.021289 | 0.037837 | 62 |
| conv | 50 | 17 | 0.015666 | 0.021633 | 0.033664 | 61 |
| conv | 50 | 27 | 0.013839 | 0.020204 | 0.034454 | 60 |

Cada par de la misma arquitectura y semilla debe compartir configuración completa salvo warmup. La misma semilla fija inicialización, orden de minibatches y secuencia de ruido de reparametrización; después del primer gradiente diferente los pesos pueden divergir. Los datos se separan por instrumento y las métricas test no intervienen en entrenamiento.
Dimensión activa: varianza poblacional de μ_j(x) entre notas de entrenamiento estrictamente mayor que0.01. KL posterior media evalúa analíticamente q(z|x) contra N(0,I) con pesos finales, sin muestrear ni decodificar. Estos indicadores pueden revelar uso débil del latente, pero no establecen por sí solos un colapso posterior ni una relación musical correcta.
Los términos finales de entrenamiento proceden de la última época registrada y usan reconstrucción sumada sobre características por muestra. La MSE held-out promedia características. Tienen escalas y procedimientos distintos y no deben compararse como si fueran la misma métrica.
Durante las primeras50 épocas, beta es0.5 desde el inicio sin warmup y crece como0.5×min(1,época/50) con warmup. Por eso no se comparan las pérdidas totales de esas primeras épocas como si usaran el mismo objetivo ponderado. Las curvas muestran reconstrucción y KL por separado, y pérdida total únicamente desde la época50. El total final corresponde al mismo beta0.5.
La generación reutiliza la evaluación cacheada: no repite WAV ni Griffin-Lim. Menor distancia al vecino y mayor diversidad no certifican calidad ni novedad. Train y test contienen distinto número de referencias, por lo que sus distancias mínimas no prueban generalización.
Las métricas de este informe operan sobre espectrogramas mel y distribuciones latentes; no dependen del método de fase usado para exportar WAV. Una comparación de escucha debe usar el mismo método de fase en todos los modelos y registrarlo por separado. Los WAV históricos no se reinterpretan como PGHI ni se regeneran en este reporte.
Los CSV conservan todas las semillas y las diferencias warmup0 menos warmup50. Un resultado peor o inconcluso sigue siendo válido. Mientras falten pares no se declara ganador. El plan llega a seis entrenamientos independientes por condición; las224 notas de prueba de cada modelo no son224 entrenamientos independientes. La ampliación usa el mismo conjunto de prueba que la tanda original: evalúa estabilidad frente a inicialización, pero no equivale a una validación externa nueva ni garantiza generalización a otros instrumentos o datasets.
