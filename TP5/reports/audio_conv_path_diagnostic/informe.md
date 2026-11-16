# ¿Interpolar antes o después del decoder?

Modelos congelados del control completo: las notas objetivo estuvieron en entrenamiento.
El objetivo solo mide las salidas. Para generarlas entran los vecinos MIDI ±1.
La mezcla latente decodifica el promedio de medias posteriores; la mezcla reconstruida promedia las dos salidas del decoder.
La mezcla mel promedia directamente los espectrogramas de entrada.
La altura se mide únicamente en el punto medio predefinido. Los puntos 0,25 y 0,75 son diagnóstico de distancia al objetivo central, no se esperan idénticos al objetivo ni se elige el mejor.
Inválidos cuentan como fallos. Las variantes de envolvente no son entrenamientos independientes.

| Modelo | Método central | Altura | Espectro | Ambos | Inválidos | MSE |
|---|---|---:|---:|---:|---:|---:|
| conv | decoded_neighbor_mix | 27/27 | 0/27 | 0/27 | 0 | 0.017409 |
| conv | latent_mix | 15/27 | 11/27 | 11/27 | 0 | 0.015513 |
| conv | spectrogram_mix | 27/27 | 0/27 | 0/27 | 0 | 0.017217 |
| mlp | decoded_neighbor_mix | 27/27 | 0/27 | 0/27 | 0 | 0.017664 |
| mlp | latent_mix | 16/27 | 8/27 | 8/27 | 0 | 0.015273 |
| mlp | spectrogram_mix | 27/27 | 0/27 | 0/27 | 0 | 0.017217 |

La comparación mide la no linealidad del decoder con pesos fijos. Una mejora de altura aislada puede reflejar dos notas mezcladas.
No prueba interpolación de notas ausentes, no certifica calidad auditiva, y no modifica la comparación principal de 12 corridas.

Verificación: las 54 predicciones latentes centrales conservan aciertos de altura, validez y aciertos espectrales del CSV original.
Diferencia máxima MSE: 9.47e-09. Diferencia máxima de frecuencia: 4.66e-05 Hz.
El códec devuelve exactamente el mismo audio al repetir una entrada. Las seis calibraciones previas son fiables: ver audit.json para tasas por variante.
