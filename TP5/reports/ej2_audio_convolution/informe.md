# Audio: VAE denso frente a convolucional

Estado: **incomplete**, 3/12 corridas terminadas.

Cada denominador incluye inválidos como fallos. El punto central se fijó antes de observar resultados.
No se usa la trayectoria para elegir la mejor fracción. Las variantes de una nota no son entrenamientos independientes.

| Modelo | Condición | Método | Altura | Espectro más próximo al objetivo | Ambos | MSE espectral |
|---|---|---|---:|---:|---:|---:|
| conv | complete | vae_interpolation | 15/27 | 11/27 | 11/27 | 0.015513 |
| conv | complete | spectrogram_mix | 27/27 | 0/27 | 0/27 | 0.017217 |
| conv | complete | lower_neighbor | 0/27 | 0/27 | 0/27 | 0.031180 |
| conv | complete | target_reconstruction | 27/27 | 27/27 | 27/27 | 0.000268 |

Las reconstrucciones directas de objetivos solo aparecen en los controles completos; no intervienen en la interpolación.
La MSE de entrenamiento no es una métrica de generalización. La cercanía espectral no certifica calidad auditiva ni novedad.
Las notas finales 55, 61 y 67 no se entrenan ni se evalúan. Este informe corresponde únicamente a validación.
Si faltan corridas o la calibración del códec falla, no se declara evidencia favorable ni un ganador.
Un resultado musical negativo es un resultado válido. No se inicia una búsqueda de hiperparámetros automáticamente.
