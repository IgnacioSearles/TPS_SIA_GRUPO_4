# Reconstrucción VAE NSynth: nueve semillas
Estado: incomplete, 8/18 modelos y 3/9 pares evaluados.

Se reutilizan las semillas 7,17,27 y las réplicas warmup50 de37,47,57. Solo se entrenan seis modelos nuevos: 67,77,87 para denso y CNN. Todos usan 200 épocas, beta0.5, warmup50 y el mismo split de884/224 por instrumento.

| Semilla | MSE denso | MSE CNN | Δ CNN menos denso |
|---|---:|---:|---:|
| 7 | 0.015828 | 0.014692 | -0.001135 |
| 17 | 0.014750 | 0.014801 | 0.000051 |
| 27 | 0.016515 | 0.014991 | -0.001524 |

Entre los 3 pares disponibles, Δ media=-0.000870; CNN con menor MSE en 2/3 semillas. Estos recuentos se actualizan solo cuando ambas arquitecturas de una semilla están evaluadas.

## Pendientes

- conv, semilla 37: training_pending, época 105/200.
- conv, semilla 47: training_pending, época 25/200.
- mlp, semilla 57: training_pending, época 75/200.
- conv, semilla 57: queued, época 0/200.
- mlp, semilla 67: training_pending, época 5/200.
- conv, semilla 67: training_pending, época 0/200.
- mlp, semilla 77: training_pending, época 5/200.
- conv, semilla 77: queued, época 0/200.
- mlp, semilla 87: queued, época 0/200.
- conv, semilla 87: queued, época 0/200.

La MSE global pondera las224notas por igual; el CSV por familia conserva los desbalances. Hay nueve inicializaciones independientes por arquitectura cuando termine el plan, no9×224 entrenamientos. La ampliación reusa el mismo conjunto de instrumentos reservados y evalúa estabilidad frente a semillas, no generalización externa.
No se declara significación estadística ni se elige ganador con pares faltantes. La evaluación trabaja en mel: no cambia con la inicialización de fase al escuchar WAV. Una mejor MSE no certifica calidad auditiva ni novedad. Se conservan fallos e inválidos. La pérdida VAE y el ejecutor de entrenamiento no se modificaron; no se reentrenan réplicas warmup50 existentes.
