# Estudio por cantidad de épocas

Cada fila se obtiene de los historiales por época de las corridas reproducibles guardadas en este directorio. Las métricas de validación corresponden a cinco folds; el lineal se muestra sobre todas las muestras para observar su capacidad de ajuste.

| Épocas | Lineal RMSE train | Sigmoide RMSE train (folds) | Sigmoide RMSE validación | Desvío entre folds | MAE validación | Brecha train-validación |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.161635 | 0.166977 | 0.167035 | 0.001630 | 0.137807 | 0.000058 |
| 25 | 0.161668 | 0.129578 | 0.129684 | 0.000936 | 0.104399 | 0.000105 |
| 50 | 0.161666 | 0.114247 | 0.114403 | 0.000826 | 0.089216 | 0.000156 |
| 100 | 0.161698 | 0.107058 | 0.107251 | 0.000810 | 0.080856 | 0.000194 |
| 200 | 0.161663 | 0.104890 | 0.105097 | 0.000939 | 0.077470 | 0.000207 |
| 300 | 0.161682 | 0.104675 | 0.104881 | 0.001020 | 0.076857 | 0.000206 |
| 600 | 0.161773 | 0.104644 | 0.104848 | 0.001063 | 0.076597 | 0.000204 |
| 1000 | 0.161666 | 0.104644 | 0.104847 | 0.001065 | 0.076585 | 0.000203 |

El menor RMSE de validación entre los cortes evaluados ocurre a las 1000 épocas (0.104847).
De 100 épocas al máximo evaluado, el RMSE de validación baja 0.002404. De 300 a 600 baja 0.00003262; de 600 a 1000 solo baja 0.00000075. 600 épocas ya está prácticamente en la meseta.
El RMSE de validación no sube al aumentar las épocas y la brecha train-validación permanece pequeña (cerca de 0.0002). No se observa overfitting en los cortes medidos.
El lineal conserva RMSE de entrenamiento alrededor de 0.162 desde el primer corte y apenas mejora con más épocas, compatible con underfitting por capacidad. La sigmoide reduce el RMSE con rapidez y se aproxima a una meseta.

Esta comparación no prueba el comportamiento después del máximo medido ni frente a una distribución futura distinta.
