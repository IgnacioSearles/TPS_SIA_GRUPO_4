# Estudio por cantidad de épocas

Cada fila se obtiene de los historiales por época de las corridas reproducibles guardadas en este directorio. Las métricas de validación corresponden a cinco folds; el lineal se muestra sobre todas las muestras para observar su capacidad de ajuste.

| Épocas | Lineal RMSE train | Sigmoide RMSE train (folds) | Sigmoide RMSE validación | Desvío entre folds | MAE validación | Brecha train-validación |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.161635 | 0.176252 | 0.176325 | 0.003370 | 0.145556 | 0.000074 |
| 25 | 0.161668 | 0.134917 | 0.135023 | 0.002252 | 0.109176 | 0.000106 |
| 50 | 0.161666 | 0.116626 | 0.116743 | 0.001415 | 0.091694 | 0.000117 |
| 100 | 0.161698 | 0.107888 | 0.108010 | 0.001157 | 0.081886 | 0.000122 |
| 200 | 0.161663 | 0.104963 | 0.105086 | 0.001175 | 0.077689 | 0.000123 |
| 300 | 0.161682 | 0.104601 | 0.104723 | 0.001207 | 0.076853 | 0.000122 |
| 600 | 0.161773 | 0.104535 | 0.104655 | 0.001236 | 0.076496 | 0.000120 |
| 1000 | 0.161666 | 0.104535 | 0.104655 | 0.001234 | 0.076472 | 0.000120 |

El menor RMSE de validación entre los cortes evaluados ocurre a las 1000 épocas (0.104655).
De 100 épocas al máximo evaluado, el RMSE de validación baja 0.003355. De 300 a 600 baja 0.00006831; de 600 a 1000 solo baja 0.00000026. 600 épocas ya está prácticamente en la meseta.
El RMSE de validación no sube al aumentar las épocas y la brecha train-validación permanece pequeña (cerca de 0.0002). No se observa overfitting en los cortes medidos.
El lineal conserva RMSE de entrenamiento alrededor de 0.162 desde el primer corte y apenas mejora con más épocas, compatible con underfitting por capacidad. La sigmoide reduce el RMSE con rapidez y se aproxima a una meseta.

Esta comparación no prueba el comportamiento después del máximo medido ni frente a una distribución futura distinta.
