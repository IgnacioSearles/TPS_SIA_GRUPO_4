# Ejercicio 1: estimación de probabilidad de fraude

Objetivo continuo: `big_model_fraud_probability`. Entrada: 6 variables. Pérdida: MSE / 2. No se calculan decisiones binarias.

## Comparación de aprendizaje

Ambos perceptrones simples se entrenaron con las mismas muestras, escala, semilla e hiperparámetros. Estos errores son de entrenamiento y describen capacidad de ajuste; no estiman generalización.

| Activación | Épocas | MAE | RMSE | Salidas fuera de [0,1] | Mejora de pérdida en últimas 20 épocas |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 1000 | 0.130264 | 0.161666 | 428 | 0.000027 |
| sigmoid | 1000 | 0.076458 | 0.104665 | 0 | -0.000000 |

El sigmoide reduce el RMSE de entrenamiento un 35.3% respecto del lineal. El mayor error residual del lineal, incluso entrenado con todos los datos, indica underfitting relativo frente al sigmoide.
Las mejoras de pérdida en las últimas 20 épocas son pequeñas: ambas curvas muestran una meseta aproximada con estos hiperparámetros. Esto no demuestra un mínimo global.
Se selecciona el sigmoide para generalización: además del menor error, su salida siempre queda en [0,1].

## Generalización: K-Fold

Se usaron 5 folds aleatorios reproducibles. Cada fold ajustó su propio escalador solo con entrenamiento, inició un modelo nuevo y evaluó las probabilidades de validación.
MAE de validación: 0.076585 ± 0.000831.
RMSE de validación: 0.104847 ± 0.001065.
RMSE promedio de train: 0.104644; la brecha train-validación es 0.000203.
Métricas con todas las predicciones fuera de muestra reunidas: MAE 0.076585, RMSE 0.104852.
Baseline constante (media de probabilidades de cada train): RMSE fuera de muestra 0.302540; el modelo reduce ese error un 65.3%.
La brecha pequeña y estable junto con la mejora casi nula al pasar de 600 a 1.000 épocas no muestra señales de overfitting en este rango.

## Modelo final

Se reentrenó una sigmoide con todas las 7500 muestras. Tiene 7 parámetros entrenables. Sus pesos están en `model_weights.npz` y el orden de variables y los parámetros de estandarización en `model_metadata.json`.
Terminó después de 1000 épocas (máximo de épocas).
Para una transacción nueva, aplicar ese mismo escalador y luego la red. El resultado es una probabilidad en [0,1].
