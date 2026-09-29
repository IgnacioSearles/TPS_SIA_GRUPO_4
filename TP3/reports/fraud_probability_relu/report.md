# Ejercicio 1: estimación de probabilidad de fraude

Objetivo continuo: `big_model_fraud_probability`. Entrada: 6 variables. Pérdida: MSE / 2. No se calculan decisiones binarias.

## Comparación de aprendizaje

Todos los perceptrones simples se entrenaron con las mismas muestras, escala, semilla e hiperparámetros. Estos errores son de entrenamiento y describen capacidad de ajuste; no estiman generalización.

| Activación | Épocas | MAE | RMSE | Salidas fuera de [0,1] | Mejora de pérdida en últimas 20 épocas |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 1000 | 0.130264 | 0.161666 | 428 | 0.000027 |
| sigmoid | 1000 | 0.076458 | 0.104665 | 0 | -0.000000 |
| relu | 1000 | 0.128023 | 0.160724 | 333 | 0.000027 |

`relu` cambia el RMSE de entrenamiento un -0.6% respecto del lineal (`identity`).
Mejoras de pérdida pequeñas en las últimas 20 épocas indican una meseta con estos hiperparámetros; no demuestran un mínimo global.
Activación usada para el estudio de generalización: `relu`.

## Partición de los datos

Se separó un 20% de las filas como test (1500 filas), estratificado por deciles de la probabilidad objetivo. Quedan 6000 filas de desarrollo: todo entrenamiento, K-Fold, ajuste de hiperparámetros y elección de umbral usa solo esas filas. El test se evalúa una única vez con el modelo final.

## Generalización: K-Fold sobre desarrollo

Se usaron 5 folds estratificados y reproducibles. Cada fold ajustó su propio escalador solo con entrenamiento, inició un modelo nuevo y evaluó las probabilidades de validación.
MAE de validación: 0.126852 ± 0.002264.
RMSE de validación: 0.159332 ± 0.002448.
RMSE promedio de train: 0.159031; la brecha train-validación es 0.000301.
Métricas con todas las predicciones fuera de muestra reunidas: MAE 0.126852, RMSE 0.159347.
Baseline constante (media de probabilidades de cada train): RMSE fuera de muestra 0.302558; el modelo reduce ese error un 47.3%.
La brecha pequeña y estable junto con la mejora casi nula al pasar de 600 a 1.000 épocas no muestra señales de overfitting en este rango.

## Modelo final

Se reentrenó el perceptrón `relu` con las 6000 filas de desarrollo. Tiene 7 parámetros entrenables. Sus pesos están en `model_weights.npz` y el orden de variables y los parámetros de estandarización en `model_metadata.json`.
Terminó después de 1000 épocas (máximo de épocas).
Test (nunca usado antes): MAE 0.130269, RMSE 0.169388; baseline constante RMSE 0.302032. Predicciones fuera de [0,1]: 74 de 1500.
Para una transacción nueva, aplicar ese mismo escalador y luego la red.
El análisis de validación y la recomendación de umbral se generan con `python -m experiments.fraud.validation_analysis`.
