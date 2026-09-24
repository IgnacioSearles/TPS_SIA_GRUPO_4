# Estudios de hiperparámetros: learning rate e inicialización

Se usó K-Fold de 5 particiones, sigmoide de salida, MSE/2 y 300 épocas por corrida. En cada fold, el escalador y la inicialización guiada se ajustan usando exclusivamente las filas de entrenamiento.

## Learning rate

| Learning rate | MAE validación medio | RMSE validación medio ± desvío | RMSE train medio |
| ---: | ---: | ---: | ---: |
| 0.001 | 0.165894 | 0.195004 ± 0.015964 | 0.195257 |
| 0.005 | 0.100020 | 0.125054 ± 0.005380 | 0.125074 |
| 0.01 | 0.086759 | 0.112253 ± 0.001319 | 0.112138 |
| 0.025 | 0.078467 | 0.105612 ± 0.000793 | 0.105419 |
| 0.05 | 0.076863 | 0.104881 ± 0.001017 | 0.104676 |
| 0.1 | 0.076598 | 0.104848 ± 0.001062 | 0.104644 |
| 0.2 | 0.076587 | 0.104850 ± 0.001064 | 0.104644 |

El mejor valor explorado fue lr=0.1, con RMSE de validación medio 0.104848. La curva `learning_rate_curves.png` permite comparar velocidad de convergencia y estabilidad entre tasas.
En esta corrida, lr=0.001 quedó en RMSE 0.195004; en el valor cercano a 0.05 (0.05) bajó a 0.104881. El mejor explorado fue 0.1; frente al mayor valor probado (0.2), el error fue 0.104848 vs. 0.104850. Valores cercanos a la meseta son prácticamente equivalentes; revisá las curvas para detectar inestabilidad en los puntos evaluados.

## Inicialización guiada

La inicialización guiada ajusta una regresión lineal de `logit(probabilidad BigModel)` sobre las features estandarizadas del train del fold. Sus coeficientes e intercepto inicializan la capa sigmoide; las probabilidades objetivo 0/1 se recortan solo para poder calcular el logit. Se agrega el ruido pequeño configurado para empezar cerca de ese ajuste. No se usa la validación para construir los pesos.

| Inicialización | RMSE validación inicial medio | RMSE validación final medio | MAE validación final medio |
| --- | ---: | ---: | ---: |
| guided | 0.122462 | 0.104861 | 0.076743 |
| random | 0.370567 | 0.104881 | 0.076863 |

Frente a la aleatoria, la guiada mejoró el RMSE final medio en 0.000020 (positivo significa menor error con guiada). Comparar también RMSE inicial y las curvas por época: si empieza mejor pero termina igual, su ventaja es acelerar el aprendizaje; si conserva menor error final, además mejora el resultado con este presupuesto de épocas.

RMSE de validación medio durante el entrenamiento:

| Época | Guiada | Aleatoria |
| ---: | ---: | ---: |
| 1 | 0.121604 | 0.321352 |
| 10 | 0.115749 | 0.164464 |
| 50 | 0.107492 | 0.114884 |
| 100 | 0.105604 | 0.107359 |
| 300 | 0.104861 | 0.104881 |

La curva `initialization_curves.png` muestra que la guiada arranca mucho más cerca (RMSE inicial 0.122 frente a 0.371) y conserva ventaja durante las primeras épocas; para la época 300 ambas llegan prácticamente al mismo error. En esta configuración, la inicialización guiada acelera la convergencia, pero no aporta una mejora final relevante.

Estos barridos son exploratorios sobre los mismos folds y sirven para seleccionar hiperparámetros; sus mínimos no son una estimación final independiente. Para informar una evaluación final no sesgada, habría que fijar los hiperparámetros y repetir evaluación con datos no usados en esta selección.
