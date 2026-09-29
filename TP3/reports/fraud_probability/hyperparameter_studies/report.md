# Estudios de hiperparámetros: learning rate e inicialización

Se usó K-Fold estratificado de 5 particiones sobre las 6000 filas de desarrollo (el test apartado no se usa), sigmoide de salida, MSE/2 y 300 épocas por corrida. En cada fold, el escalador y la inicialización guiada se ajustan usando exclusivamente las filas de entrenamiento.

## Learning rate

| Learning rate | MAE validación medio | RMSE validación medio ± desvío | RMSE train medio |
| ---: | ---: | ---: | ---: |
| 0.001 | 0.177645 | 0.208919 ± 0.015945 | 0.208758 |
| 0.005 | 0.103956 | 0.129157 ± 0.008319 | 0.128933 |
| 0.01 | 0.088807 | 0.114260 ± 0.003634 | 0.114098 |
| 0.025 | 0.078977 | 0.105884 ± 0.001292 | 0.105752 |
| 0.05 | 0.076862 | 0.104728 ± 0.001209 | 0.104604 |
| 0.1 | 0.076496 | 0.104655 ± 0.001230 | 0.104535 |
| 0.2 | 0.076474 | 0.104653 ± 0.001232 | 0.104535 |

El mejor valor explorado fue lr=0.2, con RMSE de validación medio 0.104653. La curva `learning_rate_curves.png` permite comparar velocidad de convergencia y estabilidad entre tasas.
En esta corrida, lr=0.001 quedó en RMSE 0.208919; en el valor cercano a 0.05 (0.05) bajó a 0.104728. El mejor explorado fue 0.2; frente al mayor valor probado (0.2), el error fue 0.104653 vs. 0.104653. Valores cercanos a la meseta son prácticamente equivalentes; revisá las curvas para detectar inestabilidad en los puntos evaluados.

## Inicialización guiada

La inicialización guiada ajusta una regresión lineal de `logit(probabilidad BigModel)` sobre las features estandarizadas del train del fold. Sus coeficientes e intercepto inicializan la capa sigmoide; las probabilidades objetivo 0/1 se recortan solo para poder calcular el logit. Se agrega el ruido pequeño configurado para empezar cerca de ese ajuste. No se usa la validación para construir los pesos.

| Inicialización | RMSE validación inicial medio | RMSE validación final medio | MAE validación final medio |
| --- | ---: | ---: | ---: |
| guided | 0.120023 | 0.104675 | 0.076665 |
| random | 0.369545 | 0.104728 | 0.076862 |

Frente a la aleatoria, la guiada mejoró el RMSE final medio en 0.000053 (positivo significa menor error con guiada). Comparar también RMSE inicial y las curvas por época: si empieza mejor pero termina igual, su ventaja es acelerar el aprendizaje; si conserva menor error final, además mejora el resultado con este presupuesto de épocas.

RMSE de validación medio durante el entrenamiento:

| Época | Guiada | Aleatoria |
| ---: | ---: | ---: |
| 1 | 0.119367 | 0.328479 |
| 10 | 0.114716 | 0.174721 |
| 50 | 0.107340 | 0.117372 |
| 100 | 0.105498 | 0.108246 |
| 300 | 0.104675 | 0.104728 |

La curva `initialization_curves.png` muestra que la guiada arranca mucho más cerca (RMSE inicial 0.122 frente a 0.371) y conserva ventaja durante las primeras épocas; para la época 300 ambas llegan prácticamente al mismo error. En esta configuración, la inicialización guiada acelera la convergencia, pero no aporta una mejora final relevante.

Estos barridos son exploratorios sobre los mismos folds y sirven para seleccionar hiperparámetros; sus mínimos no son una estimación final independiente. La evaluación no sesgada es la del conjunto de test apartado, que ninguno de estos barridos usa.
