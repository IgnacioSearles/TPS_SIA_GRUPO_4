# TP3 — perceptrones y probabilidad de fraude

El Ejercicio 1 usa las seis variables seleccionadas en `experiments/fraud_data.py` para aproximar la columna continua `big_model_fraud_probability`. La pérdida de entrenamiento es MSE/2; no se toman decisiones binarias en este experimento.

## Ejecutar

Desde este directorio, con las dependencias de `requirements.txt` instaladas:

```bash
python -m experiments.analyze_fraud
python -m experiments.fraud_probability experiments/configs/fraud_probability.json
python -m experiments.analyze_epoch_sweep
python -m experiments.study_hyperparameters experiments/configs/fraud_hyperparameters.json
pytest -q
```

El primer comando escribe el análisis exploratorio en `reports/fraud_eda/`, con gráficos de todas las variables y de las seleccionadas. El segundo compara una capa lineal con una sigmoide usando todas las muestras, ejecuta K-Fold con la sigmoide y entrena el modelo final con las 7.500 muestras. La configuración incluida corre hasta 1.000 épocas; los resultados quedan en `reports/fraud_probability/`:

- `report.md` y `summary.json`: conclusiones y métricas.
- `experiment_config.json`: configuración exacta de la corrida.
- `learning_curves.png`, `learning_*.csv` y `learning_*_weights.npz`: historial y pesos de cada comparación lineal/sigmoide.
- `cross_validation.csv`, `cross_validation_epoch_metrics.csv` y `cross_validation_error.png`: resultados por fold y evolución por época de train/validación.
- `epoch_sweep.csv`, `epoch_sweep.md` y `epoch_sweep.png`: comparación de varios cortes de épocas para underfitting y overfitting.
- `hyperparameter_studies/report.md`: efecto del learning rate y de la inicialización guiada, con curvas e historiales por fold en `reports/fraud_probability/hyperparameter_studies/`.
- `fold_XX_history.csv`, `fold_XX_weights.npz` y `fold_XX_metadata.json`: historial completo, pesos y escalador de cada ejecución de K-Fold.
- `out_of_fold_predictions.csv`: predicción fuera de muestra para cada transacción.
- `model_weights.npz` y `model_metadata.json`: pesos, orden de variables y estandarización para inferencia.

Se puede modificar el JSON para elegir cantidad de folds, épocas, tamaño de lote, tasa de aprendizaje y `training.epsilon`. El epsilon compara contra la pérdida MSE/2 medida con el modelo actualizado al final de cada época; `epochs` es el máximo. En cada fold, la media y el desvío se calculan solo con sus muestras de entrenamiento.

El estudio de learning rate compara varios valores con la misma inicialización aleatoria, particiones y orden de mini-batches. El estudio de inicialización guiada ajusta una regresión sobre el logit de las probabilidades objetivo usando solo el train de cada fold y usa esos coeficientes para empezar cerca de una solución; se compara con el inicio aleatorio usando el mismo learning rate. El informe incluye el efecto observado, tablas por fold, curvas y configuración exacta.

## Usar el modelo guardado

```python
import pandas as pd
from experiments.fraud_probability import predict_probability

transacciones = pd.read_csv("datasets/fraud_dataset.csv")
probabilidades = predict_probability(transacciones, "reports/fraud_probability")
```

`probabilidades` contiene un valor entre 0 y 1 por transacción. La función toma las variables en el orden guardado junto al modelo y les aplica el escalador final.
