# Barrido de profundidad de la red para dígitos

El estudio previo comparó 1 y 2 capas ocultas, pero solo con una semilla. Este
barrido fija el ancho en 64 neuronas por capa para explorar la profundidad, usa la
misma partición 80/20 y el mismo orden de lotes, y repite cada profundidad con
tres inicializaciones (semillas 42, 7 y 21).

El protocolo llega hasta 8 capas ocultas. La primera corrida se detuvo por una regla
de plateau después de probar 3 capas. La continuación reutiliza esas tres corridas y
desactiva esa parada para completar las profundidades 4 a 8. Es un criterio exploratorio: se informa
la accuracy media y su desvío, además de pérdida, F1 macro, recall del dígito 5,
época elegida y parámetros. La validación no incluye el dígito 8 en `digits.csv`.
Al aumentar la profundidad también aumenta la cantidad de parámetros; se reporta
ese conteo para interpretar las mejoras junto con el costo del modelo.

Cada corrida tiene un máximo de 350 épocas y también para antes si pasan 40 épocas
sin que la accuracy de validación mejore al menos 0,05 puntos porcentuales. La
configuración fija momentum 0,9 y tasa 0,1, manteniéndolos
constantes en todo el barrido. No se usa `digits_test.csv` ni `more_digits.csv`.

Desde `TP3`:

```bash
python -m experiments.digits.depth_sweep experiments/digits/configs/depth_sweep.json
```

Los resultados quedan en `results/digits_depth_sweep`: `runs.csv` contiene cada
semilla, `depth_summary.csv` resume por cantidad de capas y `study_summary.json`
registra el resultado. Cada modelo guarda además sus pesos e historial en
`depth_NN/seed_SEED/`. La continuación también genera `depth_comparison.png` con
accuracy y F1 macro frente a profundidad, y `depth_learning_curves.png` con las
curvas de validación medias y su variación entre semillas.

## Cola de experimentos

La cola incluye este barrido de profundidad y una comparación del ancho con dos
capas ocultas (32, 64 y 128 neuronas por capa), ambas con tres semillas y el mismo
protocolo de parada temprana. Desde `TP3`, ejecutar:

```bash
python -m experiments.digits.queue
```

La cola corre los estudios en orden y registra el estado en
`results/digits_experiment_queue/queue_state.json`. Si se interrumpe, al volver a
ejecutarla salta los estudios completos. Dentro de cada estudio también salta las
corridas completas. Una corrida interrumpida a mitad de entrenamiento empieza de
nuevo; todavía no se guarda el estado del optimizador por época.
