# Ejercicio 2: primera corrida

Se implementó y ejecutó una red de referencia de 784 → 32 → 10, con tanh en la
capa oculta, sigmoide en la salida, inicialización Xavier y SGD. Se entrenó durante
50 épocas, con tasa de aprendizaje 0,1, lotes de 128 y semilla 42.

## Resultados medidos

| Medida | Resultado |
|---|---:|
| Ejemplos de entrenamiento | 9.960 |
| Ejemplos de validación | 2.489 |
| Época elegida | 50 |
| Accuracy de entrenamiento | 93,57 % |
| Accuracy de validación | 92,85 % |
| F1 macro de validación (clases con ejemplos) | 0,8877 |
| Cincos reconocidos en validación | 20 de 54 (37,04 %) |

La accuracy de validación antes de entrenar era 8,60 %. El error cuadrático
promedio bajó de 1,3680 a 0,06513. Las curvas siguen mejorando al final de esta
corrida; no hemos establecido que 50 épocas sea la duración óptima.

No hay imágenes del 8 en digits.csv: no se aprendió con ejemplos positivos de esa
clase y la validación no puede medir su reconocimiento. El buen resultado global
convive con un recall bajo del 5, que tiene pocos ejemplos. Su confusión más frecuente
fue con el 3 (12 casos).

Esto es una referencia de desarrollo, no una evaluación final ni una comparación
completa de las variantes exigidas. No se utilizó digits_test.csv ni more_digits.csv.

## Archivos

- `learning_curves.png`: error y accuracy en entrenamiento y validación por época.
- `confusion_matrix.png`: filas reales y columnas predichas del modelo seleccionado.
- `summary.json`: métricas por clase, distribución, hash de datos y criterio de selección.
- `history.csv`: métricas de las 50 épocas.
- `best_weights.npz`: pesos seleccionados; se verificó la igualdad exacta de las
  predicciones después de guardarlos y volver a cargarlos.
- `split_indices.npz`: filas originales de cada partición.
- `validation_predictions.csv`: respuestas por imagen, incluida su fila original.
- `config.json`: configuración efectiva de esta corrida.
- `environment.json`: versiones del entorno utilizado.
- `eda/`: análisis exploratorio generado por el script existente.

## Repetir

Desde la carpeta TP3, usando un entorno con sus requirements.txt instalados:

```bash
python -m experiments.digits.baseline experiments/digits/configs/baseline.json
```

El comando escribe en `TP3/results/digits_baseline`. Estos resultados adjuntos se
obtuvieron con la misma configuración y un directorio de salida distinto.

## Próxima etapa

Primero revisar estas curvas y la fila del 5 en la matriz. Luego comparar tasas de
aprendizaje y arquitecturas con la misma partición, agregar momentum y repetir las
configuraciones finalistas con varias semillas. La evaluación con test queda para
cuando estén fijadas esas decisiones.
