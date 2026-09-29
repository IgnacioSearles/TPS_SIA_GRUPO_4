# Ejercicio 2: primera red de referencia

Desde `TP3`, con las dependencias de `requirements.txt` instaladas:

```bash
python -m experiments.digits.baseline experiments/digits/configs/baseline.json
```

Los paths del JSON son relativos al directorio desde el que se ejecuta. Los resultados
se guardan por defecto en `results/digits_baseline` (ignorado por Git); para conservar
varias corridas, cambiar `output.directory`. Una nueva corrida en el mismo directorio
reemplaza sus archivos.

## Recorrido del experimento

1. Lee exclusivamente `digits.csv`: 784 píxeles normalizados por imagen.
2. Separa aproximadamente 80 % para entrenamiento y 20 % para validación mediante
   el splitter estratificado existente. `data.split_seed` fija las filas de cada grupo.
3. Convierte cada etiqueta en un objetivo de diez componentes: la columna del dígito
   vale 1 y las demás 0 (one-hot). No vuelve a normalizar los píxeles.
4. Construye `[784,32,10]`, con tanh oculta y sigmoide de salida. Xavier inicializa
   cada matriz en ±sqrt(6/(entradas+salidas)); los términos independientes comienzan en 0.
5. Entrena 50 épocas con SGD, tasa 0.1 y lotes de 128. `seed` controla inicialización
   y orden de lotes. No hay parada temprana: se recorre el presupuesto completo.
6. Evalúa entrenamiento y validación por época. Selecciona mayor accuracy de
   validación y, en empate, menor error cuadrático; si persiste el empate conserva
   la primera época. Restaura una copia de esos pesos al terminar.
7. Guarda el modelo seleccionado y verifica que cargarlo reproduzca exactamente
   sus salidas de validación. No evalúa `digits_test.csv` ni lee `more_digits.csv`.

En la notación teórica, las entradas son ξ, los objetivos ζ y las salidas O.
El error informado es (1/(2p)) Σμ Σj (O_j^μ − ζ_j^μ)². La clasificación elige la
salida máxima (argmax); las diez sigmoides no son probabilidades que sumen uno.

## Reportes

- `config.json`: configuración resuelta para repetir el experimento.
- `split_indices.npz`: posiciones de las filas de entrenamiento y validación.
- `summary.json`: hash SHA-256 del CSV, cantidades por clase, métricas iniciales,
  métricas del modelo seleccionado y época seleccionada.
- `history.csv`: error y accuracy por época; el tiempo del entrenador no incluye
  las evaluaciones adicionales del callback.
- `best_weights.npz`: pesos de la época seleccionada.
- `validation_predictions.csv`: fila original (base cero, sin contar encabezado),
  etiqueta, predicción y diez salidas.
- `learning_curves.png` y `confusion_matrix.png`: curvas y matriz de validación.

La matriz siempre tiene 10×10 posiciones: filas reales, columnas predichas.
`recall` y `f1` son null cuando no hay ejemplos reales de una clase; `precision`
es null si no hay predicciones de esa clase. F1 macro promedia **solamente las
clases con soporte real**, enumeradas en `macro_labels`; si una clase tiene
soporte pero nunca se predice, su F1 es cero.

En el dataset revisado hay 12.449 imágenes, ninguna de 8 y solo 271 de 5.
Por tanto, validación no mide reconocimiento de 8. Un buen accuracy no permite
afirmar que la red reconoce todas las clases.

## Próximos pasos

- Analizar curvas, errores y recall del 5 de esta referencia.
- Comparar tasas de aprendizaje y arquitecturas manteniendo la partición.
- Implementar momentum para comparar mecanismos de optimización.
- Repetir finalistas con varias semillas; seleccionar con validación.
- Fijar el procedimiento final de entrenamiento y recién entonces evaluar test.

Esta primera corrida no completa las comparaciones obligatorias del ejercicio 2.
