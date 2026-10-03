# Experimentos nuevos del ejercicio 3

Se proponen tres etapas sobre la unión deduplicada de `digits.csv` y
`more_digits.csv`. Las tres usan el mismo split estratificado nuevo (`split_seed`
2026) y tres semillas de inicialización. La validación incluye las diez clases.

## 1. Factorial balanceo × augmentación

Cruza las cuatro condiciones para separar efectos e interacción: ninguna técnica,
solo balanceo, solo augmentación y ambas. Las variantes balanceadas usan el mismo
remuestreo para que la diferencia mida la augmentación y no una muestra distinta
de duplicados. Las transformaciones se aplican solo al conjunto de entrenamiento.

Selección: si alguna condición alcanza 98 % de accuracy media en validación, gana
la de mayor F1 macro entre esas condiciones; si ninguna llega, gana la de mayor
accuracy y F1 macro desempata.

## 2. Arquitecturas sobre los datos ampliados

Compara una red de una capa, las referencias de dos capas, y redes de 3, 4, 6 y
8 capas ocultas con ancho 64. Mantiene balanceo y augmentación constantes. También
incluye `[784,128,128,10]`, que ganó el barrido de ancho del ejercicio 2.

Usa la misma regla de selección de accuracy mínima 98 % y F1 macro. Los modelos
guardan sus salidas por arquitectura y semilla, así se puede retomar la cola sin
repetir corridas completas.

## 3. Robustez ante ruido gaussiano

Tras elegir arquitectura con validación, mide el modelo en `digits_test.csv` con
ruido de desviación `σ = 0, 0.05, 0.1, 0.2, 0.4`, con cinco realizaciones por
nivel distinto de cero. Informa accuracy, F1 macro y recall por dígito.

El test ya tuvo evaluaciones previas en el proyecto, así que esta etapa es un
diagnóstico descriptivo: sus resultados no se usan para volver a elegir o ajustar
el modelo.

## Ejecución

Desde `TP3`:

```bash
python -m experiments.digits.queue experiments/digits/configs/e3_queue.json
```

La cola ejecuta las etapas en orden y registra su estado en
`reports/digits_e3_queue/queue_state.json`. Los barridos guardan el resultado de
cada corrida completa; al relanzar la cola, se omiten esas corridas. Si se corta
un entrenamiento a mitad, esa combinación de arquitectura y semilla se reinicia.
La prueba de ruido no entrena modelos y es rápida.
