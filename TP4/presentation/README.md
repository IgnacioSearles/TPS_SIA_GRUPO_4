# TP4: presentación

61 diapositivas de relato y un apéndice de 34 diapositivas para preguntas.

Abrir `index.html`. Flechas: avanzar. F: pantalla completa. S: notas del orador. Funciona sin conexión; para compartirla, enviar la carpeta completa.

## Criterios

- Una idea por diapositiva. El título es la conclusión, no el tema.
- Poco texto y grande: números destacados, dos o tres líneas como máximo. Las explicaciones, cifras de detalle y supuestos van en las notas (S).
- Sin tablas: las comparaciones son tarjetas, números enfrentados o gráficos.
- Algunos gráficos de los informes son densos para una diapositiva. `analysis/presentation_figures.py` rehace una versión simple con los mismos datos y modelos (en `reports/presentacion/`); el original queda en el apéndice.

## Estructura

| Bloque | Diapositivas |
|---|---|
| Portada | 1 |
| Ejercicio 1: exploración y PCA · Kohonen · Oja · campañas | 2–37 |
| Ejercicio 2: problema 1 (los cinco QR se parecen) · problema 2 (menú y vinos) · soluciones (desaprendizaje y pseudoinversa) · esquina, daño realista y límites | 38–61 |
| Apéndice | 62–96 |

## Dónde se responde cada pregunta de la consigna

| Pregunta | Diapositiva |
|---|---|
| 1.1 EDA, selección de variables, PCA y reducción | 4–11 |
| 1.2 Hiperparámetros de Kohonen | 13–15 (barridos y vecindad uniforme en el apéndice, 66–69) |
| 1.2 Matriz U y regiones; comparación con el biplot | 16–23 |
| 1.3 Índice de Oja, interpretación y comparación con PC1 | 25–26 |
| 1.4 Fútbol: alcance, descubrimiento o profundidad | 29–30 |
| 1.4 Comparación con elegir por seguidores | 33–35 |
| 1.4 Mantenimiento | 36 |
| 2.1 Ruido, iteraciones y energía | 39–40, 53 |
| 2.1 Limitaciones del caso de uso | 41–49 (crosstalk en la 44), 57–59 |

## Regenerar

Desde `TP4/`:

```
python analysis/presentation_figures.py
python presentation/sync_assets.py
```

El primero rehace los gráficos simplificados (necesita las salidas de `analysis.som_campanas`, `analysis/hopfield_qr_variante.py` y `analysis/hopfield_qr_mejoras.py`, y la evidencia de `reports/ejercicio2/desaprendizaje/`). El segundo copia a `assets/` cada gráfico que usa `index.html` y anota su origen en `FUENTES.json`.

reveal.js está en `vendor/` (ver `vendor/VERSIONS.txt`).

## Experimentos nuevos de Hopfield

La [guía de incorporación y notas](experimentos_hopfield_nuevos.md) resume los resultados confirmados de desaprendizaje y pseudoinversa, distingue Hamming sin red y fija qué afirmaciones sostiene la evidencia. Sus cifras de desaprendizaje y pseudoinversa están en las diapositivas 51–57 y 58.

## Ejercicio 2 compactado y daño realista (26/10)

La parte de Hopfield pasó de 31 a 23 diapositivas. Las fórmulas de la red (Hebb, actualización y energía) quedaron en el apéndice (79). Se agregaron el desaprendizaje en tres pasos (52–54) y la pseudoinversa con un ejemplo de 6 neuronas (56). Las dos redes se presentan como diagnóstico (49) y comparar con los originales como el techo (57). La diapositiva 59 muestra el [daño realista](../reports/ejercicio2/dano_realista_20261026/README.md); el apéndice suma sus curvas y ejemplos (94–96). Los gráficos nuevos salen de `presentation_figures.qr_published`, que solo usa los QR y la evidencia publicada. El 27/10 se agregaron dos comparaciones lado a lado, Hebb contra desaprendizaje (51) y Hebb contra pseudoinversa (55); la diapositiva anterior de la fórmula de la pseudoinversa pasó al apéndice (80).

## Vecindad uniforme frente a gaussiana

La [guía para presentar](experimentos_kohonen_vecindades.md) resume las dos tandas del 26/10 y qué afirmaciones sostienen. Están en la diapositiva 15 y en el apéndice 68–69.

## Material para estudiar y defender

- [Guía de métodos, código, hallazgos y preguntas](guia_estudio.md).
- [Ensayo de las 96 diapositivas](mapa_estudio_diapositivas.md).
- [Ejemplos numéricos ejecutables](ejemplos_estudio.py): desde TP4, `python3 presentation/ejemplos_estudio.py`.

Revisión del 25/10 sobre `ca3229f`; incluye matices para explicar oralmente y respaldo local del 22/10, sin modificar las diapositivas.

- [Verificación del ejercicio 1 contra el PDF](verificacion_enunciado_ejercicio1.md): requisitos cubiertos, flujo real PCA/Kohonen y observaciones de reproducción.
