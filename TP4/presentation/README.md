# TP4: presentación

63 diapositivas de relato y un apéndice de 26 gráficos para preguntas.

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
| Ejercicio 1: exploración y PCA · Kohonen · Oja · campañas | 2–34 |
| Ejercicio 2: el ejercicio · problema 1 (los cinco QR se parecen) · soluciones · problema 2 (menú y vinos) · soluciones · límites | 35–63 |
| Apéndice | 64–90 |

## Dónde se responde cada pregunta de la consigna

| Pregunta | Diapositiva |
|---|---|
| 1.1 EDA, selección de variables, PCA y reducción | 4–11 |
| 1.2 Hiperparámetros de Kohonen | 13 (barridos en el apéndice) |
| 1.2 Matriz U y regiones; comparación con el biplot | 14–21 |
| 1.3 Índice de Oja, interpretación y comparación con PC1 | 23–24 |
| 1.4 Fútbol: alcance, descubrimiento o profundidad | 27–28 |
| 1.4 Comparación con elegir por seguidores | 30–32 |
| 1.4 Mantenimiento | 33 |
| 2.1 Ruido, iteraciones y energía | 37–39, 58 |
| 2.1 Limitaciones del caso de uso | 40–53 (crosstalk en la 44), 61–62 |

## Regenerar

Desde `TP4/`:

```
python analysis/presentation_figures.py
python presentation/sync_assets.py
```

El primero rehace los gráficos simplificados (necesita las salidas de `analysis.som_campanas`, `analysis/hopfield_qr_variante.py` y `analysis/hopfield_qr_mejoras.py`, y la evidencia de `reports/ejercicio2/desaprendizaje/`). El segundo copia a `assets/` cada gráfico que usa `index.html` y anota su origen en `FUENTES.json`.

reveal.js está en `vendor/` (ver `vendor/VERSIONS.txt`).

## Experimentos nuevos de Hopfield

La [guía de incorporación y notas](experimentos_hopfield_nuevos.md) resume los resultados confirmados de desaprendizaje y pseudoinversa, distingue Hamming sin red y fija qué afirmaciones sostiene la evidencia. Sus cifras de desaprendizaje y pseudoinversa están en las diapositivas 57–59 y 61.
