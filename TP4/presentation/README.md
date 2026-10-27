# TP4: presentación

59 diapositivas de relato y un apéndice de 38 diapositivas para preguntas.

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
| Ejercicio 2: problema 1 (los cinco QR se parecen) · problema 2 (menú y vinos) · soluciones (desaprendizaje y pseudoinversa) · daño realista · conclusiones y recomendaciones | 38–60 |
| Apéndice | 61–99 |

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
| 2.1 Ruido, iteraciones y energía | 39–40, 54 |
| 2.1 Limitaciones del caso de uso | 41–49 (crosstalk en la 44), 57–59 |

## Regenerar

Desde `TP4/`:

```
python analysis/presentation_figures.py
python presentation/sync_assets.py
```

El primero rehace los gráficos simplificados (necesita las salidas de `analysis.som_campanas`, `analysis/hopfield_qr_variante.py` y `analysis/hopfield_qr_mejoras.py`, y la evidencia de `reports/ejercicio2/desaprendizaje/`). El segundo copia a `assets/` cada gráfico que usa `index.html` y anota su origen en `FUENTES.json`.

reveal.js está en `vendor/` (ver `vendor/VERSIONS.txt`).

## Ejemplos para estudiar

Desde TP4, `python3 presentation/ejemplos_estudio.py` corre ejemplos numéricos chicos de cada método.
