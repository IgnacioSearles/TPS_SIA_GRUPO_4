# TP4: presentación

59 diapositivas de relato y un apéndice de 32 gráficos para preguntas.

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
| Ejercicio 1: exploración y PCA · Kohonen · Oja · campañas | 2–35 |
| Ejercicio 2: cómo funciona la red · problema 1 (los cinco QR se parecen) · problema 2 (menú y vinos) · soluciones (desaprendizaje y pseudoinversa) · esquina, daño realista y límites | 36–59 |
| Apéndice | 60–92 |

## Dónde se responde cada pregunta de la consigna

| Pregunta | Diapositiva |
|---|---|
| 1.1 EDA, selección de variables, PCA y reducción | 4–11 |
| 1.2 Hiperparámetros de Kohonen | 13–14 (barridos y vecindad uniforme en el apéndice, 65–68) |
| 1.2 Matriz U y regiones; comparación con el biplot | 15–22 |
| 1.3 Índice de Oja, interpretación y comparación con PC1 | 24–25 |
| 1.4 Fútbol: alcance, descubrimiento o profundidad | 28–29 |
| 1.4 Comparación con elegir por seguidores | 31–33 |
| 1.4 Mantenimiento | 34 |
| 2.1 Ruido, iteraciones y energía | 38–39, 52 |
| 2.1 Limitaciones del caso de uso | 40–48 (crosstalk en la 43), 56–58 |

## Regenerar

Desde `TP4/`:

```
python analysis/presentation_figures.py
python presentation/sync_assets.py
```

El primero rehace los gráficos simplificados (necesita las salidas de `analysis.som_campanas`, `analysis/hopfield_qr_variante.py` y `analysis/hopfield_qr_mejoras.py`, y la evidencia de `reports/ejercicio2/desaprendizaje/`). El segundo copia a `assets/` cada gráfico que usa `index.html` y anota su origen en `FUENTES.json`.

reveal.js está en `vendor/` (ver `vendor/VERSIONS.txt`).

## Experimentos nuevos de Hopfield

La [guía de incorporación y notas](experimentos_hopfield_nuevos.md) resume los resultados confirmados de desaprendizaje y pseudoinversa, distingue Hamming sin red y fija qué afirmaciones sostiene la evidencia. Sus cifras de desaprendizaje y pseudoinversa están en las diapositivas 50–55 y 56.

## Ejercicio 2 compactado y daño realista (26/10)

La parte de Hopfield pasó de 31 a 24 diapositivas. Se agregaron cómo funciona la red (37), el desaprendizaje en tres pasos (50–52) y la pseudoinversa con un ejemplo de 6 neuronas (53–54). Las dos redes se presentan como diagnóstico (48) y comparar con los originales como el techo (55). La diapositiva 57 muestra el [daño realista](../reports/ejercicio2/dano_realista_20261026/README.md); el apéndice suma sus curvas y ejemplos (90–92). Los gráficos nuevos salen de `presentation_figures.qr_published`, que solo usa los QR y la evidencia publicada.

## Vecindad uniforme frente a gaussiana

La [guía para presentar](experimentos_kohonen_vecindades.md) resume las dos tandas del 26/10 y qué afirmaciones sostienen. Están en la diapositiva 14 y en el apéndice 67–68.

## Material para estudiar y defender

- [Guía de métodos, código, hallazgos y preguntas](guia_estudio.md).
- [Ensayo de las 92 diapositivas](mapa_estudio_diapositivas.md).
- [Ejemplos numéricos ejecutables](ejemplos_estudio.py): desde TP4, `python3 presentation/ejemplos_estudio.py`.

Revisión del 25/10 sobre `ca3229f`; incluye matices para explicar oralmente y respaldo local del 22/10, sin modificar las diapositivas.

- [Verificación del ejercicio 1 contra el PDF](verificacion_enunciado_ejercicio1.md): requisitos cubiertos, flujo real PCA/Kohonen y observaciones de reproducción.
