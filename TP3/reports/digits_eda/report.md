# EDA de dígitos (datos de aprendizaje)

Muestras: 12449. Cada imagen tiene 28 × 28 píxeles en [0, 1].

Se considera activo un píxel con valor > 0,1. La caja del trazo es el rectángulo mínimo que contiene los píxeles activos. El centro se calcula ponderando las coordenadas por el valor de cada píxel; las coordenadas van de 0 a 27.

| Dígito | Muestras | Proporción | Intensidad media | Píxeles activos (mediana) | Área activa (mediana, Q1–Q3) | Caja ancho × alto (medianas) | Centro x, y (medianas) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1480 | 11.89% | 0.1733 | 176 | 22.4% (19.4%–25.4%) | 19 × 20 | 14.0, 14.0 |
| 1 | 1685 | 13.54% | 0.0767 | 77 | 9.8% (8.4%–11.6%) | 8 × 20 | 14.0, 13.9 |
| 2 | 1489 | 11.96% | 0.1490 | 153 | 19.5% (17.1%–22.1%) | 19 × 20 | 14.0, 14.0 |
| 3 | 1532 | 12.31% | 0.1417 | 148 | 18.9% (15.7%–21.7%) | 16 × 20 | 14.0, 14.0 |
| 4 | 1460 | 11.73% | 0.1211 | 126 | 16.1% (13.8%–18.6%) | 16 × 20 | 14.0, 14.0 |
| 5 | 271 | 2.18% | 0.1283 | 136 | 17.3% (14.5%–20.2%) | 19 × 20 | 14.0, 14.0 |
| 6 | 1479 | 11.88% | 0.1365 | 139 | 17.7% (15.3%–20.5%) | 15 × 20 | 14.0, 14.0 |
| 7 | 1566 | 12.58% | 0.1148 | 118 | 15.1% (12.9%–17.3%) | 16 × 20 | 14.0, 14.0 |
| 8 | 0 | 0% | — | — | — | — | — |
| 9 | 1487 | 11.94% | 0.1228 | 127 | 16.2% (13.9%–18.8%) | 14 × 20 | 14.0, 14.0 |

La intensidad media es el promedio de los 784 valores de píxel de una imagen, promediado por clase. Incluye el fondo; por eso se complementa con la intensidad del trazo, el área activa y su geometría.

## Medidas completas

Se calcularon 23 medidas por imagen: intensidad total y del trazo; píxeles > 0,1 y > 0,5; caja, densidad y proporción; centro y dispersión horizontal/vertical; fracción de masa en la mitad izquierda/superior; simetría y variación entre píxeles vecinos. `image_features.csv` guarda el detalle por imagen y se genera localmente, pero se omite del repositorio por tamaño. `class_features.csv` conserva el resumen por dígito (media, mediana, desvío y cuartiles).

| Familia | Columnas | Interpretación |
| --- | --- | --- |
| Intensidad | `pixel_mean`, `pixel_std`, `total_intensity`, `foreground_mean` | Brillo global, variación, suma de píxeles y brillo de los activos |
| Área activa | `active_pixels`, `active_fraction`, `strong_pixels`, `strong_fraction` | Cantidad y proporción de píxeles > 0,1 y > 0,5 |
| Forma | `bbox_width`, `bbox_height`, `bbox_area`, `bbox_aspect_ratio`, `density_within_bbox` | Tamaño, proporción y ocupación de la caja del trazo |
| Ubicación | `center_x`, `center_y`, `spread_x`, `spread_y`, `mass_left_fraction`, `mass_top_fraction` | Centro, dispersión y distribución de intensidad |
| Regularidad | `horizontal_symmetry_error`, `vertical_symmetry_error`, `horizontal_edge_variation`, `vertical_edge_variation` | Diferencia al reflejar la imagen y entre píxeles vecinos |

Las medidas son descriptivas para el EDA; el MLP recibe directamente los 784 píxeles. La matriz de correlaciones ayuda a ver cuáles medidas aportan información similar.

Imágenes sin píxeles activos: 0. Imágenes duplicadas: 0.

Los gráficos muestran distribución, ejemplos, promedios por clase y dispersión del área activa. El test se reserva para la evaluación final.

**Limitación del aprendizaje:** faltan ejemplos de las clases [8]; no se puede aprender a reconocerlas con este archivo.
