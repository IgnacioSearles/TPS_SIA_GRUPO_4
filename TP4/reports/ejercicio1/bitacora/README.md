# Bitácora de experimentos — ejercicio 1

Actualización: **21/10/2026**. Esta bitácora registra qué probamos, qué evidencia tenemos y qué decisiones siguen abiertas. Los resultados son sobre 50 creadores; las semillas cambian inicialización y orden de entrenamiento, no constituyen nuevas muestras del catálogo.

## Estado de la evidencia

| Etapa | Estado al corte | Qué permite concluir |
| --- | --- | --- |
| EDA y PCA | Realizados | Selección de variables y referencia lineal |
| Pruebas cortas Kohonen/Oja | Finalizadas, sin fallos registrados | Funcionamiento técnico; no selección de hiperparámetros |
| Pausa y reanudación | Dos corridas de cada algoritmo completadas | Verificación técnica de recuperación |
| Geometría a 500 épocas | Plan original pausado tras 23/300 | Antecedente reutilizado en la continuación |
| Continuación de esas 23 a 1.000 | 23/23 completas | Primera revisión de duración, limitada a mapas pequeños |
| Geometría a 1.000 épocas | **300/300 completas, sin fallos** | Comparaciones de capacidad, forma, topología, inicialización y estabilidad |
| Representaciones a 1.000 épocas | **360/360 completas, sin fallos** | Se mantienen 27 variables estandarizadas, sin log ni PCA |
| Aprendizaje y radio (4×4) | **840/840 completas, sin fallos** | Se confirman tasa 0,3 → 0,01 y radio gaussiano 2 → 0,5 |
| Modelo de interpretación 4×4 y regiones | 5 semillas | 5 regiones estables (ARI 0,79); selección para campañas |
| Oja | 4 tasas × 5 semillas | η = 0,001; el índice es PC1 |
| Resto de etapas | Configuradas; sin ejecución registrada | Duración, early stopping, orden e interacciones pendientes |

## 21/10 — Tasa de aprendizaje y radio

840 corridas en 4×4: 260 de tasa y 580 de radio y vecindad, con ambas topologías y 5 semillas. El radio final es lo que más mueve las métricas:
- con 0,1, el QE baja pero el TE sube a 0,21–0,52, porque el mapa pierde la topología;
- con 1, el QE sube a 3,3;
- con radios constantes grandes, quedan hasta 75 % de neuronas vacías.

Las tasas constantes altas no convergen (con 0,6, TE 0,30). **Decisión: se confirman 0,3 → 0,01 y radio gaussiano 2 → 0,5.** La alternativa 0,6 → 0,01 da las mismas 5 regiones. Detalle en [tasa y radio](../aprendizaje_vecindad_kohonen/README.md).

También se entrenó el mapa sobre 2, 3, 6 y 10 componentes principales. PCA 10 reproduce las regiones (ARI 0,91) y las 12 selecciones de campaña; con menos componentes se pierden los temas o cambian los grupos. Detalle en la [comparación con PCA](../som_pca/README.md).

## 21/10 — Modelo de interpretación, regiones y Oja

El 5×5 exploratorio dejaba 2 creadores por neurona, con 10 neuronas de un solo creador. Se pasa a **4×4 rectangular**, el candidato compacto de geometría, que no deja neuronas vacías. Las neuronas se agrupan en 5 regiones con Ward restringido a vecinas. Se eligieron 5 regiones comparando de 2 a 8: con 5, la pureza de tema es 0,66 (al azar, 0,41), el ARI entre semillas 0,79 y ninguna región tiene menos de 6–7 creadores. Con 5×5, el ARI de las regiones es 0,53.

Las regiones separan tamaño, estilo y tema, pero no audiencia argentina ni costo. Por eso la selección para campañas tiene dos pasos: región según el objetivo y, dentro de ella, costo por cuenta argentina. Detalle en [Kohonen: regiones y campañas](../som/README.md).

Oja: con η₀ = 0,05 diverge en 2 de 5 semillas; con 0,001 converge en menos de 15 épocas a PC1 (1 − 2·10⁻¹⁰). El índice reproduce PC1 y no sirve como selección automática. Detalle en el [índice de Oja](../oja/README.md).

## 20/10 — Representaciones

360 corridas (12 representaciones × 3×3, 4×4 y 5×5 × 2 topologías × 5 semillas) con el protocolo de 1.000 épocas. Como el error de cuantización no se compara entre espacios distintos, se midió qué separa cada mapa frente a variables fijas: pureza de tema principal y η² de tamaño, audiencia argentina y tutoriales, contra una referencia al azar con la misma ocupación. **Decisión: mantener 27 variables estandarizadas.** Con 31 se pierde pureza de tema (−0,10); con 24 o log se pierde separación por audiencia argentina en 5×5; PCA 2 duplica el error topográfico; PCA 25 coincide exactamente con 27, como debía. Detalle en el [informe de representaciones](../representacion_kohonen/README.md).

El [inventario CSV](experimentos.csv) contiene 13 directorios de experimentos y su estado al instante indicado en `snapshot_utc`. Es una fotografía, no un monitor. `finished` cuenta intentos finalizados y `failed` identifica los fallidos; `legacy_finished` identifica manifiestos antiguos sin campo de estado. No aparecen etapas que todavía no tienen manifiesto.

**No sumar las filas como evidencia independiente:** las 23 corridas originales y sus continuaciones están incluidas en las 300 finales de geometría. El plan original permanece pausado como antecedente, pero el barrido recuperado está completo. Los directorios de `results/` son referencias locales y no se publican.

## 17/10 — Preparación y controles técnicos

Se inspeccionaron los datos y se preparó el análisis de las variables numéricas. Partimos de 31 variables y conservamos 27: se excluyeron `price_ars`, `organic_posts_pct`, `total_engagement_per_1000` y `estimated_argentina_reach`. El objetivo fue evitar incorporar el precio como criterio de semejanza y reducir redundancias derivadas o composicionales. No se encontraron faltantes ni duplicados. Se estandarizaron las variables para que sus unidades no dominaran las distancias.

PCA sobre la selección explica aproximadamente **49,71% con PC1**, **58,89% con PC1+PC2** y **91,53% con diez componentes**. Dos componentes sirven para visualizar, pero pierden una porción considerable de la variación. PC1 contrapone escala, vistas y tendencias con otros rasgos de contenido e interacción; no es una medida universal de calidad. Las decisiones y gráficos están en el [informe de EDA/PCA](../README.md).

Se ejecutaron pruebas iniciales: Kohonen (12 corridas de 60 épocas), Oja (6 de 150), controles de refactorización/topologías (4 corridas Kohonen de 8 épocas) y dos controles Oja de 8 épocas, uno con PCA de diez componentes. Son controles de ejecución, no evidencia suficiente de convergencia o superioridad de un modelo.

Se incorporó un único checkpoint reemplazable por corrida y se verificó la pausa/reanudación en dos corridas de cada algoritmo. El análisis sustantivo del índice de Oja frente a PC1 queda pendiente; no confundir los controles técnicos con esa entrega.

## 18/10 — Barrido inicial y revisión de las 500 épocas

La prueba rápida de la batería completó 8/8 corridas. Luego se inició geometría: **15 formas × 2 topologías × 2 inicializaciones × 5 semillas = 300 corridas**, originalmente a 500 épocas. Se pausó con 23 completas y un checkpoint parcial.

Decisión: conservar lo entrenado y extender esas 23 a 1.000 épocas. El decaimiento original termina en la época 500; después se mantienen aprendizaje 0,01 y radio 0,5. Esto permite estudiar la continuación de la misma trayectoria. Repartir el decaimiento sobre 1.000 épocas sería otro experimento.

Las 23 continuaciones completaron sin fallos. El [primer informe de convergencia](../convergencia_kohonen.md) describe ese corte: predominaban mapas pequeños y no habilitaba extrapolar a mapas grandes. Se conserva como antecedente de cómo fue cambiando la decisión.

## 20/10 — Recuperación y geometría completa

La reanudación del plan original detectó un cambio del código respecto de su huella guardada. Se preparó una recuperación explícita que verifica compatibilidad de configuraciones, datos y checkpoints, reutiliza las 23 continuaciones y el entrenamiento parcial, y conserva la procedencia. El resultado consolidado está en `results/kohonen_geometry_1000/`: **300 corridas a 1.000 épocas y cero fallos**.

El protocolo común usa 27 variables estandarizadas, vecindad gaussiana, mezcla por época, aprendizaje exponencial 0,3→0,01 y radio `max(filas,columnas)/2`→0,5 durante 500 épocas; luego valores finales constantes. Las métricas se recalcularon desde los pesos guardados. Hay **60 configuraciones**, cada una con semillas 0–4.

### Patrones y decisiones

| Observación | Evidencia | Decisión / límite |
| --- | --- | --- |
| Más neuronas reducen QE, pero fragmentan | En 8×8, aproximadamente 42–45% de celdas vacías y 49–52% de creadores solos | Revisar ocupación y estabilidad junto con QE; no elegir por mínimo error |
| Estabilidad y resolución compiten | ARI 1 en 2×2; aproximadamente 0,49–0,55 en 8×8 | Un mapa estable puede resumir demasiado; no maximizar ARI aisladamente |
| Hexagonal preserva más vecindades bajo la métrica usada | Menor TE medio en 30/30 comparaciones emparejadas; QE mayor en 28/30 | Hay un compromiso; TE usa hasta seis vecinos en hexagonal frente a cuatro en rectangular |
| QE parecido puede ocultar particiones distintas | 3×4 rectangular: QE 2,9715 vs. 2,9731; ARI 0,756 vs. 0,980 según inicialización | Mantener varias semillas y medir asignaciones, no solo error |
| La forma está mezclada con el radio inicial | 2×8 inicia con radio 4; 4×4 con radio 2 | Controlar radio explícitamente antes de atribuir diferencias a la forma |
| 500 épocas no alcanzan por igual para todos los tamaños | Mejora media 500→1.000: 0,08–0,54% en 2×2 frente a 13,20–15,71% en 8×8 | Usar 1.000 como protocolo siguiente, preservando decaimiento de 500 |
| Casi todas las curvas llegan a una meseta descriptiva | 299/300 cumplen la heurística de ventanas finales | No equivale a óptimo global ni a validación comercial |

La heurística compara dos ventanas finales de 50 épocas: cambio absoluto de QE medio menor al 0,5%, amplitud relativa de QE final menor al 1% y cambio absoluto de TE medio hasta 0,02. La excepción es 8×8 rectangular, uniforme, semilla 1: todavía mejora 0,686% entre ventanas. **Probar 2.000 épocas en ese caso es una propuesta; todavía no está ejecutado.**

La interpretación completa, cifras por configuración y límites están en el [informe de geometría](../geometria_kohonen/README.md). Las comparaciones entre semillas describen sensibilidad del entrenamiento; no prueban generalización a nuevos creadores. Los diez pares de ARI comparten semillas y no son diez ensayos independientes.

### Qué sigue abierto

Para hablar de ventas o adopciones haría falta evidencia comercial externa.

## Resultados resumidos publicados

- [Inventario de ejecuciones y estado al corte](experimentos.csv): metadatos, sin pesos ni datos del catálogo.
- [Resumen de las 60 configuraciones de geometría](../geometria_kohonen/resumen.csv): promedios, dispersión y diagnósticos entre semillas.
- [30 comparaciones de topologías](../geometria_kohonen/topologias.csv).
- [Figura comparativa](../geometria_kohonen/comparacion.png): capacidad, errores, ocupación y estabilidad.

Los datos originales, asignaciones por creador, checkpoints e historiales completos permanecen ignorados por Git. Se publican los agregados y el código para reconstruirlos. Desde `TP4`, con las corridas locales disponibles:

```bash
python3 -m analysis.experiment_inventory
python3 -m analysis.geometry_summary
```

Para añadir una entrada futura: indicar fecha, protocolo, estado y fuente; separar observación de decisión; explicar qué comparación permite y qué limitaciones conserva. Actualizar el CSV con el primer comando antes de publicar un nuevo corte. Los comandos de entrenamiento y recuperación están en el [protocolo](../../../docs/experimentos_kohonen.md).
