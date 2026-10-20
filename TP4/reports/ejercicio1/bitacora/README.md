# Bitácora de experimentos — ejercicio 1

Actualización: **20/10/2026**. Esta bitácora registra qué probamos, qué evidencia tenemos y qué decisiones siguen abiertas. Los resultados son sobre 50 creadores; las semillas cambian inicialización y orden de entrenamiento, no constituyen nuevas muestras del catálogo.

## Estado de la evidencia

| Etapa | Estado al corte | Qué permite concluir |
| --- | --- | --- |
| EDA y PCA | Realizados | Selección de variables y referencia lineal |
| Pruebas cortas Kohonen/Oja | Finalizadas, sin fallos registrados | Funcionamiento técnico; no selección de hiperparámetros |
| Pausa y reanudación | Dos corridas de cada algoritmo completadas | Verificación técnica de recuperación |
| Geometría a 500 épocas | Plan original pausado tras 23/300 | Antecedente reutilizado en la continuación |
| Continuación de esas 23 a 1.000 | 23/23 completas | Primera revisión de duración, limitada a mapas pequeños |
| Geometría a 1.000 épocas | **300/300 completas, sin fallos** | Comparaciones de capacidad, forma, topología, inicialización y estabilidad |
| Representaciones a 1.000 épocas | En ejecución; ver corte exacto en CSV | Todavía no hay conclusiones del barrido completo |
| Resto de etapas | Configuradas; sin ejecución registrada | Aprendizaje, radios, duración, early stopping, orden e interacciones pendientes |

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

El barrido de representaciones ya está ejecutándose con 1.000 épocas y decaimiento de 500. No vamos a comparar directamente QE de espacios con distintas variables o dimensiones. Las alternativas incluyen selección estandarizada, transformaciones logarítmicas y PCA; deben evaluarse también por ocupación, estabilidad e interpretación.

Después corresponde controlar aprendizaje y radios, confirmar candidatos con semillas nuevas e interpretar perfiles y matriz U. El análisis de campañas previo usaba 5×5 rectangular, 500 épocas y semilla 0: es exploratorio y debe regenerarse para el modelo elegido. Faltan la interpretación final de Oja y evidencia comercial externa para hablar de ventas o adopciones.

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
