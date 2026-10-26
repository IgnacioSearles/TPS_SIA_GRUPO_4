# Kohonen: dos versiones uniformes frente a gaussiana — 26/10/2026

**Finalizaron 1.260/1.260 entrenamientos nuevos sin fallos**, más dos referencias uniformes. Se compararon con 630 modelos gaussianos anteriores, tras verificar hashes de datos y fuentes y reproducir tres controles. La presentación permaneció idéntica según su inventario de hashes. No se guardaron redes, pesos ni copias sintéticas.

## Resultado principal

Las variantes intercambian ajuste y topología. La uniforme que termina en radio 0,5 reduce QE de entrenamiento, pero empeora TE y estabilidad; en las 100 submuestras también empeora el QE de excluidos. La uniforme que termina en 1 conserva vecinas y reduce TE, a cambio de mayor QE de entrenamiento y una pequeña diferencia desfavorable de QE de excluidos frente a la gaussiana de referencia. Estos resultados no eligen una función ganadora universal: los radios de cada función tienen efectos distintos y aquí no se calibró cada familia con igual presupuesto.

El [protocolo](PROTOCOLO.md) conserva condiciones, semillas y limitaciones. Ambas variantes usan el núcleo existente de Kohonen con configuración `neighborhood='uniform'`; no se modificó el núcleo ni el modelo final de campañas.

## Finalistas: treinta semillas por configuración

| Grilla | Variante | QE ↓ | TE ↓ | ARI regiones entre semillas ↑ |
| --- | --- | --- | --- | --- |
| rect4x4 | Gaussiana 2 → 0,5 | 2.7890 | 7.5% | 0.811 |
| hex4x4 | Gaussiana 2 → 0,5 | 2.8519 | 3.5% | 0.823 |
| hex2x8 | Gaussiana 2 → 0,5 | 2.8491 | 0.1% | 0.842 |
| rect4x4 | Uniforme 2 → 0,5 | 2.6480 | 24.3% | 0.669 |
| rect4x4 | Uniforme 2 → 1 | 3.1834 | 2.5% | 0.725 |
| hex4x4 | Uniforme 2 → 0,5 | 2.6122 | 12.1% | 0.573 |
| hex4x4 | Uniforme 2 → 1 | 3.2646 | 0.0% | 0.581 |
| hex2x8 | Uniforme 2 → 0,5 | 2.5873 | 6.5% | 0.767 |
| hex2x8 | Uniforme 2 → 1 | 3.2967 | 0.0% | 0.892 |

QE y TE están calculados sobre el catálogo completo. El ARI compara las cinco regiones, no pesos idénticos ni ventas. TE usa otra adyacencia entre rectangular y hexagonal: comparar funciones dentro de la misma geometría es el contraste más directo. La inicialización, datos, semillas, tasas y número de actualizaciones se mantuvieron.

## Sensibilidad al catálogo: cien submuestras de cuarenta creadores

| Variante | QE entrenamiento | QE excluidos ↓ | TE excluidos ↓ | ARI entre semillas ↑ | Solapamiento arte |
| --- | --- | --- | --- | --- | --- |
| Gaussiana 2 → 0,5 | 2.6963 | 3.6915 | 28.1% | 0.821 | 61.6% |
| Uniforme 2 → 0,5 | 2.4116 | 3.7903 | 36.9% | 0.610 | 61.8% |
| Uniforme 2 → 1 | 3.1347 | 3.7094 | 25.2% | 0.703 | 64.9% |

Cada submuestra ajusta el escalador solo con los 40 incluidos y evalúa diez excluidos sin perturbación. Las tres semillas se promedian por submuestra para comparar diferencias. En cada par de funciones se usa exactamente la misma escala; los valores entre submuestras distintas describen este protocolo, no distancias en una única escala global.

La reducción del QE de entrenamiento de la uniforme 0,5 no se trasladó a mejor representación de excluidos. En la uniforme 1, el mejor orden espacial no implicó el menor QE. Por eso no alcanza con elegir una única métrica.

Las recomendaciones usan las reglas y referencia gaussianas congeladas, con alineación de etiquetas por coincidencia. El solapamiento de arte es estabilidad descriptiva, no validación comercial de los nuevos perfiles. Las matrices U y los perfiles agregados de las dos referencias están en `uniform_references.json`.

## Augmentation: mismo diseño en las tres variantes

Veinte particiones 40/50, tres semillas y máximos de interpolación 0/5/15/30 %, con 40.000 actualizaciones por modelo. Cada intensidad se compara contra cero de su propia variante; la evaluación usa perfiles reales excluidos.

| Variante | Intensidad | Mejora QE excluidos | Particiones con mejora | Criterios que no cumple |
| --- | --- | --- | --- | --- |
| Gaussiana 2 → 0,5 | 5% | 0.002% | 55% | mejora QE, frecuencia |
| Gaussiana 2 → 0,5 | 15% | 0.680% | 70% | mejora QE |
| Gaussiana 2 → 0,5 | 30% | 1.053% | 60% | frecuencia |
| Uniforme 2 → 0,5 | 5% | 0.665% | 75% | mejora QE |
| Uniforme 2 → 0,5 | 15% | 0.795% | 70% | mejora QE |
| Uniforme 2 → 0,5 | 30% | 1.183% | 75% | TE |
| Uniforme 2 → 1 | 5% | 0.347% | 65% | mejora QE, frecuencia |
| Uniforme 2 → 1 | 15% | 0.451% | 70% | mejora QE |
| Uniforme 2 → 1 | 30% | 0.667% | 75% | mejora QE |

**Ningún candidato pasó todos los criterios previos.** La uniforme 0,5 con intensidad 30 % superó el 1 % medio de mejora y mejoró en el 75 % de particiones, pero excedió la tolerancia de aumento de TE. No se promovió augmentation ni se cambiaron los criterios después de observar los resultados.

## Por qué importa el radio

En una grilla con distancia mínima entre neuronas distintas igual a uno, la uniforme con radio menor a uno actualiza únicamente la ganadora. En el calendario 2 → 0,5 eso sucede desde la época 251: durante las últimas 750 épocas no aprenden sus vecinas. La gaussiana con ancho final 0,5 todavía da a una vecina inmediata influencia `exp(-2) ≈ 0,1353`.

La uniforme con radio final 1 incluye a las vecinas inmediatas por la frontera `distancia <= radio`; todas reciben factor uno. La fórmula estricta `<` del apunte se distingue de su descripción de radio 1. La tasa sigue siendo exponencial 0,3 → 0,01 en 500 épocas, no la variante 1/i. Esta prueba cambia vecindad/radio, no todos los detalles posibles del apunte.

## Qué queda separado

- Estos resultados completan las pruebas recientes pedidas; no repiten los 1.500 barridos históricos.
- Oja y Hopfield no se reentrenaron porque no dependen de la vecindad del SOM.
- Radios constantes ya figuraban en el barrido histórico: 20 configuraciones, con cinco semillas cada una, incluyen 0,5/1/2/4/8, ambas funciones y topologías. El nuevo versus anidado se realiza aparte para calibrar y confirmar de manera equilibrada.
- El experimento no modificó `presentation/` ni sustituyó el modelo de referencia. Los resultados quedaron locales, sin push.

## Archivos

`protocol.json`, `control_checks.json`, `controls.jsonl`, `runs.jsonl`, `runs.csv`, `summary.csv`, `summary.json`, tablas por etapa y `augmentation_gates.csv`. Los controles gaussianos de `controls.jsonl` son métricas, no pesos. `comparacion.png` resume los contrastes y `matrices_u.png` permite comparar las dos referencias uniformes. Los CSV usan posiciones de fila, sin nombres de creadores.

Las configuraciones nuevas son `configs/kohonen_uniform_matched.json` y `configs/kohonen_uniform_neighbors.json`. Reproducción y reanudación se describen en el protocolo. Pasaron 119 tests en el cierre de esta tanda (124 después de sumar el versus), incluidos controles de vecindad, presupuestos, hashes y exclusión del escalador.
