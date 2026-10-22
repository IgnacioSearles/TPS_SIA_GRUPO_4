# Desaprendizaje sin módulos compartidos

## Configuración y comparación

Se repitió la prueba corta el 22 de octubre de 2026 con los cinco QR, quitando
955 módulos que valen lo mismo en todos ellos. Una única red almacena los 414
módulos restantes. Al devolver un QR, los módulos compartidos se reponen con sus
valores conocidos; no se intenta inferirlos a partir de la consulta dañada.

Configuración: `configs/hopfield_unlearning_stripped_smoke.json`.
Modos dirigido y sueños, eta=0.05, checkpoints 5/20 intentos, semilla de
entrenamiento 0, ruido 0/0.2/0.4 y semillas de consulta 0/1. Son los mismos
parámetros y las mismas consultas completas de la [prueba con 1369 neuronas](README.md).
Se genera el ruido ANTES de quitar los módulos, con la misma utilidad y semillas.
Las tasas se normalizan por N=414, la cantidad de neuronas efectiva, tanto en Hebb
como en cada update antihebbiano.

Se evaluaron **180 recuperaciones**, contando cuatro checkpoints entrenados y dos
controles, cada uno con 30 consultas. Por nivel hay sólo diez consultas distintas
(cinco QR por dos semillas); p=0 repite los mismos cinco originales intactos.

Datos: [ejecución completa](evidencia/20261022T224138937279Z/).
`config.json` registra dimensiones, cantidad de módulos eliminados y hashes de
los patrones completos y reducidos. `COMPLETED` confirma el final de la ejecución.

## Resultados de recuperación exacta y estable

| Método, sin compartidos | Originales estables | Ruido 0 | Ruido 0.2 | Ruido 0.4 |
| --- | ---: | ---: | ---: | ---: |
| Hebb sin desaprendizaje | 0/5 | 0/10 | 0/10 | 0/10 |
| Dirigido, 5 intentos | 1/5 | 2/10 | 1/10 | 1/10 |
| Dirigido, 20 intentos | **5/5** | **10/10** | **10/10** | 4/10 |
| Sueños, 5 intentos | 1/5 | 2/10 | 1/10 | 1/10 |
| Sueños, 20 intentos | **5/5** | **10/10** | 8/10 | **6/10** |
| Vecino por Hamming, sin red | No aplica | 10/10 | 10/10 | 10/10 |

Los 20 intentos dirigidos produjeron 20 updates. En sueños hubo 18 updates y dos
salidas protegidas (recuerdo o inverso), que no se penalizaron. No se observaron
ciclos ni topes en las consultas de evaluación de esta muestra.

En ambos modos, la suma de campos incorrectos sobre los cinco originales pasó
de 130 (Hebb) a 86 (5 intentos) y a cero (20 intentos). El Hamming medio de salida
al consultar originales intactos pasó de 70.6 a cero. Con ruido 0.4 fue 77.8 para
Hebb, 32.4 para dirigido con 20 intentos y 24.5 para sueños con 20 intentos.

El modo dirigido con 20 intentos recuperó cócteles/bebidas y especiales en las dos
consultas de ruido 0.4, pero ninguna de menú principal, vinos o postres. En sueños,
recuperó cócteles/bebidas y especiales en ambas, menú principal y postres en una
de dos y vinos en ninguna. Con ruido 0.2, los dos fallos de sueños fueron vinos.

## Qué cambió frente a la red completa

La prueba completa con 1369 neuronas y 20 intentos no estabilizó ningún original
ni consiguió recuperación exacta en los niveles evaluados. La variante reducida
sí estabilizó los cinco y recuperó parte de las consultas dañadas. La restauración
de módulos comunes aporta información conocida y corrige automáticamente cualquier
daño en esas posiciones; forma parte del método y debe explicitarse al compararlo.

No hay un ganador universal entre dirigido y sueños: en esta muestra dirigido
fue mejor con ruido 0.2 y sueños con ruido 0.4. La muestra no permite inferir que
esas diferencias se mantendrán con otras semillas.

## Siguiente paso preparado

Se agregó `configs/hopfield_unlearning_stripped_pilot.json`, la misma exploración
de **6650 recuperaciones** con `remove_shared: true`, posteriormente ejecutada.
Ver [pruebas completas, confirmación y daño espacial](PRUEBAS_COMPLETAS.md).
Se mantiene el criterio de selección: cinco originales estables, recuperación
media con peso igual por nivel/QR y confirmación con ruido nuevo, sin ajustar
después de observar la confirmación. [Protocolo](../../../docs/experimentos_desaprendizaje.md).

```bash
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_stripped_pilot.json
```

En esta variante, mezcla e inversa, categorías y campos se definen en el espacio
de 414 neuronas. El Hamming de salida se calcula sobre el QR completo restaurado.
`query_hamming` mide el daño completo y `network_query_hamming` sólo el daño de
los módulos que recibe efectivamente la red. Invertir el estado activo y luego
restaurar los compartidos NO equivale a invertir todo el QR.
