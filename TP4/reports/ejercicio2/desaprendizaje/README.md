# Desaprendizaje: implementación y primera prueba

Se implementó desaprendizaje antihebbiano sobre una única red de Hebb con los cinco
QR completos. La recuperación sigue siendo sincrónica y la diagonal permanece en
cero. No se usa pseudoinversa. [Protocolo completo](../../../docs/experimentos_desaprendizaje.md).

## Primera ejecución, 22 de octubre de 2026

Configuración: `configs/hopfield_unlearning_smoke.json`. Modos dirigido y sueños,
eta=0.05, semilla de entrenamiento 0, checkpoints 5 y 20 intentos; cinco QR,
ruido 0/0.2/0.4 y dos semillas de consulta. Se evaluaron 180 consultas en total
contando checkpoints y controles. No son 180 objetos independientes: son daños
repetidos sobre los mismos cinco QR.

Datos regenerables: [directorio de ejecución](evidencia/20261022T223236748114Z/).
La copia publicada contiene `COMPLETED`, configuración efectiva, resultados
individuales comprimidos, estabilidad, historial y resúmenes; no contiene la red.

| Método/checkpoint | QR originales estables | Campos incorrectos, suma sobre cinco QR | Hamming medio de salida, consulta sin ruido | Recuperación exacta, todos los niveles |
| --- | ---: | ---: | ---: | ---: |
| Hebb sin modificar | 0/5 | 715 | 143 | 0% |
| Dirigido, 5 intentos | 0/5 | 358 | 129.8 | 0% |
| Dirigido, 20 intentos | 0/5 | 113 | 22.6 | 0% |
| Sueños, 5 intentos | 0/5 | 358 | 129.8 | 0% |
| Sueños, 20 intentos | 0/5 | 113 | 22.6 | 0% |

Con 20 intentos hubo 20 actualizaciones en ambos modos. En esos checkpoints no se
observaron salidas iguales a la mezcla original ni a su inversa, ni ciclos o topes
en las consultas evaluadas. Sin embargo, ningún original llegó a ser estable y
ninguna recuperación fue exacta: se desplazó el resultado hacia otros estados.

Con ruido 0.4, Hamming medio: 143 para Hebb, 29.7 para dirigido con 20 intentos y
57.1 para sueños con 20 intentos. El vecino por Hamming recuperó exactamente todos
los casos de esta muestra pequeña; es una referencia, no una prueba universal.

## Interpretación y siguiente experimento

La intervención debilita la mezcla y reduce la cantidad de módulos incorrectos,
pero esta prueba NO demuestra que haya resuelto la memoria asociativa. La
estabilidad de los cinco originales sigue siendo el primer requisito pendiente.

Se preparó el barrido exploratorio de 6650 recuperaciones, posteriormente
ejecutado (ver [pruebas completas](PRUEBAS_COMPLETAS.md)): modos dirigido/sueños, tasas 0.01/0.05/0.1, presupuestos
10/50/100 y semillas de entrenamiento 0/1/2 sólo para sueños. Se ejecuta con:

```bash
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_pilot.json
```

El protocolo define selección por estabilidad y recuperación, seguida de una
confirmación con semillas de consulta 100–119 y parámetros congelados. Si ningún
candidato estabiliza los cinco QR, el resultado a informar será negativo bajo
este protocolo, sin reemplazar el criterio por sólo distancia a los originales.

## Repetición sin módulos compartidos

Se repitió la misma prueba corta con 414 neuronas activas y restauración de los 955 módulos comunes. Ver [configuración, resultados y comparación](SIN_COMPARTIDOS.md).

## Pruebas completas finalizadas

Se ejecutaron exploración, selección, confirmación y daño espacial. Ver [resultados completos y comparación con Hamming](PRUEBAS_COMPLETAS.md).
