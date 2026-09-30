# Auditoría de los conjuntos de dígitos

Qué puede y qué no puede enseñar cada archivo. Todo sale de contar las filas de los CSV; no interviene ningún modelo.

## Muestras por dígito

| Dígito | digits.csv | more_digits.csv | digits_test.csv |
| ---: | ---: | ---: | ---: |
| 0 | 1480 | 1776 | 245 |
| 1 | 1685 | 2022 | 283 |
| 2 | 1489 | 1787 | 258 |
| 3 | 1532 | 1839 | 252 |
| 4 | 1460 | 1752 | 245 |
| 5 | 271 | 542 | 223 |
| 6 | 1479 | 1775 | 239 |
| 7 | 1566 | 1879 | 257 |
| 8 | 0 | 585 | 243 |
| 9 | 1487 | 1784 | 252 |
| **Total** | **12449** | **15741** | **2497** |

## Clases ausentes y techo de accuracy

`digits.csv` no tiene **ninguna muestra del dígito 8**. Un modelo entrenado solo con ese archivo nunca puede acertarlos: la unidad de salida correspondiente no vio jamás un ejemplo positivo.

En `digits_test.csv` ese dígito aparece en 243 de 2497 filas, así que la accuracy de test queda acotada por **90,3 %** antes de contar cualquier otro error. Es una cota superior, no una predicción: un modelo real queda por debajo.

El 98 % que pide el enunciado del ejercicio 3 es **inalcanzable** con `digits.csv` solo, por cómo están armados los datos y no por la calidad del modelo.

## Superposición entre archivos

| Par de archivos | Imágenes en ambos |
| --- | ---: |
| digits.csv ∩ more_digits.csv | 3689 |
| digits.csv ∩ digits_test.csv | 0 |
| more_digits.csv ∩ digits_test.csv | 0 |

El test no comparte ninguna imagen con los archivos de entrenamiento (0).

## Unir los archivos de entrenamiento

Concatenar `digits.csv` y `more_digits.csv` da 28190 filas, pero solo **24501 imágenes distintas**: 3689 están repetidas. Unir sin quitar los repetidos hace que esas imágenes pesen el doble que las demás sin que nadie lo haya decidido.

Una vez unidos están los diez dígitos, así que el techo de accuracy sube a **100,0 %**. Ese salto es lo que explica la mayor parte de la mejora entre los ejercicios 2 y 3, y no las técnicas que se apliquen encima.

## Desbalance que queda

El reparto sigue siendo desparejo: el dígito 1 aparece 3212 veces y el 8 solo 585, una razón de **5,5 a 1**. El test, en cambio, está parejo (alrededor del 10 % por dígito), así que los dígitos escasos pesan en la nota final mucho más de lo que pesaron durante el entrenamiento. Conviene compensarlo con pesos por clase o remuestreo.

## Qué hacer con esto

1. Unir los archivos de entrenamiento quitando las 3689 imágenes repetidas.
2. Compensar el desbalance que queda en los dígitos escasos.
3. Reservar una parte para validación y dejar `digits_test.csv` para una única medición final.
4. Al comparar técnicas, mirar el error de validación (generalización) y no el de entrenamiento.

## Reproducir

```bash
python -m experiments.digits.dataset_audit
```

