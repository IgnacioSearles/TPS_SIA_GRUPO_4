# Ejercicio 3: balanceo y augmentación

Se compararon las cuatro combinaciones de balanceo y augmentación sobre el mismo
conjunto de validación estratificado (4900 imágenes, diez dígitos), usando tres
semillas. La tabla muestra medias y desvíos entre semillas cuando corresponde.
El conjunto de entrenamiento combina `digits.csv` y `more_digits.csv`, sin
imágenes duplicadas. `digits_test.csv` no interviene en estos resultados.

| Condición | Filas de entrenamiento | Accuracy de validación | F1 macro | Recall del 5 | Recall del 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sin balanceo ni augmentación | 19601 | 96,17 % ± 0,19 | 0,9501 | 87,90 % | 85,75 % |
| Solo balanceo | 25700 | 96,08 % ± 0,25 | 0,9506 | 89,60 % | 87,46 % |
| Solo augmentación | 19601 | 98,07 % ± 0,18 | **0,9760** | 96,18 % | 92,02 % |
| Balanceo y augmentación | 25700 | **98,07 % ± 0,05** | 0,9756 | **97,66 %** | **94,59 %** |

## Lectura

La augmentación da la mejora principal: alrededor de 1,9 puntos porcentuales de
accuracy frente a no usar ninguna de las dos técnicas. El balanceo por sí solo
cambia poco la accuracy y el F1 macro. Sumado a la augmentación, mantiene una
accuracy prácticamente igual, con algo menor F1 macro, pero eleva el recall de
las clases 5 y 8. Si importa más detectar esas clases minoritarias, esa
combinación puede ser preferible.

Este factorial evalúa el efecto de las dos técnicas sobre los datos ampliados;
no compara el uso de `digits.csv` frente al conjunto combinado. Los resultados
por corrida están en `runs.csv` y las curvas de validación en
`validation_curves.png`.

## Reproducir

Desde `TP3`, ejecutar la cola completa:

```bash
python -m experiments.digits.queue experiments/digits/configs/e3_queue.json
```
