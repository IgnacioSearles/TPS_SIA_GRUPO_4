# Revisión de la comparación de capacidad RGB

Se completaron seis entrenamientos nuevos (latentes 8 y 16, semillas 7,17,27),
reutilizando los tres modelos de latente 4. No se modificaron la matemática del
VAE, beta, capas ocultas, datos de entrenamiento ni presupuesto de épocas.

## Resultado principal

La validación eligió **latente 8** con el criterio fijado previamente. Fue mejor
que 4 y 16 en IoU media de validación y más estable que 16 entre las tres semillas.
La prueba final independiente confirmó una mejora frente a 4:

| Métrica final | Latente 4 | Latente 8 |
|---|---:|---:|
| IoU macro de silueta | 0,7878 | 0,8736 |
| F1 de borde, tolerancia 1 px | 0,8735 | 0,9644 |
| Distancia media de contorno | 0,6072 px | 0,3203 px |
| MSE dentro de la figura | 0,02785 | 0,01444 |

Son medias de tres semillas sobre las mismas 512 referencias finales. El cambio
de IoU es aproximadamente +8,6 puntos porcentuales; la distancia de contorno baja
un 47 % y el MSE en la figura un 48 %. No son tasas de clasificación correcta.

La mejora incluye las familias que motivaron el estudio:

| Familia, IoU final media | Latente 4 | Latente 8 |
|---|---:|---:|
| Rectángulos | 0,7642 | 0,8460 |
| Triángulos | 0,7064 | 0,7765 |

El latente 16 tuvo IoU de validación 0,8528, frente a 0,8726 del latente 8;
no se abrió la prueba final para 16 porque no fue elegido. No afirmamos que
16 sea peor en cualquier problema: fue menos conveniente en este protocolo.

## Inspección visual realizada

Se revisaron las nueve galerías completas de esta comparación: **576 imágenes
generadas**. También las comparaciones prefijadas de reconstrucción de las tres
semillas, tanto en validación como en la prueba final. No se asignó una etiqueta
cuantitativa de validez a cada generación ni se hizo evaluación humana ciega.

En las reconstrucciones con latente 8 se conservan mejor alargamientos,
orientaciones y algunas puntas que con latente 4. También se reduce la pérdida
de color en varios ejemplos. La mejora numérica tiene una contrapartida visible,
pero sigue habiendo contornos suaves y rectángulos demasiado redondeados.

Un contraejemplo queda a la vista: en la comparación final prefijada de la
semilla 7, el último rectángulo sigue pareciendo circular con latente 8. Esta
figura no se eliminó de la presentación. Por tanto, aumentar el latente ayuda,
pero **no resuelve por completo la pérdida de esquinas**.

En las generaciones desde N(0,I), latentes 8 y 16 producen algunas figuras con
formas y colores reconocibles, junto con ruido de color, bordes difusos y
formas ambiguas. No se observa una limpieza visual uniforme que permita afirmar
que el problema generativo quedó solucionado. La selección de 8 se hizo por
reconstrucción geométrica; no es una certificación de superioridad generativa.

## Conclusión para presentar

El experimento respalda que el tamaño del código era uno de los factores que
limitaban la conservación de información. Ampliarlo a 8 mejora la reconstrucción
en otra partición de la misma distribución, con las tres semillas. Aumentarlo
a 16 no aporta una mejora adicional en la validación de este estudio.

Para el objetivo de imágenes limpias y figuras precisas, el resultado continúa
siendo parcial. Conviene presentar dos conclusiones separadas: mejoró la
representación/reconstrucción y persisten fallos de generación desde el prior.
No se debe usar la mejora de IoU para declarar buenas todas las generaciones.

La relación con sonidos sigue siendo el mismo procedimiento variacional sobre
dos representaciones distintas. La capacidad puede adaptarse a cada modalidad.
No se entrenaron nuevos modelos de sonido ni se compararon numéricamente sus
MSE con los de RGB. Tampoco se entregaron parámetros geométricos al entrenador:
los parámetros de las referencias solo intervienen al evaluar siluetas y color.

## Verificación y reproducibilidad

- Las 146 pruebas anteriores y las 10 nuevas quedaron verificadas; no quedan
  fallas pendientes en la caché de pytest. Una prueba integral detectó un cambio
  del evaluador durante la primera ejecución concurrente. Se repitió con los
  archivos estables y pasó; los resultados reales se calcularon después de fijar
  el evaluador definitivo.
- La calibración de las 512 referencias contra sí mismas dio IoU/F1=1 y contorno=0.
- Validación y prueba final tienen semillas y hashes propios, sin duplicados con
  entrenamiento, prueba histórica ni entre sí. La selección se guardó antes de
  evaluar la prueba final; no se retocó después.
- Se conserva el código, los CSV por muestra/familia/semilla, galerías completas,
  manifiestos y hashes de los pesos. La base y los experimentos previos permanecen.

## Archivos para revisar

- [Informe numérico reproducible](art_capacity_report/report.md).
- [Reconstrucciones finales, semilla 7](art_capacity_final/reconstruction_seed7.png),
  [17](art_capacity_final/reconstruction_seed17.png), [27](art_capacity_final/reconstruction_seed27.png).
- [Comparación de generaciones, semilla 7](art_capacity_validation/generation_seed7.png),
  [17](art_capacity_validation/generation_seed17.png), [27](art_capacity_validation/generation_seed27.png).
- [Protocolo y comandos](protocolo.md).

Las animaciones de la capacidad elegida están en `seleccion/seed=S/presentation/`.
Usan referencias de la prueba histórica, no la nueva prueba de confirmación.
Los pares están prefijados y sus extremos son reconstrucciones.
