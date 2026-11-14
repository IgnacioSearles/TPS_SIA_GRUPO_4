# Revisión: tipo de autoencoder y ancho de red

Se completaron nueve entrenamientos nuevos y se reutilizaron tres VAE pequeños:
cuatro condiciones, tres semillas, latente 8 y 200 épocas. Se mantuvieron datos,
MSE, Adam y presupuesto. No se entrenaron nuevos modelos de audio.

## Resultado final

| Modelo | IoU de silueta ↑ | Distancia de contorno ↓ |
|---|---:|---:|
| VAE [128,32] | 0,8722 | 0,3237 px |
| AE [128,32] | 0,8681 | 0,3332 px |
| VAE [256,128] | 0,8885 | 0,2758 px |
| AE [256,128] | 0,8903 | 0,2698 px |

Medias de tres semillas sobre las mismas 512 imágenes finales nuevas. La red
ancha fue elegida para ambos tipos usando validación antes de abrir esta prueba.
Los cuatro contrastes y condiciones finales estaban prefijados.

Ampliar el VAE mejoró IoU en **+1,63 puntos porcentuales**; ampliar el AE, en
**+2,23 puntos**. Ambos contrastes fueron positivos en las tres semillas. La
cantidad de parámetros aproximadamente se duplicó: de 0,80 a 1,65 millones.
La reducción dimensional sigue siendo 3.072 valores RGB a un código de 8:
384 a 1, sin contar pesos ni confundirla con compresión de archivos sin pérdida.

A igual ancho, la diferencia AE−VAE fue −0,0041 en la red pequeña y +0,0018 en
la ancha. Los signos cambiaron entre semillas. No hay una ventaja consistente
del AE que respalde culpar principalmente al entrenamiento variacional por los
contornos redondeados. No es una demostración de equivalencia ni de ausencia de
costo de KL: la comparación cambia KL, muestreo y cabeza de varianza juntos.

## Familias que motivaron el estudio

IoU final media por familia:

| Modelo | Triángulos | Rectángulos |
|---|---:|---:|
| VAE [128,32] | 0.7696 | 0.8462 |
| AE [128,32] | 0.7616 | 0.8395 |
| VAE [256,128] | 0.8089 | 0.8596 |
| AE [256,128] | 0.8063 | 0.8593 |

El resultado mejora, pero siguen existiendo errores geométricos. IoU y F1 de
borde no son clasificadores de familia; pueden ser altos con esquinas suavizadas.

## Revisión visual realizada

Se inspeccionaron las doce galerías principales completas: **768 generaciones**.
También las tres comparaciones prefijadas de reconstrucción final, dos referencias
por familia. No se hizo etiquetado cuantitativo de validez ni evaluación humana
ciega. Las 384 imágenes suplementarias del AE desde N(0,I) están disponibles,
pero no se presentan como el método principal de generación del AE.

Las reconstrucciones de redes anchas conservan mejor algunas puntas, elongaciones
y orientaciones. Las diferencias AE/VAE de igual ancho son menos marcadas que
las diferencias entre anchos. Persisten bordes suaves y rectángulos redondeados.
No alcanzamos una reproducción precisa de toda la geometría.

En el VAE ancho se observan varias generaciones de rectángulos y triángulos más
reconocibles y menos artefactos en ciertos casos respecto del VAE pequeño. Sigue
habiendo ruido de color, salidas ambiguas y formas deformadas. Es una mejora
visual moderada, no una solución completa para generar imágenes limpias.

El AE con gaussiana ajustada suele producir fondos limpios, pero también formas
redondeadas y casos defectuosos, incluso con la red ancha. Esta comparación
enfrenta dos procedimientos explícitos: AE con densidad posterior ajustada solo
a entrenamiento y VAE con prior normal estándar. No prueba superioridad intrínseca
de un tipo de red ni igualdad de diversidad o cobertura de familias.

## Ajuste de entrenamiento y generalización

MSE global recomputado en modo de reconstrucción, comparable entre estos modelos:

| Modelo | Entrenamiento | Validación nueva |
|---|---:|---:|
| VAE [128,32] | 0.00305 | 0.00371 |
| AE [128,32] | 0.00335 | 0.00391 |
| VAE [256,128] | 0.00183 | 0.00310 |
| AE [256,128] | 0.00202 | 0.00311 |

La red ancha reduce más el error de entrenamiento que el reservado. Hay una
brecha mayor, por lo que seguir ampliando sin estudiar datos/generalización no
tiene una mejora garantizada. Estos números sugieren una limitación compartida
de representación, pérdida, optimización o cobertura de datos; todavía no
identifican cuál es dominante. Este estudio no varió dataset, pérdida o épocas.

## Conclusión

La capacidad era un factor, pero ampliarla de esta manera aporta una mejora
adicional moderada. El AE no elimina el problema a igual arquitectura. La hipótesis
previa de que el esquema variacional fuera el responsable principal no queda
respaldada por esta comparación.

El VAE ancho es la variante VAE preferida por el criterio geométrico prefijado.
Se puede usar para la presentación junto a los casos fallidos, manteniendo la
misma idea que en sonidos: representación aprendida, reconstrucción, generación
y recorridos. Cambiar ancho entre modalidades no altera la matemática del VAE.
El AE queda como control; no sustituye automáticamente al modelo del ejercicio 2.

No se ajustaron nuevas variantes después de observar la prueba final. No se
modificaron resultados anteriores ni se comparó numéricamente MSE RGB y mel.

## Verificación y archivos

Pasaron tres pruebas nuevas, incluida la integración de las cuatro condiciones,
gaussiana ajustada solo a entrenamiento, conservación de pesos y reutilización.
También pasaron catorce pruebas de los evaluadores geométricos y ajuste gaussiano
reutilizados. No se modificó la implementación de las redes o sus gradientes.

- [Informe numérico reproducible](art_width_report/report.md).
- [Protocolo y comandos](protocolo.md).
- Reconstrucciones finales: [semilla 7](art_width_final/reconstruction_seed7.png),
  [17](art_width_final/reconstruction_seed17.png), [27](art_width_final/reconstruction_seed27.png).
- Comparaciones de generación: [semilla 7](art_width_validation/generation_seed7.png),
  [17](art_width_validation/generation_seed17.png), [27](art_width_validation/generation_seed27.png).
- Galerías completas y vecinos: `art_width_validation/variante/seed=S/`.

Las gaussianas y coordenadas NPZ están en los resultados originales del repositorio;
la copia para revisar en Windows contiene CSV, manifiestos y gráficos. Los modelos
entrenados conservan los GIF y el acceso al explorador en sus carpetas de corrida.
