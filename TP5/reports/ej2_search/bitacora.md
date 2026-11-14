# Bitácora del estudio generativo: imágenes y sonidos

Se distinguen resultados terminados, decisiones e hipótesis pendientes. Las
fechas son las de Buenos Aires. Esta bitácora continúa el estilo de informes
del repositorio: [cierre de notas omitidas](../ej2_notas_omitidas/cierre.md),
[búsqueda 1a](https://github.com/IgnacioSearles/TPS_SIA_GRUPO_4/blob/597b555/TP5/reports/ej1_busqueda/resumen.md)
y [búsqueda 1b](https://github.com/IgnacioSearles/TPS_SIA_GRUPO_4/blob/597b555/TP5/reports/ej1b_busqueda/resumen.md).
Las dos últimas están en `origin/otras-redes-para-1a`; se consultaron sin fusionar
la rama. `origin/main` seguía en `fec6b30` al hacer fetch el 14 de noviembre.

## Antecedentes incorporados al registro — 13 de noviembre de 2026

- Se cerró el estudio de alturas omitidas con evidencia negativa, no se reabre.
- Se prepararon experimentos de figuras, AE/VAE y beta, y timbres a altura fija.
  El AE con gaussiana ajustada es un control de muestreo, no un VAE. Ver
  [experimentos](../../docs/generative_experiments.md) y
  [análisis del muestreo](../../docs/latent_sampling_results.md).
- Se agregó arte RGB, composiciones, evaluación de capacidad y ancho. Las
  figuras tendían a redondearse; mejorar MSE no bastaba para asegurar geometría.
  Ver [arte y sonidos](../../docs/art_audio_results.md),
  [capacidad](../../docs/art_capacity_results.md) y
  [ancho](../../docs/art_width_results.md).
- Se abrió una búsqueda convolucional manual NumPy: cantidad de datos,
  resolución, latente, ancho, tasa, beta y refinamiento aprendido del decoder.
  No se tocaron las fórmulas de AE/VAE ni se añadieron pérdidas auxiliares.
- Control con igual fuente y 50 épocas: sin capa de refinamiento, 13/64
  generaciones limpias; con una capa inicializada como identidad, 43/64.
  El aporte observado es de capacidad espacial aprendida, no de reglas de figuras.
- Se pausaron tres afinaciones por el reinicio solicitado. No tenían checkpoints
  intermedios; se preservaron padres y corridas terminadas.

## 14 de noviembre de 2026 — reanudación y primer registro en Git

**Estado de esta entrada:** no hay todavía un modelo confirmado como
satisfactorio. No se abrió el conjunto final reservado de esta búsqueda.

Se retomaron las tres afinaciones desde sus padres y terminaron. También
terminaron el piloto con segunda capa de refinamiento y las semillas 17/27
de la fuente de 200 épocas. Estas últimas aún no tienen evaluación conjunta
al registrar esta entrada. Todos los números de la tabla son de la semilla 7:

| Candidato | IoU macro, 512 referencias | Limpias, 64 generaciones | Limpias, auditoría de 256 |
|---|---:|---:|---:|
| 4k, 50 épocas adicionales | 0,9329 | 59/64 | 223/256 |
| 4k, capa densa adicional | 0,9334 | 59/64 | 227/256 |
| 8k, 25 épocas adicionales | 0,9461 | 61/64 | 226/256 |
| 4k, segunda convolución de refinamiento | 0,9331 | 62/64 | 234/256 |

La segunda convolución alcanza 91,4 % en la auditoría; su intervalo de Wilson
condicional del 95 % es 87,3–94,3 %. Es un piloto seleccionado durante una búsqueda,
no una prueba de superioridad estadística ni confirmación en tres semillas.
Las coordenadas de generación son comunes; no son réplicas independientes.
La limpieza automática tampoco certifica que cada figura sea geométricamente
inequívoca. Se conservaron todos los casos, incluidos los fallos.

**Decisiones:** más datos mejora aquí la reconstrucción; la capa densa aporta
poco y la segunda convolución merece replicación. Se evaluarán primero las
semillas ya terminadas; luego se replicará la segunda convolución desde sus
padres correspondientes. Se mantienen los criterios anteriores, la revisión
visual de diversidad y el conjunto final cerrado hasta congelar la selección.

**Cambios de implementación:** datasets sintéticos, soporte RGB, galerías y GIF;
métricas de geometría/limpieza; convoluciones y gradientes manuales; transferencia
de pesos por identidad; protocolo y cachés trazables; checkpoints completos
atómicos. La arquitectura y beta cambian entre ramas, la fórmula MSE + beta·KL no.
La relación con audio es aprender distribuciones y generar representaciones;
no se comparten pesos, latentes ni una escala de MSE entre RGB y mel.

**Validación al registrar:** la batería anterior pasó 189 pruebas. Después de
agregar checkpoints y refinamiento múltiple pasaron 51 pruebas específicas,
incluida equivalencia exacta al reanudar AE/VAE con cuatro optimizadores y dos
precisiones. Antes del commit pasaron 57 pruebas específicas, incluyendo dos
pruebas adicionales de la protección de pérdidas en un clon limpio. La batería
completa de la versión actual está en ejecución y su
resultado se añadirá al terminar; no se la da por aprobada todavía.

**Evidencia versionada:** [snapshot de esta entrada](evidence/2026-11-14_pre_commit/README.md),
[resumen reproducible](progress.md), [protocolo y comandos](plan.md).
Los pesos, datasets y checkpoints pesados quedan en `results/`, ignorado por Git.
El snapshot conserva tablas, configuraciones y hashes para identificar lo medido;
incluye galerías completas de los dos pilotos destacados, sin seleccionar muestras.


### Continuación posterior al commit `08ad2ce`

El commit `08ad2ce` registra código, configuraciones y la evidencia histórica de
la entrada anterior. Después se evaluó la fuente de 200 épocas con las tres
semillas: IoU media 0,93355 y 164/192 generaciones limpias (85,4 %); no cumple
el umbral medio del 90 %. Por semilla obtuvo 55/64, 57/64 y 52/64.

Se iniciaron las réplicas 17 y 27 de `cnn4k_lowbeta_conv2_50`, cada una desde su
padre correspondiente, con 50 épocas y checkpoints cada 10. El piloto 7 se
reutiliza. `scripts/continue_art_search_refinement.sh` reproduce esta continuación:
entrena lo pendiente, evalúa las tres semillas, audita 256 muestras por semilla
y actualiza el resumen. No abre las referencias finales ni declara éxito.
Repetirlo después de una interrupción reutiliza corridas completas y checkpoints;
no lanzarlo mientras siga activo, para no duplicar procesos sobre los mismos archivos.

La batería completa terminó con **214 pruebas aprobadas en 287,82 s**. Fue
recolectada antes de añadir las dos pruebas de protección para clon limpio;
estas también pasaron dentro de la batería específica de 57 pruebas. El log y
el alcance preciso quedan en [verificación](verification.json).

La inspección del asistente de las 64 generaciones del piloto conv2 y las 64
del piloto 8k muestra triángulos y cuadriláteros reconocibles, fondos más limpios
y también contornos híbridos, esquinas redondeadas o lados cóncavos. La mejoría
de limpieza automática no elimina esos defectos. La revisión formal de al menos
cinco casos limpios inequívocos por familia y semilla sigue pendiente.

Se verificaron los hashes de los 219 archivos de evidencia y la preparación
de la búsqueda desde un archivo de Git del commit, sin resultados locales.
Ambas verificaciones pasaron.


### Revisión de la confirmación y de la presentación

Terminaron las tres semillas de conv2: IoU media 0,93374, limpieza principal
182/192 (94,8 %) y auditoría 716/768 (93,2 %). Cumple validación automática,
pero conserva formas híbridas y aún faltan revisión formal de diversidad y
prueba final reservada. No se abrieron referencias finales ni se iniciaron más
corridas en esta revisión. Se integró la presentación remota (`644e2b0`) con el
merge `83025b8`, sin conflictos ni pérdida de resultados locales.

La [revisión y propuesta](revision_2026-11-14.md) registra estado, límites y
agregados concretos al deck. La evidencia por semilla se conserva en
`evidence/2026-11-14_conv2_confirmation/`.
