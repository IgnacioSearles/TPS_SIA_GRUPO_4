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
