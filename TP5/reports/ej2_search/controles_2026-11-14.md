# Controles completados y conclusiones para presentar

Fecha: 14 de noviembre de 2026. Se terminaron dos afinaciones visuales de
50 épocas y dos reproducciones de audio de 200 épocas. Se reutilizaron las
otras corridas terminadas; no se modificaron pesos anteriores ni pérdidas.

## Imágenes: control pareado de refinamiento

Iguales pesos padres por semilla (hashes verificados), datos, beta=0,02, tasa,
minibatches, 50 épocas adicionales y coordenadas de generación. La variante
con dos bloques agrega una convolución inicializada como identidad. Las fases
heredan pesos y reinician Adam/RNG; los checkpoints dentro de cada fase sí
restauran su estado completo.

| Semilla | Un bloque: limpias/64 | Dos bloques: limpias/64 | Un bloque: limpias/256 | Dos bloques: limpias/256 |
|---|---:|---:|---:|---:|
| 7 | 59 | 62 | 223 | 234 |
| 17 | 61 | 62 | 238 | 242 |
| 27 | 55 | 58 | 233 | 240 |

El segundo bloque pasa de 175/192 (91,1 %) a 182/192 (94,8 %): **+3,65 puntos
porcentuales**. En auditoría pasa de 694/768 (90,4 %) a 716/768 (93,2 %):
**+2,86 puntos**. La diferencia de limpieza es positiva en las tres semillas.
La IoU media pasa de 0,93359 a 0,93374: prácticamente igual, sin mejora consistente
por semilla. La mejora encontrada es de limpieza; no resuelve la geometría.

No se calculan intervalos o significación tratando todas las imágenes como
independientes: son tres modelos pareados y coordenadas comunes. Los límites
de limpieza son los mismos, calibrados antes sobre originales.

La revisión conservadora del asistente reconoce 3, 5 y 4 círculos limpios
inequívocos, contra el mínimo previo de cinco en cada semilla. Por eso el
candidato **no cumple el criterio completo del estudio**, aunque pase todos
los umbrales automáticos. No se congela la selección ni se abre el conjunto
final reservado. El juicio visual es subjetivo, de un solo revisor no ciego,
y no una tasa de acierto con etiquetas verdaderas de las generaciones.

Ver `matched_control/` y `visual_review_2026-11-14/` para tablas, índices y hashes.

## Audio: el muestreo del AE también necesita un control

Se reprodujeron AE y VAE con la configuración de NSynth de la presentación:
200 épocas, semilla 7, MSE, latente 16, capas [512,128], Adam 0,001; el VAE usa
beta 0,5 y calentamiento de 50 épocas. Son corridas nuevas porque los pesos
originales que originaron los assets no estaban en este checkout. No se
reemplazan ni se identifican como esos pesos originales.

Se usaron 256 vectores normales base comunes. La gaussiana ajustada usa solamente
los códigos de entrenamiento del AE y no modifica sus pesos. La tabla es
descriptiva sobre espectrogramas mel normalizados:

| Método | MSE al vecino de entrenamiento más próximo | Diversidad MSE entre pares |
|---|---:|---:|
| AE, normal estándar | 0.032129 | 0.051841 |
| AE, gaussiana ajustada | 0.017566 | 0.058363 |
| VAE, normal estándar | 0.017087 | 0.043278 |

Cambio relativo de distancia al ajustar el AE: **-45.3 %**.
El ajuste acerca las generaciones del AE al conjunto conocido. El contraste
AE/VAE con normal estándar no aislaba solamente la capacidad de generar:
también dependía de dónde se elegían los códigos. Esto no prueba que el AE
ajustado tenga mejor calidad auditiva ni que el VAE sea innecesario.

Datos usados para ajustar/evaluar: 1108 notas de entrenamiento. Hay una semilla de modelo y no referencias independientes para afirmar
generalización. Cercanía y diversidad no certifican calidad perceptual ni
novedad: se exportaron todos los espectrogramas y los WAV de los seis primeros
índices, con el mismo códec, para escucha. No se hizo evaluación auditiva humana
ni se afirma que el asistente escuchó los sonidos.

Una gaussiana global ajustada después del entrenamiento no transforma el AE
en VAE. En el VAE se aprenden media y varianza por entrada y KL durante el
entrenamiento; el AE ajustado conserva su codificación determinista original.

El promedio de valores mel normalizados no se interpreta como energía acústica.
Los WAV comparten normalización por pico y el mismo códec. El MSE RGB no se
compara numéricamente con el MSE mel.

## Qué se agrega a la exposición y por qué

1. **Muestreo del AE:** tres métodos, mismos vectores base, ejemplos prefijados.
   Separa entrenamiento variacional de adecuación de la distribución de muestreo.
2. **VAE con figuras RGB:** originales, reconstrucciones y generaciones reales.
   Hace visible la misma pregunta generativa que en audio, con modelos separados.
3. **Limpieza frente a geometría:** control pareado y dos contraejemplos de la
   métrica automática. Explica el aporte limitado de la capa aprendida y evita
   presentar una tasa de limpieza como validez de formas.

Los protocolos y resultados completos van al apéndice. El guion detallado,
texto visible, ubicación y explicaciones están en `propuesta_presentacion_fina.md`.
Se conserva el resultado negativo de notas omitidas y no se cambia la consigna
por nuestros criterios exploratorios más estrictos.

## Reproducibilidad

- `python -m scripts.summarize_art_controls` comprueba el pareado y exporta la tabla.
- `python -m src.analysis.nsynth_sampling` reutiliza las reproducciones completas.
- `python -m src.analysis.art_search_report` regenera el resumen de búsqueda.
- Nueve pruebas específicas del nuevo control y ajuste gaussiano pasaron.
- Los modelos fuente de audio conservan sus hashes después de evaluar.
- Los tres archivos protegidos de pérdidas siguen sin cambios.


## Actualización remota durante la tanda

Se integraron los commits `065aef5` y `16fd2b5`, con activaciones del ejercicio 2,
barras de error y correcciones de notas. La propuesta está ajustada a las 55
diapositivas actuales. El control de muestreo complementa esa búsqueda, sin
repetirla ni atribuir a activación un cambio de distribución de muestreo.
Se recomienda actualizar la comparación actual y sumar dos páginas de RGB,
con dos de respaldo; no sumar otra página redundante del mismo contraste AE/VAE.

Se inspeccionó la comparación de los seis espectrogramas prefijados por método:
el AE ajustado presenta líneas más definidas en varios ejemplos, pero conserva
casos con energía difusa. Es una observación visual limitada, no escucha subjetiva
ni clasificación exhaustiva de los 768 audios posibles.

Tras integrar los cambios remotos, pasaron **17 pruebas específicas en 23,43 s**.
La evaluación NSynth reutilizó el informe terminado y sus hashes sin reentrenar.
El alcance de la verificación queda en `verification_controls.json`.
