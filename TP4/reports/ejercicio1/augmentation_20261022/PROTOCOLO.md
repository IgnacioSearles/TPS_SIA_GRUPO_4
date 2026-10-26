# Protocolo: augmentation en Kohonen

## Pregunta

¿Pequeñas interpolaciones locales durante el entrenamiento mejoran la representación de perfiles reales excluidos, sin deteriorar topología ni estabilidad descriptiva de las campañas? Se mantiene el SOM 4×4 rectangular y cinco regiones de Ward contiguo. Se prueba una familia concreta de augmentation, no todas las formas de generación sintética ni todos los posibles radios.

## Perturbación

1. Extraer 40 de los 50 creadores, sin reemplazo; reservar diez.
2. Ajustar el escalador exclusivamente con los 40 incluidos y conservar las 27 variables actuales.
3. Para cada incluido, identificar hasta tres vecinos más cercanos del mismo tema principal, dentro de esos 40 y excluyendo al propio perfil. El tema es el mayor porcentaje temático de sus datos de entrada, no una etiqueta de éxito comercial. Si un tema tuviese un único perfil se lo deja intacto.
4. En cada época y para cada incluido, elegir un vecino y un coeficiente `α ~ Uniforme(0, intensidad)`. Con probabilidad 1/2 presentar el original; en las demás presentaciones usar `x' = (1−α)x + α vecino`.
5. Aplicar la regla habitual del SOM a la presentación elegida. La misma elección de vecino y coeficiente vale para todas las coordenadas del perfil.

La operación se hace en el espacio estandarizado. Como la transformación es afín, equivale a interpolar las 27 variables originales y después estandarizar. La mezcla convexa conserva positividad, límites y sumas de porcentajes entre extremos; cuando las sumas originales tienen redondeo, sus interpolaciones conservan ese mismo rango de redondeo. El tema principal no cambia al mezclar perfiles del mismo tema. Las columnas derivadas que excluye el preprocesador no se generan ni se usan para entrenar. No se crean nombres, identificadores ni nuevos registros del catálogo.

Ejemplo ilustrativo: entre A con 100.000 seguidores y B con 120.000, `α = 0,15` produce 103.000. Se interpola el resto del perfil con ese mismo coeficiente. El valor 15 % mide cuánto se avanza hacia B; no significa aumentar o disminuir cada variable un 15 %.

La perturbación apunta hacia vecinos y puede suavizar la distribución y sus fronteras. Su coherencia algebraica no prueba que los perfiles generados existan. Es una hipótesis experimental, no un modelo validado del error de medición de estos datos.

## Condiciones y presupuesto

- Sin augmentation: intensidad 0.
- Intensidades máximas 0,05; 0,15 y 0,30, con la misma probabilidad de presentar originales.
- 20 particiones nuevas 40/50, generador 20261023; tres semillas 400, 401 y 402 por partición.
- Cuatro condiciones × 20 particiones × tres semillas = **240 entrenamientos**, más una referencia.
- Cada entrenamiento: 1.000 épocas × 40 presentaciones = **40.000 actualizaciones**. No se aumenta el conjunto para ganar más actualizaciones.
- Misma inicialización con muestras originales y mismo orden de presentaciones entre condiciones de igual partición y semilla. El generador de perturbaciones es independiente del usado por el entrenamiento y se inicializa con `[semilla_modelo, 20261022]`.
- Mismos hiperparámetros existentes: tasa 0,3 → 0,01, radio gaussiano 2 → 0,5, decaimiento en las primeras 500 épocas, sin parada temprana. Se evalúa el estado final.
- Las perturbaciones se regeneran cada época y se mantienen activas durante todo el entrenamiento. No se ajusta su intensidad después de observar el resultado.

La prueba inicial de funcionamiento se ejecuta en un directorio temporal separado y no entra en estos resultados. El nivel cero se comprueba contra Kohonen estándar: pesos e historial deben ser idénticos. No se cambia la implementación compartida en `src/tp4`.

## Evaluación y regla de selección fijada antes de entrenar

La medida principal es el **QE sobre los diez perfiles reales excluidos**, sin perturbarlos. También se registran QE y TE de los originales de entrenamiento, TE de excluidos, ocupación, ARI entre regiones y estabilidad de los tres elegidos por campaña.

Para comparar condiciones se usa la misma partición y semilla. Se promedian primero las tres semillas y se calculan diferencias por cada una de las 20 particiones. Los percentiles 5–95 son descripciones de esas diferencias, no intervalos de confianza de una población de creadores. Las particiones se superponen y parten de un único catálogo de 50.

Una intensidad puede pasar a confirmación únicamente si cumple todos estos criterios:

1. Reduce al menos **1 %** el QE medio de excluidos frente al nivel cero.
2. Reduce el QE en al menos **70 %** de las particiones, tras promediar las tres semillas.
3. No aumenta el TE medio de excluidos en más de **0,02**.
4. No reduce el solapamiento medio de los elegidos de arte en más de **0,02**.

Son umbrales prácticos declarados en `pilot/protocol.json`, no umbrales estadísticos derivados de los resultados. Si varios niveles cumplen, se elige el de menor QE de excluidos, con menor intensidad como desempate. Si ninguno cumple, se conserva el entrenamiento sin augmentation y se informa el resultado negativo. No se relajan los criterios para obtener una mejora.

Si hay candidato, `selection.json` congela la elección y se compara contra cero en 20 nuevas particiones (generador 20261024), con semillas 500, 501 y 502: otros **120 entrenamientos**. Son otras particiones del mismo catálogo, no una validación en creadores de una población independiente. El generador de vecinos y el escalador siguen usando solamente los 40 incluidos de cada partición.

La referencia con semilla 0 se entrena con el catálogo completo solo para describir regiones y campañas. Las etiquetas se alinean por coincidencia de miembros con esa referencia. Las recomendaciones se calculan sobre los 50 originales proyectados, con las reglas comerciales actuales. Esa alineación utiliza la referencia completa y es descriptiva; no prueba eficacia comercial ni garantiza el mismo significado semántico de todas las regiones. La referencia no interviene en el escalador, vecinos, copias ni entrenamiento de las condiciones evaluadas.

## Reproducción

Desde TP4, con las dependencias de `requirements.txt`, usando un directorio de salida nuevo:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m analysis.kohonen_augmentation --output /tmp/augmentation/pilot --workers 3
OPENBLAS_NUM_THREADS=1 python3 -m analysis.summarize_kohonen_augmentation /tmp/augmentation/pilot
# Solo si selection.json habilita un candidato:
OPENBLAS_NUM_THREADS=1 python3 -m analysis.kohonen_augmentation --output /tmp/augmentation/confirmation --selection /tmp/augmentation/pilot/selection.json --workers 3
OPENBLAS_NUM_THREADS=1 python3 -m analysis.summarize_kohonen_augmentation /tmp/augmentation/confirmation
python3 -m unittest tests.test_kohonen_augmentation
```

Se conservan manifiesto de tareas, hashes, métricas por corrida y etiquetas por posición de fila. No se guardan pesos, redes ni copias sintéticas. `runs.csv` permite revisar resultados individuales, `splits.csv` contiene medias por partición, `summary.csv` y `campaigns.csv` resumen condiciones. El consolidado rechaza tareas faltantes, duplicadas, condiciones incompletas o presupuestos distintos de 40.000 actualizaciones.
