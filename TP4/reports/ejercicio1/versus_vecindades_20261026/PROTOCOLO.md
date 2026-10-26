# Versus uniforme–gaussiana: protocolo anidado

Fecha: 26/10/2026. El usuario pidió aprovechar los experimentos recientes para comparar ambas funciones y eligió calibración equilibrada con confirmación, con una estimación previa de 40–60 minutos adicionales en esta máquina. No se modifica la presentación. El inventario y los hashes de `presentation/` se guardan antes de entrenar y se comprueban al finalizar.

## Preguntas distintas

1. **Contraste controlado:** con la misma grilla, tasa y calendario de radio, ¿qué cambia al usar una función u otra? La validación interna contiene todas las combinaciones emparejadas y permite describir este efecto dentro del rango ensayado.
2. **Contraste de procedimientos calibrados:** si ambas reciben el mismo presupuesto para elegir radio y tasa, ¿cómo representan perfiles que no participaron de esa elección? La respuesta principal usa prueba externa anidada.

El mismo número de radio no produce igual influencia en las dos funciones. La uniforme vale uno dentro del radio inclusivo y cero fuera; la gaussiana decae suavemente. Una comparación con un único radio numéricamente igual no sustituye una calibración equilibrada. Tampoco se afirma que el mejor candidato de este rango sea el mejor posible de toda la familia.

## Condiciones fijadas

- Catálogo de 50 creadores, 27 variables seleccionadas y z-score.
- Grilla rectangular 4×4, inicialización con ejemplos, vecindad uniforme o gaussiana.
- Cinco opciones de radio: constantes **0,5; 1 y 2**; exponenciales **2 → 0,5** y **2 → 1**.
- Dos tasas iniciales: **0,1 y 0,3**, ambas con tasa final 0,01 y decaimiento exponencial.
- Decaimiento durante las primeras 500 épocas y valores finales constantes durante las otras 500; 1.000 épocas, sin parada temprana.
- No se usa augmentation en este versus para aislar la comparación principal. Sus experimentos emparejados recientes quedan documentados aparte.

Son **10 candidatos por familia**, 20 en total, con el mismo esfuerzo de calibración por candidato. Los radios constantes 4 y 8 ya aparecían en el barrido histórico y no se agregan a este rango acotado; no se declara que nunca se hayan probado.

## Particiones externas

Se hacen **tres repeticiones de cinco particiones**. En cada repetición se permutan los 50 índices y se separan cinco grupos de diez para prueba. Cada bloque externo usa 40 para entrenamiento/calibración y diez para prueba.

- Hay 15 bloques externos.
- Cada creador aparece exactamente una vez como prueba en cada repetición: tres veces en total.
- El generador externo se fija en 2026102601.
- Los índices completos se escriben en `protocol.json` antes de entrenar.

## Calibración dentro de cada bloque de cuarenta

Dentro de los 40 disponibles se generan tres separaciones de **32 para entrenamiento y ocho para validación interna**. Son tres holdouts internos, no tres folds exhaustivos. Se prueban dos semillas de entrenamiento por separación, iguales para todos los candidatos de ese bloque.

**Los diez externos no participan ni en el escalador, ni en entrenamiento, ni en las métricas que eligen parámetros.** Cada escalador interno se ajusta solo con sus 32 incluidos. La estabilidad de regiones para elegir candidatos se mide sobre los 40 disponibles, sin siquiera transformar los diez externos.

Presupuesto: 15 bloques × 20 candidatos × tres holdouts × dos semillas = **1.800 entrenamientos internos**. Cada uno hace 32.000 actualizaciones. Cada familia recibe 900 entrenamientos.

## Selección predeclarada por familia y bloque

Se promedian las seis evaluaciones internas de cada candidato. Se considera apto si cumple a la vez:

- TE medio de validación ≤ **0,30**.
- Fracción media de neuronas vacías en entrenamiento ≤ **0,20**.
- ARI medio de regiones entre las dos semillas de cada holdout ≥ **0,60**.

Entre los aptos se elige **menor QE medio de validación**; se desempata por menor TE e identificador estable del candidato. Son límites prácticos de este protocolo, iguales para ambas familias; no son umbrales universales ni pruebas estadísticas. Evitan que minimizar QE sea el único objetivo y que un TE bajo por colapso del mapa se considere suficiente.

Si no hay candidato apto, se elige como fallback el menor TE, después QE, fracción vacía e identificador. Se registra explícitamente que no hubo candidato apto. Ese fallback permite medir el procedimiento pero no lo convierte en una solución que cumple los criterios. El análisis informa frecuencias de fallback y separa los bloques donde ambas familias encontraron candidatos aptos.

Las treinta selecciones —dos familias × 15 bloques— se escriben en `selections.json`, con hash de todas las calibraciones, **antes de empezar a medir pruebas externas**. El consolidado vuelve a ejecutar la regla y exige las mismas elecciones. No se cambian parámetros a partir de resultados de prueba.

## Confirmación externa

Cada candidato elegido se reentrena con los 40 disponibles y un escalador ajustado solo con esos 40. Se usan tres semillas nuevas, idénticas entre familias dentro del bloque y disjuntas de las de calibración. Se evalúan los diez externos originales, sin perturbaciones.

Presupuesto: 15 bloques × dos familias × tres semillas = **90 entrenamientos externos**, de 40.000 actualizaciones cada uno. Total del versus: **1.890 modelos**. La prueba de funcionamiento se ejecuta aparte y no entra en estas cifras.

Se registran QE/TE de entrenamiento y prueba, ocupación, asignaciones regionales y distancias/errores topográficos por creador de prueba. Cada creador recibe nueve evaluaciones por familia: tres repeticiones × tres semillas. No se guardan nombres, pesos o modelos.

La confirmación compara **procedimientos de calibración**: cada bloque puede seleccionar un radio/tasa distinto. No se declara que se evaluó una única configuración fija de cada familia ni se despliega un modelo final automáticamente.

## Resumen e interpretación

- Comparación primaria: diferencia de QE externo **uniforme − gaussiana** en cada bloque, promediando primero sus tres semillas. Negativo favorece a uniforme.
- TE, ocupación y estabilidad regional se muestran por separado: una mejora de QE no implica superioridad en todo.
- Los errores por creador se promedian sobre sus nueve exposiciones y se contrastan de forma emparejada.
- Se publican las frecuencias de elección de radios/tasas y la tabla de validación interna de las diez parejas de candidatos.
- Los percentiles 5–95 describen las 15 diferencias entre bloques; **no son intervalos de confianza**. Repeticiones y particiones comparten datos de entrenamiento y creadores.

La separación anidada protege contra elegir parámetros con los diez externos del mismo bloque. No crea una población nueva: los resultados siguen condicionados a este catálogo y protocolo. No se mide conversión comercial ni se exige que las regiones nuevas imiten las etiquetas gaussianas anteriores para seleccionar candidatos.

## Reproducción

Desde TP4 con sus dependencias, usando un directorio de salida nuevo:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m analysis.kohonen_neighborhood_versus --output /tmp/versus-nuevo --workers 6
OPENBLAS_NUM_THREADS=1 python3 -m analysis.summarize_neighborhood_versus /tmp/versus-nuevo
# Reanudar sin cambiar fuentes, datos o presentación:
OPENBLAS_NUM_THREADS=1 python3 -m analysis.kohonen_neighborhood_versus --output /tmp/versus-nuevo --workers 6 --resume
python3 -m unittest tests.test_neighborhood_versus
```

El manifiesto registra todos los candidatos, particiones, semillas y hashes antes del primer modelo. Los resultados se guardan por bloque completo para poder reanudar sin conservar pesos. La selección se congela antes de confirmar. El consolidado comprueba presupuestos, cobertura por creador, elecciones y métricas individuales.

La estimación previa se basó en los 5–6,5 segundos medios por modelo observados con seis procesos en la tanda anterior; es aproximada y depende de carga, tamaño del subconjunto y análisis posterior.
