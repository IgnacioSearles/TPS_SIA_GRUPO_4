# Confirmación del ejercicio 1

## Referencia y alcance

Los informes ya publicados por el grupo son la referencia histórica. Las corridas anteriores también se hicieron fuera de esta copia del repositorio y se usaron resultados de un compañero. Que aquí no estén sus salidas ni sus pesos no significa que no se hayan ejecutado. Esta confirmación genera evidencia adicional reproducible; no reconstruye esas corridas ni atribuye sus resultados.

El manifiesto `protocol.json` se escribe antes de entrenar: contiene las tareas, semillas, índices de submuestras, configuración base y hashes del catálogo y del código. La referencia se vuelve a entrenar con semilla 0 y queda identificada en `reference.json`. No se guardan pesos, modelos ni nombres en estos resultados. Los índices son posiciones de fila del catálogo identificado por su hash.

## 1. Finalistas de Kohonen

Se comparan 4×4 rectangular, 4×4 hexagonal y 2×8 hexagonal. Cada candidato tiene 16 neuronas y usa las mismas 30 semillas nuevas, 100–129. Son 90 entrenamientos de 1.000 épocas: tasa 0,3 → 0,01, radio 2 → 0,5 y decaimiento exponencial en las primeras 500 épocas; inicialización con muestras, vecindad gaussiana y sin parada temprana. Se evalúa la última época.

Las 27 variables se estandarizan con los 50 creadores, como en la referencia. Se registran error de cuantización (QE, distancia media al prototipo ganador), error topográfico (TE, proporción cuyo primer y segundo ganador no son vecinos), ocupación y estabilidad de las cinco regiones de Ward contiguo. ARI compara particiones sin depender de sus nombres: 1 significa agrupaciones idénticas; 0 corresponde al nivel esperado por azar bajo su ajuste.

Se comparan QE de forma pareada por semilla. TE usa la adyacencia propia de cada topología, por lo que un TE menor en hexagonal no basta para elegirla. Un radio numéricamente igual tampoco produce vecindarios equivalentes: la conclusión está limitada a estas configuraciones. No se vuelve a optimizar cada candidato después de observar los resultados.

## 2. Sensibilidad al catálogo

Se extraen 100 submuestras de 40 de los 50 creadores, sin reemplazo dentro de cada submuestra y con generador 20261022. Cada submuestra se entrena con las semillas 200, 201 y 202: 300 entrenamientos del SOM 4×4 rectangular. Se mantienen las cinco regiones y el resto de la configuración.

**Cada escalador se ajusta exclusivamente con los 40 incluidos.** Los diez excluidos se proyectan después. Se compara QE de entrenamiento contra QE de excluidos dentro de la misma corrida, y ARI contra las regiones de referencia sobre los 50 proyectados. Los QE de submuestras distintas están expresados en escalas distintas: sus resúmenes describen el protocolo; no permiten comparar distancias absolutas de catálogos distintos.

Las tres semillas de una submuestra comparten datos. Para describir la variación entre submuestras se promedian primero esas tres corridas y se resumen las 100 medias. Los intervalos p05–p95 son cuantiles descriptivos; no son intervalos de confianza de una población de creadores. Las submuestras también se superponen.

Para campañas se conservan las reglas actuales (regiones objetivo, tema principal y los tres menores costos por alcance argentino). Las etiquetas se alinean con la referencia por máxima coincidencia de miembros. Esto utiliza los 50 perfiles y solo mide estabilidad descriptiva: no es validación independiente de recomendaciones ni de conversiones. Se proyectan los 50 para comparar siempre el mismo catálogo de candidatos; no se elimina a un candidato por haber quedado fuera del entrenamiento. El solapamiento es cantidad de elegidos compartidos dividida por tres. Las listas con menos de tres candidatos se registran por separado. Una alineación por coincidencia tampoco garantiza que cada región conserve el mismo significado comercial.

## 3. Oja: aprendizaje y sensibilidad a extremos

Primero se confirman tres tasas iniciales, 0,0001; 0,001 y 0,01, con diez semillas nuevas 100–109 por tasa. Cada entrenamiento dura 1.000 épocas con tasa final inicial/100 y normalización de los pesos al terminar cada época, como en la implementación existente. Se registra el valor absoluto del coseno con PC1, la primera época en alcanzar 0,999 y 0,99999, y cualquier divergencia. Cruzar un umbral por primera vez no significa sostenerlo durante todo el entrenamiento.

Después se fija la tasa 0,001 y se quitan los uno o tres creadores con más seguidores, definidos antes de ver los resultados. Para cada eliminación hay dos controles: conservar el escalador completo y reajustarlo con los restantes. Son cuatro condiciones × diez semillas = 40 entrenamientos. El escalador completo es deliberadamente un control de sensibilidad, no un esquema de validación con datos excluidos.

Se separan dos preguntas:

1. ¿Oja reproduce PC1 del conjunto sobre el que se entrenó? Se mide su alineación con la PC1 propia de cada condición.
2. ¿PC1 cambia al retirar extremos? Se compara la dirección de PCA de cada condición contra la dirección original y el orden de sus puntuaciones sobre los mismos 50 perfiles.

Cuando se reajusta el escalador, los coeficientes se convierten a las unidades estandarizadas de referencia: `w_común = w × escala_referencia / escala_condición`, seguido de normalización. Se alinea el signo antes de comparar rankings. El origen del centrado no cambia el orden de los puntajes. Diez inicializaciones no son diez catálogos independientes: la sensibilidad de PCA depende de estas cuatro condiciones concretas. Retirar más seguidores no demuestra robustez frente a cualquier clase de atípico.

## Reproducción y salidas

Desde TP4, con las dependencias de `requirements.txt`:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m analysis.confirm_exercise1 --workers 3 --output /tmp/confirmacion-ejercicio1
OPENBLAS_NUM_THREADS=1 python3 -m analysis.summarize_exercise1_confirmation /tmp/confirmacion-ejercicio1
python3 -m unittest tests.test_confirm_exercise1
```

El directorio de salida debe ser nuevo. Hay 460 corridas planificadas más una referencia; la prueba inicial de funcionamiento se ejecuta aparte y no entra en estos resultados. `runs.jsonl` conserva métricas y etiquetas por corrida, nunca pesos. Las tablas resumen finalistas, comparaciones pareadas, submuestras, campañas, firmeza por posición de fila y Oja. Las matrices de coasignación indican la fracción de corridas en que dos creadores quedan juntos; son resultados, no matrices de pesos de la red.

## Augmentation: propuesta secundaria

No forma parte de estas 460 corridas. Con 50 perfiles tabulares, una perturbación sintética sería un ensayo de sensibilidad a imprecisión, no evidencia de nuevos creadores independientes. Si se prueba más adelante, hay que declarar variables y magnitudes, conservar composiciones que suman 100 y recalcular columnas derivadas. Los originales usados como evaluación deben quedar fuera del ajuste del escalador y de la generación de copias de entrenamiento para una evaluación independiente. Para el objetivo descriptivo de tolerancia al ruido puede usarse el catálogo actual, indicándolo. Una mejora de QE sobre datos sintéticos por sí sola no demuestra mejores recomendaciones.

En Oja, para ruido independiente de media cero e igual varianza en cada coordenada estandarizada, `Cov(x + ε) = Cov(x) + σ²I`. En esperanza se conservan los autovectores y se reduce la fracción de varianza explicada por PC1 al aumentar la varianza total. Una muestra finita puede rotar la dirección estimada. Esta deducción no cubre ruido correlacionado, restricciones composicionales ni la dinámica concreta del entrenamiento.

Referencia sobre generación tabular y conservación de correlaciones y restricciones: [Davila et al., 2024](https://arxiv.org/abs/2405.20959). La propuesta para este catálogo es una decisión experimental, no un resultado probado por ese artículo.
