# Análisis del estado experimental — 22/10/2026

> Corte inicial, anterior a los experimentos de desaprendizaje de esta conversación.
> La actualización posterior de Hopfield está en la [bitácora del ejercicio 2](ejercicio2/bitacora/README.md)
> y las [pruebas completas](ejercicio2/desaprendizaje/PRUEBAS_COMPLETAS.md).
> Sus métricas se publican de forma liviana, sin pesos ni checkpoints. Las
> observaciones originales sobre otros ejercicios se conservan como corte histórico.

El trabajo ya sostiene dos historias: en creadores, los métodos describen perfiles, pero la decisión comercial requiere criterios adicionales; en QR, la correlación entre patrones explica el fracaso de Hebb y orienta las mejoras. El siguiente paso de mayor valor es confirmar robustez y utilidad frente a referencias simples.

Este análisis revisa código, configuraciones, manifiestos, CSV agregados e informes. No ejecuta nuevos barridos. Se recalcularon la dimensión/rango de la matriz de creadores y la estabilidad sin ruido, módulos compartidos y solapamientos de los QR. Los resultados de Oja, regiones y barridos ampliados de Hopfield se citan de sus informes; sus tablas individuales no están presentes en esta copia. Distinguir estos niveles de evidencia es necesario.

## 1. Estado verificable y discrepancias

| Trabajo | Evidencia disponible al corte |
| --- | --- |
| EDA/PCA | Datos, código, tablas e informe; matriz 50×27, rango 25 verificados |
| Controles iniciales | Kohonen: 12 corridas de 60 épocas y 4 de 8; Oja: 6 de 150 y 2 de 8 |
| Pausa/reanudación | 2 corridas por algoritmo; prueba rápida de batería: 8 corridas |
| Geometría | 300/300 a 1.000 épocas, sin fallos registrados |
| Representaciones | 360/360, sin fallos registrados |
| Tasa de aprendizaje | 260/260, sin fallos registrados |
| Radio/vecindad | **500/580 únicas verificables**, sin fallos registrados; el informe y el CSV agregado declaran 580 |
| Regiones e interpretación | Informe de 5 semillas y comparación de cantidades de regiones |
| Oja sustantivo | Informe de 4 tasas × 5 semillas; 2 divergencias con la tasa alta |
| Hopfield y variantes | Código e informes; no están los CSV individuales de los barridos en esta copia |
| Otras etapas SOM | Duración sistemática, early stopping, orden, estabilidad ampliada, interacciones y confirmación: sin manifiestos de ejecución encontrados |

En vecindad hay dos directorios: uno contiene 500 corridas exitosas y el otro 150; esas 150 repiten pares configuración-semilla del primero. No son 650 observaciones independientes ni completan las 580 previstas. Las 80 sin respaldo local corresponden a radio inicial 8 y finales 0,5/1, con ambos decaimientos, vecindades y topologías. Puede faltar una copia más reciente, pero con lo disponible no corresponde afirmar que están completas.

Por etapa se verifican 300 + 360 + 260 + 500 = **1.420 resultados** en los cuatro barridos principales. Esto todavía no implica 1.420 entrenamientos independientes: entre etapas también hay controles compartidos. Las primeras 23 corridas de geometría y sus continuaciones forman parte de las 300 consolidadas.

La bitácora CSV conserva un corte del 20/10 y todavía muestra representación con 8/360. Algunos informes históricos dicen que Oja, interpretación o tasas están pendientes, aunque documentos posteriores los desarrollan. También se afirma en una entrada que se quitó el precio como criterio de semejanza, pero se conserva `price_usd`: se eliminó su duplicado en pesos, no el precio. Conviene corregir estos puntos antes de la entrega.

## 2. Creadores: qué hicieron, teoría y conclusiones

### EDA y preprocesamiento

Se revisaron 50 creadores, 33 columnas y 31 variables numéricas; no se detectaron faltantes ni duplicados. Los atípicos válidos, especialmente especialistas temáticos, se conservaron. Se quitaron precio ARS, porcentaje orgánico, engagement total y alcance argentino estimado para reducir redundancias. El último es una combinación no lineal, por lo que retirarlo sí puede cambiar la geometría; no equivale a quitar una copia lineal.

Las 27 variables se estandarizan con z-score y desvío muestral. Sin escalar, las magnitudes de volumen concentran 99,996% de la varianza y dominan la distancia. Estandarizar iguala escalas, pero no elimina el peso adicional de un bloque con muchas variables correlacionadas. Los grupos de proporciones que suman 100 explican dos dependencias exactas: rango 25.

Se encontraron dos grandes bloques opuestos: visibilidad/formato viral y profundidad del contenido. Seguidores predice precio, pero casi nada la audiencia argentina ni el costo por cuenta argentina. Esto anticipa por qué seguidores no basta para elegir una campaña local.

### PCA

PCA busca direcciones ortogonales de máxima varianza. La implementación usa SVD sobre datos centrados y estandarizados; no aplica whitening. PC1 explica **49,71%**, PC1–PC2 **58,89%**, seis componentes **81,12%** y diez **91,53%**.

PC1 contrapone volumen, tendencias y formato breve con textos, guardados, comentarios y tutoriales. PC2 mezcla arte/exhibición con geografía/perros; PC3 incorpora publicidad/fútbol y PC4 maquillaje/fútbol. Las visualizaciones 2D y 3D ayudan a interpretar, pero descartan variación importante. Máxima varianza no equivale a máxima utilidad comercial.

Se compararon 31 variables, 27 seleccionadas y logaritmos en magnitudes. Log comprime diferencias absolutas, cambiando la noción de semejanza. Retener algo más de varianza en pocas componentes no demuestra una mejor representación de campañas.

### Kohonen: teoría y métricas

Para cada muestra se encuentra la neurona ganadora, b = argmin ||x−w||, y se actualizan prototipos con `w_j ← w_j + η h(b,j)(x−w_j)`. La vecindad permite que neuronas contiguas representen perfiles cercanos; el decaimiento reduce la adaptación durante el entrenamiento. Es la idea del [trabajo original de Kohonen](https://tcosmo.github.io/assets/soms/doc/kohonen1982.pdf).

- **QE:** distancia media al prototipo ganador. Depende del espacio de entrada y no se compara directamente entre representaciones.
- **TE:** proporción cuyas dos mejores neuronas no son adyacentes. Su definición cambia con la geometría.
- **Ocupación/singletons:** distinguen resumir perfiles de individualizar creadores.
- **ARI:** compara particiones sin depender del nombre de las neuronas. Estabilidad no implica utilidad ni generalización.
- **Matriz U:** distancia de cada prototipo a sus vecinos; ayuda a leer fronteras, sin probar que existan grupos naturales separados.

### Geometría y duración

Se probaron 15 formas × 2 topologías × 2 inicializaciones × 5 semillas. El plan inicial de 500 épocas se pausó con 23 completas; se continuaron hasta 1.000 y luego se recuperó el plan completo, conservando la procedencia. El decaimiento termina en 500; las épocas 501–1.000 mantienen tasa 0,01 y radio 0,5. No equivale a distribuir el decaimiento durante 1.000.

Más neuronas baja QE, pero fragmenta: 8×8 deja alrededor de 42–45% de celdas vacías y 49–52% de creadores solos. 2×2 tiene ARI 1, pero resume demasiado. La hexagonal reduce TE en 30/30 comparaciones y suele aumentar QE; tiene hasta seis vecinos frente a cuatro, por lo que no se debe interpretar como superioridad universal.

La mejora 500→1.000 es pequeña en mapas chicos y llega a 13–16% en 8×8. 299/300 cumplen la heurística final de meseta. Eso acredita estabilización operativa, no un óptimo global. La forma también quedó confundida con el radio inicial automático: 2×8 empieza con 4 y 4×4 con 2.

El **4×4 rectangular** es una elección razonable para interpretar perfiles con este catálogo. 2×8 hexagonal y 4×6 hexagonal siguen siendo comparadores útiles, controlando explícitamente el radio.

### Representaciones

Se ejecutaron 12 representaciones × 3 tamaños × 2 topologías × 5 semillas: **360 corridas**. Incluyen 27, 31 y 24 variables, logaritmos, PCA 2/6/10/13/25 y log+PCA 6/10/13.

Las 31 variables reducen pureza temática alrededor de 0,10; quitar tres medidas de volumen no ofrece una mejora general y pierde geografía en 5×5. Log cambia el compromiso entre tema y audiencia argentina. PCA 2 pierde tema y aumenta TE. PCA 10/13 conserva gran parte de la estructura. PCA 25 reproduce las 27 variables: control de conservación de distancias al mantener todo el subespacio informativo.

**Mantener 27 variables estandarizadas es defendible**, aunque no se demostró que sea el óptimo para cualquier objetivo. El tema principal usado para evaluar se deriva de variables presentes en la entrada: es coherencia descriptiva, no validación externa de grupos verdaderos. La referencia aleatoria publicada usa ocupaciones de la representación base; un control más estricto debería respetar la ocupación de cada alternativa.

### Aprendizaje, radio y vecindad

Las 260 corridas de tasa están completas. El informe conjunto declara 840, pero localmente se respaldan **760 únicas dentro de esas dos etapas**: 260 de tasa y 500 de vecindad.

El radio final gobierna el compromiso: 0,1 baja QE pero eleva TE a aproximadamente 0,21–0,52; radios constantes grandes colapsan ocupación y dejan hasta 75% de celdas vacías. En una vecindad uniforme, radio menor que la distancia a otra neurona deja únicamente a la ganadora. En la gaussiana, radio 0,1 vuelve despreciable la influencia vecina: se aproxima a cuantización competitiva sin organización topológica.

La tasa constante 0,6 produce mayor inestabilidad; el decaimiento permite refinar. Se conserva **0,3→0,01 y radio gaussiano 2→0,5**, exponenciales, durante 500 de las 1.000 épocas. La alternativa 0,6→0,01 da las mismas regiones en la comparación informada. No se demostró equivalencia estadística general entre ambas.

### Regiones, PCA como entrada y campañas

Se agruparon los 16 prototipos mediante Ward restringido a neuronas vecinas. Se exploraron 2–8 regiones y se eligieron cinco por interpretabilidad, tamaño y estabilidad. Sus perfiles son perros/comunidad/tutoriales; nicho/reseñas patrocinadas; arte/exhibición; entretenimiento mediano; entretenimiento masivo. ARI regional entre semillas ≈ **0,79**, frente a 0,53 del 5×5.

Las regiones explican tamaño y estilo, pero audiencia argentina (η² ≈ 0,08) y costo (≈ 0,02) quedan cerca del azar. No contradice la mayor separación geográfica por neurona: agrupar 16 celdas en cinco regiones pierde detalle.

Se entrenó también el modelo con PCA 2/3/6/10. Con diez componentes, ARI con la referencia ≈ **0,91** y **12/12 selecciones** coinciden. Con dos o tres componentes las regiones son muy estables, pero pierden temas; con seis cambia la partición. Una solución estable puede estar omitiendo precisamente las variables de interés.

La propuesta comercial elige región según objetivo, filtra tema y ordena por costo por 1.000 cuentas argentinas. Frente a seguidores, las medias individuales publicadas pasan de 39,3→24,0 en fútbol, 41,9→26,4 en perros, 46,3→24,6 en maquillaje y 41,3→35,2 en arte. Esto describe el catálogo; no mide conversiones. La media de costos individuales no es el costo agregado de una cartera, y sumar alcances ignora superposición de audiencias.

La región objetivo de cada campaña está fijada manualmente en `TARGET_REGIONS`; es un juicio comercial apoyado en perfiles. El aporte adicional del SOM debe compararse con reglas simples de tema, audiencia, estilo y presupuesto.

### Oja

La actualización es `y=x·w; w←w+η y(x−yw)`. El término corrector controla el crecimiento hebbiano. Con entrada centrada y condiciones adecuadas aproxima PC1: [Oja, 1982](https://neurophysics.ucsd.edu/courses/physics_171/Oja_1982.pdf).

Se probaron cuatro tasas iniciales y cinco semillas, con decaimiento. Con 0,05 hay 2/5 divergencias; 0,01 oscila más y 0,0001 aprende más lentamente. **0,001** alcanza alineación 0,999 en 7–14 épocas y termina con error angular del orden de 10⁻¹⁰. El orden coincide con PC1.

Esto valida la implementación frente a su referencia lineal; Oja y PCA no constituyen dos evidencias independientes del mismo eje. El signo es arbitrario. El índice describe visibilidad/profundidad, no calidad universal: tiene poca relación con audiencia argentina y costo. Actualizar online requeriría conservar también un protocolo coherente de centrado y escalado.

## 3. QR: qué hicieron, teoría y conclusiones

### Red original y lector

Se almacenan cinco QR de 37×37: **N=1.369**, valores ±1, Hebb `W=ΞᵀΞ/N`, diagonal cero y actualización sincrónica. El barrido principal tiene 11 niveles p=0…0,5 × 20 semillas × 5 QR: **1.100 consultas**.

La memoria asociativa busca llevar una consulta dañada a un atractor correspondiente a un recuerdo. Para pesos simétricos, la energía `E=−sᵀWs/2` es una herramienta de análisis; la garantía habitual de no aumento exige actualización asincrónica bajo las condiciones correspondientes. No se traslada automáticamente a la actualización simultánea: [Hopfield, 1982](https://authors.library.caltech.edu/records/w41x7-8bn13).

El informe registra **0/1.100 recuperaciones exactas**: 1.051 hacia la mezcla y 49 hacia su inverso. Se verificó de nuevo que los cinco originales son inestables incluso sin ruido, con 71–225 módulos que cambian, y que terminan en la mezcla. Comparten **955/1.369 módulos (69,8%)**, con solapamientos entre **0,576 y 0,860**.

La cota aproximada 0,15N supone otro régimen de patrones; estar por debajo no garantiza estabilidad de QR correlacionados. Energía descendente tampoco implica recuperación correcta.

Se separó ubicar un QR, decodificarlo y obtener la URL correcta. Ninguna salida de la red original se decodifica. El lector solo funciona con ruido aleatorio muy bajo: el informe registra 66/100 a p=0,005 y 1/100 a p=0,02. El barrido fino evalúa al lector, no a la red.

### Controles y ortogonalidad

1. **Cinco patrones aleatorios con el mismo N:** 100% de recuperación hasta p=0,4 en el conjunto probado. Apoya que el fracaso no proviene de una incapacidad general de la implementación.
2. **Fracción compartida controlada:** transición cerca de 25% para cinco patrones en ese generador. La explicación `1/(k−1)` es una aproximación de señal/interferencia, no un umbral universal para cualquier conjunto finito o patrón correlacionado.
3. **Carga de patrones:** patrones aleatorios toleran muchas más memorias que los QR. El criterio exacto desde una consulta ruidosa es más exigente que algunas nociones de capacidad asintótica.
4. **Subconjuntos:** se enumeraron pares, ternas y grupos de cuatro; el par menú/vinos aparece en conjuntos difíciles. Menor solapamiento suele mejorar recuperación, sin determinarla por sí solo.
5. **Mapas de crosstalk y trayectorias:** localizan interferencia en los módulos distintivos y muestran estados espurios con energía baja.
6. **Resolución duplicada:** N pasa a 5.476 y no mejora: repetir información no cambia los solapamientos normalizados.

### Quitar módulos compartidos

Se entrena sobre 414 módulos variables y se reponen los otros 955. Se compararon solapamientos, estabilidad, todos los subconjuntos, ruido, lector y ejemplos, incluyendo controles que solamente reponen lo conocido.

Los cinco QR siguen sin recuperarse exactamente, pero todas las ternas quedan estables sin ruido. Con ternas, recuperación media **95,7% a p=0,25**. Reponer módulos conocidos explica parte de la mejora; los controles muestran una mejora adicional de la dinámica. Con los cinco, el lector puede rescatar algunos resultados de vinos aunque no sean exactos: decodificación y recuperación exacta son objetivos distintos.

### Varias redes y selector

Se compararon 3+2, un reparto desfavorable, 2+2+1 con QR completos y cinco redes individuales. Se enumeraron diez repartos de terna/par y quince de dos pares/un individuo.

El mejor 3+2 sin compartidos recupera **100% hasta p=0,35** en las consultas publicadas. El **2+2+1 con QR completos** es más robusto y evita retirar módulos compartidos; los quince repartos logran 100% hasta p=0,3.

La consulta pasa por todas las redes y un selector verifica pertenencia a los patrones guardados y elige por Hamming. Se evalúa un sistema de recuperación más selección, no solo una red aislada. Cinco redes individuales se aproximan a una búsqueda por plantillas en este protocolo; no debe afirmarse equivalencia para cualquier consulta/dinámica sin comprobarla.

### Pseudoinversa y daño estructurado

La regla `W=Ξᵀ(ΞΞᵀ)^+Ξ`, con diagonal anulada, reduce interferencia. Los cinco QR resultan estables y se informa **100% hasta p=0,35**, 96% a 0,4 y 59% a 0,45. Estabilidad de recuerdos no garantiza cuencas de atracción ilimitadas ni ausencia de otros atractores.

Se agregaron esquinas blancas y manchas negras. Pseudoinversa y dos redes recuperan todas las consultas publicadas con esquina hasta 17×17; ambas recuperan manchas hasta 21×21. El lector directo falla mucho antes. Son daños sobre módulos ya registrados y alineados, no fotografías con perspectiva, blur o iluminación variable.

Se incluyó la referencia de elegir el original más cercano. Bajo ruido binario independiente equiprobable con p<0,5 y prioridades iguales, minimizar Hamming es la decisión de máxima verosimilitud. No es un techo universal bajo manchas, borrados o probabilidades distintas: allí cambia el modelo de observación.

### Redes por zona

Se dividió la imagen en zonas de varios tamaños, con y sin módulos compartidos. Ningún esquema recupera exactamente los cinco QR en el barrido. Las zonas pueden converger a recuerdos diferentes y ensamblar una imagen incoherente. Con zonas 10×10 el lector rescata algunos casos de menú/vinos mediante su propia corrección. Fragmentar la imagen no resuelve la interferencia entre identidades.

## 4. Experimentos siguientes, ordenados por valor

| Prioridad | Experimento concreto | Pregunta y medición |
| --- | --- | --- |
| 1 | Reconciliar resultados y documentación; recuperar artefactos o completar las 80 de vecindad faltantes | ¿Cada cifra publicada tiene respaldo? Deduplicar por configuración, datos y semilla |
| 2 | Confirmar 2–3 candidatos SOM con semillas 100–129, radio controlado | ¿La elección sobrevive a semillas nuevas? QE/TE, ocupación, ARI regional y coincidencia de seleccionados |
| 3 | Remuestreo de creadores, por ejemplo 100 submuestras del 80% | ¿Dependen las regiones de unas pocas observaciones? Comparar sobre creadores comunes, coasignación y frecuencia de selección; reajustar escalado dentro de cada muestra |
| 4 | SOM frente a reglas simples y agrupamientos directos | ¿Qué agrega el mapa? Mismo catálogo, presupuesto y criterios; comparar perfiles, costo agregado y estabilidad, sin atribuir conversiones |
| 5 | Confirmación Hopfield con ruido nuevo, por ejemplo semillas 100–199, niveles finos 0,30–0,48 | ¿Se mantiene la ventaja? Emparejar consultas y comparar pseudoinversa, 3+2, 2+2+1 y Hamming; separar por QR e informar incertidumbre |
| 6 | QR desconocidos y consultas incompatibles, con regla de rechazo | ¿El sistema devuelve menús erróneos con confianza? Medir falso reconocimiento, rechazo y lectura de URL incorrecta |
| 7 | Actualización asincrónica contra sincrónica con los mismos pesos | ¿Qué cambia al recuperar? Energía, ciclos, exactitud y presupuesto comparable de actualizaciones; no asumir que arreglará recuerdos inestables |
| 8 | Daño por localización y por tipo: esquinas, manchas, borrados con máscara conocida | ¿Influye más el área o qué módulos se dañan? Distinguir área cubierta de módulos efectivamente alterados; Hamming restringida a módulos observados cuando la máscara sea conocida |
| 9 | Más conjuntos independientes de QR, variando URL/prefijo/corrección/máscara | ¿La explicación se mantiene fuera de estos cinco? Variar correlación y carga sin cambiar todo a la vez |
| 10 | Nuevas muestras o campañas reales y seguimiento temporal | ¿Los perfiles predicen utilidad? Nuevos creadores/meses; conversiones atribuibles si se quiere evaluar ventas o adopciones |

En SOM también quedan disponibles los barridos de early stopping, orden e interacciones. No hace falta ejecutarlos todos por inercia: una prueba factorial pequeña alrededor de finalistas puede ser más informativa que 600 interacciones amplias. El criterio de éxito de early stopping debería incluir ahorro y conservación de regiones/selecciones, además de QE. Para Oja, son útiles escalas, centrado, tasas, brecha entre los dos primeros autovalores y datos secuenciales con referencia PCA coherente.

Los porcentajes perfectos observados requieren contexto: 100 éxitos en 100 consultas no garantizan éxito universal. Las semillas son repeticiones del mecanismo de ruido/entrenamiento sobre los mismos objetos. No sustituyen nuevos creadores ni nuevos conjuntos de QR. Conviene reservar semillas y objetos de confirmación antes de seguir eligiendo configuraciones.

La prioridad práctica es **cerrar trazabilidad, confirmar finalistas y medir el valor añadido frente a alternativas simples**. Ya hay bastante exploración; falta establecer cuánto de sus conclusiones sobrevive fuera de los casos utilizados para elegirlas.
