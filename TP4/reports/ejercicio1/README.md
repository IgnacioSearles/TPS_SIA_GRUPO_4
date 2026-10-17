# Ejercicio 1: EDA y PCA inicial

## Reproducción

Desde TP4: `python3 analysis/eda_pca.py`. Requiere las dependencias de `requirements.txt` y `data/creadores.csv`. El PCA usa la implementación compartida de `src.tp4.pca`, mediante SVD de NumPy, sin scikit-learn. Las validaciones, representaciones, tablas y gráficos están separados en módulos de `analysis`. Genera tablas y gráficos en esta carpeta. Las pruebas verifican ortogonalidad, reconstrucción completa, autovalores y coherencia con el preprocesamiento. Esta etapa no implementa todavía Kohonen ni Oja.

## Calidad y exploración

Hay **50 creadores, 33 columnas y 31 variables numéricas**. No hay faltantes, infinitos, filas duplicadas, identificadores repetidos ni nombres repetidos. Los porcentajes están entre 0 y 100; temas, propósitos del contenido y publicidad suman 100 por fila. No es necesario imputar.

Las escalas son muy diferentes: seguidores van de 9.604 a 775.589; precios de USD 75 a 6.334,71; audiencia argentina de 19,55% a 94,43%. Por eso centrar sin escalar daría mayor influencia a las unidades grandes. **Se aplica z-score con desvío muestral a cada variable seleccionada.** Los parámetros quedan en `escalado.csv` para transformar creadores nuevos con la misma referencia.

El criterio de 1,5 IQR señala 11 observaciones en maquillaje y 11 en arte. No se eliminan: son porcentajes válidos y la especialización es justamente información útil para los clientes. También señala cinco alcances argentinos estimados, tres engagements totales y un valor en likes, comentarios y compartidos. Son señales para inspección, no evidencia de errores. No se recortan ni se borran creadores. Ver `atipicos.csv`, `descriptivos.csv` y `distribuciones.png`.

## Selección de variables

Se excluyen `creator_id` y `name` de la matriz numérica; permanecen para identificar resultados. De las 31 variables numéricas se conservan **27**:

| Variable excluida | Motivo | Información conservada |
| --- | --- | --- |
| `price_ars` | Es exactamente 1.500 veces el precio USD, salvo precisión flotante | `price_usd` |
| `organic_posts_pct` | Es 100 menos publicidad | `sponsored_posts_pct` |
| `total_engagement_per_1000` | Es la suma de las cuatro interacciones | Likes, comentarios, guardados y compartidos por separado |
| `estimated_argentina_reach` | Es alcance × porcentaje argentino, redondeado; error máximo 0,499 cuentas | Alcance y porcentaje argentino |

Esto evita dar peso adicional a información calculada. El alcance argentino es una relación **no lineal**: eliminarlo puede cambiar la geometría lineal del PCA. La decisión es preservar sus dos factores para distinguir volumen de pertinencia geográfica; se conserva en el CSV original para evaluar campañas.

No se descartan automáticamente seguidores, alcance, impresiones o reproducciones por su alta correlación: miden aspectos diferentes del caso de uso y la consigna pide agrupar considerando todas las variables. Queda el riesgo de dar más peso al bloque de escala/visibilidad por tener varias medidas similares. Comparar una representación con una medida por bloque o pesos por bloque es una alternativa pendiente para Kohonen; el PCA por sí solo no corrige ese peso semántico.

Se conservan los cuatro temas y los cuatro propósitos. Cada grupo suma 100, de modo que introduce una dependencia lineal exacta después del centrado: la matriz de 27 columnas tiene rango 25. SVD lo maneja y aparecen dos componentes de varianza nula. Conservar todos mantiene simetría entre categorías; eliminar una categoría por grupo es posible pero modifica las distancias al elegir una referencia. Una transformación composicional ILR es otra opción si interesa interpretar proporciones relativas; requiere una geometría distinta y revisar el tratamiento de ceros si llegan datos nuevos. No se aplica en esta primera versión.

## PCA y reducción

| Representación | PC1 | PC2 | PC1 + PC2 | Componentes ≥80% | ≥90% | ≥95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 variables, z-score | 47,97% | 9,76% | 57,72% | 6 | 10 | 13 |
| **27 seleccionadas, z-score** | **49,71%** | **9,19%** | **58,89%** | **6** | **10** | **13** |
| 27 seleccionadas, log1p en magnitudes y z-score | 52,40% | 8,57% | 60,97% | 6 | 9 | 12 |

En la variante log se transforman seguidores, alcance, impresiones, reproducciones y precio USD. Comprime diferencias absolutas entre creadores grandes y destaca diferencias relativas. **La base inicial elegida es sin log**: mantiene la interpretación de las magnitudes originales y el aumento de varianza retenida por las dos primeras componentes con log es modesto. Ese porcentaje no mide calidad comercial ni demuestra que una transformación sea superior. Log sigue siendo una alternativa razonable si la agrupación buscada prioriza diferencias relativas.

**Decisión:** usar PC1–PC2 como visualización, no como representación suficiente para entrenar la red. Retienen 58,89%, perdiendo 41,11%. Para Kohonen se propone una base de comparación con las 27 variables estandarizadas y otra con **10 componentes sin whitening**, que retienen **91,53%**. Se elige el umbral del 90% como compromiso explícito, no como regla universal. Las alternativas son 6 componentes (81,12%), 13 (95,63%) o conservar todas las dimensiones informativas. La decisión final requiere medir error de cuantización, error topográfico y estabilidad del SOM; aún no fueron evaluados. Whitening igualaría los pesos de componentes con distinta varianza y puede amplificar dimensiones débiles; no se utiliza.

## Interpretación

Los números siguientes son **correlaciones variable-componente**, no coeficientes del autovector. Están en `cargas_correlacion.csv`; los coeficientes para proyectar están en `coeficientes.csv`. El signo de una componente es arbitrario: el script fija positivo su coeficiente de mayor módulo.

- **PC1 (49,71%)**: hacia valores negativos aparecen impresiones (−0,918), reproducciones (−0,913), alcance (−0,909), tendencias (−0,900), seguidores (−0,869) y precio (−0,869). Hacia valores positivos: longitud del texto (+0,937), guardados (+0,841), visitas al perfil (+0,773), comentarios (+0,767) y tutoriales (+0,729). Sugiere un contraste entre visibilidad/formato breve y contenido más elaborado con interacciones de profundidad. Son asociaciones del catálogo; no prueban causalidad ni ventas.
- **PC2 (9,19%)**: predominan exhibición de productos/resultados (+0,659) y arte (+0,632), en oposición a audiencia argentina (−0,568) y perros (−0,411). Mezcla contenido y geografía: no corresponde etiquetarla como una dimensión pura de calidad o de temática.
- **PC3 (7,00%)**: publicidad (+0,731) y fútbol (+0,659), frente a perros (−0,462). Agrega diferenciación comercial y temática ausente del plano inicial.
- **PC4 (5,83%)**: maquillaje (+0,640) frente a fútbol (−0,586). Refuerza que PC1–PC2 no resumen todas las diferencias relevantes para los clientes.

![Varianza explicada](varianza.png)

![Biplot](biplot.png)

En el biplot los puntos son scores divididos por el desvío de cada componente, y las flechas son correlaciones multiplicadas por dos para legibilidad. Es un biplot de correlación con escala visual explícita: no interpretar la distancia punto-flecha como cantidad original ni usar distancias del dibujo como distancias de entrenamiento. El color indica el tema con porcentaje máximo, solo como ayuda descriptiva; no son clusters descubiertos. Las flechas se proyectan sobre dos ejes que representan únicamente el 58,89% de la varianza.

## Implicancias para las campañas y límites

Para refugios conviene explorar afinidad con perros, audiencia argentina y visitas al perfil/comentarios, junto al costo. Para fútbol hay perfiles con alcance y descubrimiento y otros más educativos: depende del objetivo y presupuesto; PC3 y PC4 aportan información temática. Para belleza y materiales de arte se debe revisar afinidad temática y reseñas/tutoriales/exhibición, además de volumen y precio. **No elegir automáticamente por PC1 ni por seguidores.** Tener más guardados o visitas tampoco demuestra conversiones: el catálogo no contiene ventas, consultas ni adopciones atribuibles.

No se seleccionan ganadores de campaña en esta etapa. El próximo paso es entrenar y comparar SOM con estas representaciones, contrastar las regiones con las componentes e interpretar perfiles y costos. Con solo 50 registros y 27 variables, conviene evaluar estabilidad frente a semillas y cambios de muestra antes de dar por firmes los grupos. Los ejes dependen del catálogo: una actualización mensual puede rotarlos; para comparar meses hay que conservar el preprocesamiento/modelo de referencia o documentar el reajuste.

Los CSV, JSON y PNG generados contienen información derivada de los datos y se excluyen de Git junto con el CSV original. Este informe y el script sí pueden versionarse.
