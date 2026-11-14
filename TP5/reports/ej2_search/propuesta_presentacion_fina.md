# Propuesta detallada para completar la presentación

Esta propuesta conserva el hilo de la presentación actual: aprender una
representación, reconstruir y generar. Los nuevos resultados no convierten el
estudio en una competencia entre audio e imágenes. Son aplicaciones separadas
del mismo mecanismo variacional. Las posiciones se refieren a las 54 diapositivas
actuales, antes de agregar páginas.

Los números definitivos de los controles se registran en el informe de cierre
de esta tanda. No se edita el deck original mientras se prepara esta propuesta.

## Agregado 1: el efecto de la distribución de muestreo

**Ubicación:** después de la actual diapositiva 36, que compara seis muestras
del AE y el VAE desde N(0,I).

**Título:** «La distribución de muestreo también importa».

**Qué mostrar:** tres filas de espectrogramas y botones de audio: AE desde
N(0,I), el mismo AE desde una gaussiana ajustada a sus códigos, y VAE desde
N(0,I). Usar los tres primeros índices del nuevo control en el cuerpo principal;
conservar los seis ejemplos y los 256 casos en el respaldo. Indicar «reproducción
nueva, semilla de modelo 7» para distinguirla de los pesos originales del deck.

**Texto visible:**

- AE: aprende a reconstruir, sin imponer una distribución a sus códigos.
- AE ajustado: mismos pesos, otra distribución para elegir z.
- VAE: aprende posteriores por entrada y se regulariza hacia un prior conocido.

**Qué explica y por qué agregarlo:** la comparación original mezcla dos efectos:
cómo se entrenó el modelo y si el muestreo visita regiones compatibles con sus
códigos. Este control evita concluir que un AE es incapaz de generar. También
responde por qué ajustar una gaussiana al AE no lo convierte en VAE.

**Guion oral, aproximadamente 50 segundos:**

> Hasta acá elegimos los mismos puntos de una normal estándar para los dos
> decodificadores. Eso es natural para el VAE porque durante el entrenamiento
> penalizamos que sus posteriores se alejen de esa normal. En el AE nadie fijó
> dónde debían quedar los códigos. Por eso agregamos una alternativa: medimos
> su media y covarianza y elegimos puntos de esa distribución. No tocamos los
> pesos del AE ni lo reentrenamos para hacer este ajuste. Sigue siendo un AE:
> no aprendió una distribución por entrada ni recibió KL en su pérdida.
> El resultado del control nos permite separar el efecto del muestreo del efecto
> del entrenamiento variacional. Una gaussiana global todavía puede caer entre
> regiones poco pobladas.

**Explicación técnica para preguntas:** en los dos métodos estándar z=epsilon.
Para el AE ajustado, z=m+L·epsilon, con L·Lᵀ igual a la covarianza empírica de
sus códigos de entrenamiento. En el VAE, al entrenar una entrada x, se usa
z=mu(x)+sigma(x)·epsilon. La gaussiana global del AE y el posterior por entrada
del VAE cumplen funciones diferentes.

**Límite:** cercanía al entrenamiento y diversidad MSE no son notas de calidad
auditiva ni pruebas de novedad. Hay una sola semilla de modelo y las referencias
pertenecen al entrenamiento del protocolo original. No afirmar generalización.
Los WAV se deben escuchar antes de elegir qué reproducir en clase.

## Agregado 2: la misma pregunta con figuras RGB

**Ubicación:** después del estudio de notas omitidas, actual diapositiva 43,
y antes de las conclusiones. Introducirlo como extensión visual del ejercicio 2.

**Título:** «VAE con figuras RGB».

**Qué mostrar:** arriba, cuatro originales de validación y sus reconstrucciones;
abajo, ocho generaciones desde N(0,I), con índices prefijados. Los originales
son las referencias 0–3 y las generaciones los índices 0,8,…,56 de la semilla 7.
Los archivos trazables están en `presentation_evidence/`:

- `original_reconstruction.png`.
- `fixed_generations.png`.
- `manifest.json` y `raw_examples.npz`, para verificar procedencia y valores.

**Texto visible:**

- Cuatro familias, variaciones de tamaño, orientación y color.
- RGB de 32×32, latente de 8 dimensiones.
- Mismo mecanismo variacional, modelos y representaciones independientes.

**Por qué agregarlo:** las figuras permiten reconocer rápidamente qué conserva
la reconstrucción y qué deforma. Complementan la escucha y el espectrograma,
y muestran que generar una representación plausible no exige inferir una regla
musical ni resolver la tarea más fuerte de las notas omitidas.

**Guion oral, aproximadamente 45 segundos:**

> Para observar el mismo problema de otra manera usamos figuras simples con
> colores. El modelo recibe píxeles, no el nombre de la figura ni sus medidas.
> Arriba vemos imágenes que no recibió durante el entrenamiento y sus
> reconstrucciones. Abajo generamos sin una imagen de entrada: elegimos z de
> una normal estándar. El encoder, la reparametrización y la regularización
> siguen la misma idea que en audio. Cambian los datos y la arquitectura, y los
> modelos tienen pesos propios. Por eso comparamos el comportamiento, no el MSE
> numérico de una imagen contra el de un espectrograma.

**Para preguntas:** hay 3.072 valores RGB y un vector z de 8 coordenadas: una
reducción dimensional de 384 a 1 respecto de z. No es una tasa de compresión de
archivos: omite precisión, parámetros del posterior y almacenamiento del modelo.
El VAE RGB usa beta final 0,02; el de audio, 0,5. No son experimentos con todos
los hiperparámetros iguales ni un control que aísle la modalidad.

## Agregado 3: capacidad espacial y límites geométricos

**Ubicación:** inmediatamente después de la nueva diapositiva de figuras.

**Título:** «Limpieza de la imagen y geometría de la figura».

**Qué mostrar:** la comparación pareada `paired_examples.png`, con una y dos
convoluciones, y dos ejemplos de `clean_but_deformed.png`. Los fallos de la
semilla 27, índices 13 y 61, se eligieron expresamente para ilustrar el límite;
no se usan para estimar una frecuencia. Las tasas se calculan con todos los casos.

**Texto visible:**

- Convoluciones aprendidas para combinar posiciones vecinas.
- Misma MSE + beta·KL, sin reglas geométricas ni filtros correctivos.
- Pasar limpieza automática no garantiza una figura inequívoca.

**Por qué agregarlo:** explica la mejora sin vender una intervención externa
que dibuja figuras. También impide que el porcentaje de limpieza o una MSE baja
se interpreten como éxito total. El control iguala fuente, datos, beta, tasa,
épocas y pasos; permite estudiar el aporte de la capa adicional.

**Guion oral, aproximadamente un minuto:**

> Agregamos procesamiento espacial dentro del decodificador. Antes de producir
> RGB, una convolución combina los mapas de características de posiciones vecinas.
> La nueva capa comienza como identidad: al insertarla, el modelo hace lo mismo
> que antes. Después todos sus pesos aprenden usando la misma pérdida.
> El control recibe el mismo entrenamiento adicional, pero no esa segunda capa.
> Podemos medir si esto mejora la limpieza sin atribuirle el efecto de entrenar
> más. Aun así, estas dos salidas pasan limpieza automática y tienen contornos
> irregulares. La red no tiene una regla que le exija tres lados o cuatro ángulos.
> Por eso separamos reconstrucción, limpieza y reconocimiento visual de familias.

**Qué significa la capa:** es una convolución 3×3 de 16 canales de entrada y
16 de salida, a resolución completa antes de RGB: suma 2.320 parámetros.
Inicialmente el centro del kernel copia cada canal y los demás coeficientes
son cero. Luego se entrenan también esos coeficientes. No es postprocesado,
no recibe etiquetas de figuras y no cambia la matemática del VAE. Un VAE
tradicional también puede ser convolucional.

**Por qué siguen redondeándose algunas formas:** la MSE favorece promedios
de píxeles cuando varias salidas son compatibles con un z; la superposición
de posteriores y el compromiso con KL pueden aumentar esa incertidumbre.
La resolución 32×32 limita la precisión de las esquinas. Además, una
representación continua puede atravesar híbridos entre familias, y algunas
elipses del dataset son casi circulares. Son mecanismos compatibles con los
resultados, no causas aisladas y demostradas por esta tanda. La capa local
puede mejorar bordes sin aprender una condición global de «triángulo válido».

**Resultado que debe decirse:** pasó los umbrales automáticos de validación,
pero no la revisión visual conservadora predefinida de cinco círculos limpios
inequívocos por semilla. No se abrió la prueba final ni se declara un generador
plenamente satisfactorio. Este criterio es del estudio, no una exigencia adicional
de la consigna del ejercicio 2.

## Respaldo: dos diapositivas en el apéndice

1. **«Protocolo y controles»**: datos de entrenamiento y validación separados,
   tres semillas para RGB, una para el control nuevo de NSynth, coordenadas
   prefijadas, presupuesto pareado, beta y arquitectura por modalidad. Distinguir
   las referencias finales aún cerradas y las figuras ilustrativas elegidas.
2. **«Resultados por semilla y casos que fallan»**: tabla completa, geometría
   y limpieza separadas, revisión subjetiva identificada como del asistente,
   enlaces a las galerías completas. No incluir toda la búsqueda de candidatos
   en el cuerpo de la exposición.

## Ajustes a las diapositivas existentes

- **33–35, distribución latente:** sustituir «no quedan huecos» por «estos
  diagnósticos muestran mayor compatibilidad con el prior». Coincidir en medias,
  varianzas o una proyección 2D no establece igualdad de distribuciones en 16D.
- **36, AE frente a VAE:** explicitar «muestreo desde N(0,I)» y enlazar al nuevo
  control. No extrapolar que ningún AE pueda generar.
- **39, novedad:** título sugerido «Muestras generadas y sus vecinos en el
  dataset». La distancia no nula no descarta memorización aproximada.
- **7, 9 y 13, capacidad de las letras:** añadir «en las configuraciones
  exploradas». Los resultados con latente 1 y ocultas pequeñas no son cotas
  matemáticas de representación para cualquier red.
- **44, conclusiones:** sumar «la representación y el muestreo importan, y la
  métrica debe medir la propiedad que queremos conservar». Mantener el resultado
  negativo de notas omitidas: interpolar un timbre no demuestra descubrir alturas.

## Explicación unificadora para cerrar

> En estas aplicaciones aprendemos a reconstruir mediante una representación de
> menor dimensión. El VAE agrega incertidumbre por entrada y regularización
> hacia un prior desde el cual podemos muestrear. La representación y la
> capacidad del decodificador influyen en qué se conserva. En audio no basta
> acertar una altura si el espectro no corresponde. En imágenes no basta limpiar
> el fondo si la figura deja de ser reconocible. Los controles nos permiten
> mostrar qué aprendió cada modelo y dónde sigue fallando.

El resultado es una extensión de tres diapositivas principales y dos de respaldo,
con un guion de unos tres minutos. Conserva el alcance del TP y evita presentar
el experimento visual como una demostración independiente de arte generativo.
