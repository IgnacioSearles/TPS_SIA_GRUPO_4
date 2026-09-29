# ITBA - Instituto Tecnológico de Buenos Aires

## Trabajo Práctico N°3: Perceptrón Simple y Multicapa
**Sistemas de Inteligencia Artificial 2026**

---

## Ejercicio (validación)

Los ejercicios a continuación sirven para validar el correcto funcionamiento de las herramientas implementadas, las cuales son:
* perceptrón simple escalón
* perceptrón simple lineal
* perceptrón simple no lineal
* perceptrón multicapa

Estos ejercicios NO se presentan pero incentivamos fuertemente a los alumnos a probarlos ya que los conjuntos de datos son simples y permitirán mayor certeza sobre la correcta implementación de los algoritmos.

### Perceptrón Simple Escalón
Función lógica "AND" con entradas:
`x = {{-1, 1}, {1,1}, {1,1 , {1, 1}}`
y salida esperada:
`y = {-1, -1, -1, 1}`

### Perceptrón Simple Lineal
Pueden tomar un conjunto de muestras (por ejemplo, 50) de una función lineal (por ejemplo, `y=x`) y tratar de que el perceptrón ajuste el conjunto de datos.

### Perceptrón Simple No Lineal
Pueden tomar un conjunto de muestras (por ejemplo, 50) de una función no lineal (por ejemplo, `y=tanh(x)`) y tratar de que el perceptrón ajuste el conjunto de datos.

### Perceptrón Multicapa
Función lógica "XOR" con entradas:
`x = {{-1, 1}, {1,1 , {1,1}, {1, 1}}`
y salida esperada:
`y = {1, 1, -1, -1}`

Se recomienda realizar los cálculos a mano para las arquitecturas:
* `[2, 2, 1]` (entrada de dos elementos, una capa oculta y una capa de salida)
* `[2, 3, 2, 1]` (entrada de dos elementos, dos capas ocultas y una capa de salida)

Pueden comparar cómo se comporta el multicapa con el perceptrón simple escalón para resolver este problema.

---

## Ejercicio 1

La empresa CompanyX se dedica a la venta de artículos online donde la prevención de fraude es una sección importante del negocio. Actualmente, está utilizando un modelo llamado BigModel para estimar la probabilidad de que una transacción sea fraudulenta. El problema es que dicho modelo es muy caro de usar para inferencia. CompanyX decide contratarlos a ustedes como parte de su equipo para desarrollar un modelo llamado TinyModel que logre la performance que tiene BigModel pero que sea mucho más barato de utilizar y almacenar. Esto es lo que se conoce como Knowledge Distillation.

Deben iterar rápidamente. Para ello deberán implementar y evaluar el perceptrón simple lineal y el perceptrón simple no lineal con una función de activación adecuada para el problema.

El objetivo es estimar la probabilidad de que una transacción online es fraudulenta, siendo 0 equivalente a 0% probabilidad de serlo y 1, 100% probabilidad de serlo. El cliente les provee con un conjunto de datos `fraud_dataset.csv`.

Primero, deberán realizar una comparación del perceptrón lineal con el no lineal con respecto al aprendizaje:
a) ¿observan underfitting?
b) ¿observan saturación de las capacidades?
c) de acuerdo al potencial de aprendizaje que tiene cada perceptrón, deben seleccionar uno para realizar un estudio de generalización, ¿cuál seleccionarían?

*Aclaración:* el estudio se realiza utilizando todas las muestras del conjunto de datos.

Una vez seleccionado uno de los perceptrones, deben proceder a realizar un estudio de generalización.
a) ¿qué métricas de evaluación seleccionaron y por qué?
b) ¿qué estrategia utilizaron para manipular el conjunto de datos durante la generalización? ¿cómo creen que se elige el mejor conjunto de entrenamiento?
c) ¿cuál es el mejor modelo que obtuvieron para presentar a su cliente? CompanyX les solicita que, sumado a tener un modelo más pequeño, realicen una recomendación del umbral de detección de fraude.

### Opcionales
* Utilizar otra función de activación en el perceptrón simple no lineal, como ReLU, ¿qué efecto tiene en las conclusiones anteriores? (práctico)
* El dataset provisto por la empresa tiene distintos features que ustedes utilizan para entrenar su modelo más pequeño, ¿qué otros features pueden construir a partir de la información que tienen, que puede ayudar al modelado del problema? ¿cuáles podrían descartar? (teórico)
* Investigar el concepto de calibración: ¿por qué sería apropiado realizar un análisis y ajuste de calibración en este caso? (teórico)

**Aclaración:** NO arrancar ejercicios adicionales antes de tener implementado los puntos necesarios para el trabajo práctico. El objetivo de los mismos es expandir sobre las ideas importantes en las cuales se centra el TP y, para quellos con ganas de más, pueden inclusive atacar estos puntos por fuera de la materia.

**Aclaración:** Es de suma importancia que exploren el conjunto de datos antes de trabajar con él: ¿qué dice la documentación de cada columna? ¿en qué rangos se mueven las distintas columnas? ¿de qué está compuesto mi dataset? ¿los datos están limpios?

---

## Ejercicio 2

CompanyX quedó muy conforme con el servicio provisto para prevención de fraude, así que han decidido hacer otro encargo. Si bien la venta de productos es online, durante la distribución de los mismos se completan muchos datos a mano. Parte de los esfuerzos de la empresa incluyen pasar esos datos a formato digital. Es por esto que CompanyX decidió invertir en desarrollar un modelo de detección automática de dígitos escritos a mano, para acelerar parte del proceso.

Deberán realizar un estudio utilizando un perceptrón multicapa para resolver el problema de clasificación de dígitos, entre el 0 y el 9. El conjunto de datos al cual la empresa les da acceso es `digits.csv` para la etapa de aprendizaje y `digits_test.csv` para estudiar la generalización. Deberán contestar:

(a) ¿cómo evalúo el desempeño de mi sistema?
(b) ¿qué variantes realizo para encontrar la solución?

Como mínimo deberán analizar:
* variantes de tasa de aprendizaje
* variantes de arquitectura
* variantes de mecanismos de optimización

los ítems representan lo básico a explorar. Es razonable que decidan variar y ver cómo funcionan otros hiperparámetros.

**Aclaración:** el conjunto de datos `digits.csv` deben utilizarlo tanto para el ajuste de parámetros como de hiperparámetros. Consideren `digits_test.csv` al equivalente a poner el modelo en producción y observar cómo se comporta con los datos del "mundo real". Lo mismo aplica para el ejercicio 3.

---

## Ejercicio 3

Los resultados obtenidos en la primera iteración con CompanyX no fueron satisfactorios. La empresa está solicitando alcanzar una accuracy mayor o igual al 98%. En el tiempo en que avanzó el desarrollo han podido recolectar nuevos datos, que les están disponibilizando en el dataset `more_digits.csv`.

(a) ¿Cuál es el mejor resultado que pudieron obtener con este nuevo conjunto de datos?
(b) ¿Qué técnicas utilizaron para mejorar el rendimiento con respecto al caso anterior?
(c) Además de sus propias técnicas, ¿existen otros factores que influyeron en el cambio de rendimiento entre este ejercicio y el anterior?

### Opcionales
Estos opcionales aplican para los ejercicios 2 y 3.
* ¿Es posible afirmar que el modelo propuesto es robusto al ruido? ¿cómo reacciona frente a distintos grados de ruido? para comprobarlo, pueden tomar la mejor solución que tengan y tomar el conjunto de datos para generalización, agregándole ruido a las distintas muestras. El ruido puede modelarse de muchas formas, por ejemplo, se puede usar ruido gaussiano, común en imágenes digitales. (práctico)
* ¿Cómo hace la red neuronal para distinguir entre los distintos dígitos? ¿qué es lo que ocurre adentro? este tipo de preguntas es característica de un área de investigación llamada "interpretabilidad". Una conjunto de técnicas que pueden explorar son los métodos de atribución. (práctico)

**Aclaración:** NO arrancar ejercicios adicionales antes de tener implementado los puntos necesarios para el trabajo práctico. El objetivo de los mismos es expandir sobre las ideas importantes en las cuales se centra el TP y, para quellos con ganas de más, pueden inclusive atacar estos puntos por fuera de la materia.

**Aclaración:** Les recomendamos que su trabajo práctico considere las siguientes funcionalidades:
* Operaciones matriciales para mejorar la performance.
* Reportar progreso cuando ponen a correr los modelos.
* Configuración extensible, que puedan almacenar para revisar parámetros.
* Métodos para guardar y levantar un modelo. Esto permite seguir entrenando un modelo que tuvieron corriendo mucho tiempo sin tener que arrancar de cero. Hacerlo en conjunto con su configuración extensible.
* División de responsabilidades: almacenar información relacionada a cada experimento (loss por época, hiperparámetros, velocidad de ejecución, etc) y realizar el análisis (gráficos, tablas, etc) por separado. Similar al enfoque que toman en simulación de sistemas.

Son recomendaciones que las pueden tener ir en cuenta a medida que avanzan y no necesariamente desde el principio. La idea es que les permita ahorrar tiempo una vez que las implementaciones son correctas, están revisadas y necesitan realizar varios experimentos.