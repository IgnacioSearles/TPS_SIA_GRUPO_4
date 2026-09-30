# Unidad 8 - Redes Neuronales

## 1. Introducción a las Redes Neuronales Artificiales

Una **red neuronal artificial (RNA)** es un algoritmo de aprendizaje automático modelado conceptualmente según la estructura y función del sistema nervioso biológico. Consiste en una interconexión de nodos de procesamiento llamados **neuronas artificiales**, que procesan y analizan información en conjunto.

Cada neurona:
1. Recibe uno o más estímulos o entradas ($x_i$).
2. Pondera las entradas mediante un conjunto de parámetros ajustables llamados **pesos sinápticos** ($w_i$).
3. Aplica una función de combinación (suma ponderada) y un sesgo o umbral (**bias**, $w_0$ o $\theta$).
4. Pasa el resultado por una **función de activación** para producir una salida ($O$), la cual puede transferirse a otras neuronas de la red o constituir una salida final del sistema.

### Comparación de Paradigmas de Aprendizaje

* **Aprendizaje Supervisado:** El modelo se entrena a partir de pares ordenados de datos etiquetados $\{(x^\mu, \zeta^\mu)\}_{\mu=1}^p$, donde para cada entrada $x^\mu$ se conoce la respuesta deseada $\zeta^\mu$. El objetivo es ajustar los parámetros para minimizar el error de predicción y generalizar ante nuevos datos (ej. clasificación y regresión).
* **Aprendizaje No Supervisado:** El modelo recibe únicamente entradas $\{x^\mu\}_{\mu=1}^p$ sin etiquetas. Debe hallar por sí solo estructuras, patrones, agrupaciones (*clustering*) o representaciones de menor dimensionalidad intrínsecas a la distribución de los datos.

---

## 2. El Perceptrón

### 2.1. Analogía Biológica y Modelo Computacional

El modelo del perceptrón se inspira en el funcionamiento celular de una neurona biológica:
* **Dendritas:** Vías de recepción de señales químicas/eléctricas de entrada. En el modelo computacional corresponden a las entradas $x_1, x_2, \dots, x_n$.
* **Sinapsis:** Uniones especializadas cuya fuerza o eficiencia modula la señal entrante. Corresponden a los pesos sinápticos $w_1, w_2, \dots, w_n$.
* **Soma (Cuerpo celular):** Integra los impulsos recibidos realizando una suma ponderada de estímulos.
* **Axón:** Canal por donde se transmite el potencial de acción hacia otras neuronas si el estímulo supera cierto umbral de excitación.
* **Función de activación:** Determina matemáticamente si la neurona dispara o no un estímulo y con qué intensidad:

$$\text{Estímulo neto: } h = \sum_{i=1}^n w_i x_i - u = \sum_{i=0}^n w_i x_i \quad \text{con } x_0 = 1, w_0 = -u$$

$$O = f(h)$$

Donde $u$ (o $\theta$) representa el umbral de disparo (*bias* o sesgo).

---

## 3. Hiperplano de Separación y Perceptrón Simple

### 3.1. Interpretación Geométrica: Separación Lineal

En problemas de clasificación binaria (por ejemplo, asignar una muestra a la clase $+1$ o $-1$), el perceptrón simple actúa definiendo un **hiperplano de separación** en el espacio de entradas $\mathbb{R}^n$:

$$\sum_{i=1}^n w_i x_i + w_0 = 0 \iff \mathbf{w}^T \mathbf{x} = 0 \quad (\text{con } x_0 = 1)$$

* En $\mathbb{R}^2$, este hiperplano es una recta: $w_1 x_1 + w_2 x_2 + w_0 = 0$.
* En $\mathbb{R}^3$, es un plano; en $\mathbb{R}^n$, es un hiperplano de dimensión $n-1$.

La clase predicha se determina según de qué lado del hiperplano recae la muestra:

$$O = \text{signo}(\mathbf{w}^T \mathbf{x}) = \begin{cases} +1 & \text{si } \sum_{i=0}^n w_i x_i \ge 0 \\ -1 & \text{si } \sum_{i=0}^n w_i x_i < 0 \end{cases}$$

### 3.2. Regla de Aprendizaje del Perceptrón Simple

Cuando el estímulo evaluado $x^\mu$ arroja una salida $O^\mu$ distinta a la salida esperada $\zeta^\mu \in \{-1, +1\}$, los pesos sinápticos deben corregirse en dirección al objetivo. La actualización del vector de pesos se define como:

$$\mathbf{w}(t+1) = \mathbf{w}(t) + \Delta \mathbf{w}$$

$$\Delta w_i = \eta (\zeta^\mu - O^\mu) x_i^\mu$$

Donde:
* $\eta > 0$: Tasa de aprendizaje (*learning rate*).
* $x_i^\mu$: $i$-ésima componente de la entrada del patrón $\mu$.
* $\zeta^\mu$: Salida esperada para el patrón $\mu$.
* $O^\mu$: Salida calculada por la neurona para el patrón $\mu$.

Si $\zeta^\mu = O^\mu$, $\Delta w_i = 0$ (no hay corrección).

### 3.3. Algoritmo de Entrenamiento del Perceptrón Simple

```text
1. Inicializar los pesos w = [w_0, w_1, ..., w_n] con valores aleatorios pequeños.
2. Definir una tasa de aprendizaje eta > 0 y un número máximo de épocas.
3. Repetir por cada época hasta convergencia (error total nulo o criterio de corte):
     Para cada patrón mu = 1, ..., p en el conjunto de entrenamiento:
       a. Presentar la entrada x^mu (con x_0^mu = 1).
       b. Calcular la salida de la neurona:
            O^mu = sign( sum_{i=0}^n w_i * x_i^mu )
       c. Calcular el error del patrón:
            error = (zeta^mu - O^mu)
       d. Si error != 0, actualizar los pesos:
            w_i <- w_i + eta * error * x_i^mu    para i = 0, ..., n
```

---

## 4. Aprendizaje vs. Generalización

* **Aprendizaje (Entrenamiento):** Proceso iterativo de optimización de los pesos sinápticos $\mathbf{w}$ sobre el conjunto de entrenamiento para minimizar una función de costo o error empírico.
* **Generalización:** Capacidad del modelo de clasificar o estimar correctamente las salidas para entradas nuevas que **no** formaron parte del conjunto de entrenamiento.
* **Garbage In, Garbage Out (GIGO):** Principio fundamental que establece que si los datos de entrenamiento presentan sesgos sistemáticos, etiquetas incorrectas, ruido excesivo o carecen de representatividad, el modelo aprenderá representaciones espurias y sus predicciones serán defectuosas independientemente del poder de la arquitectura.

---

## 5. Perceptrón Simple Lineal (ADALINE)

Cuando el objetivo no es clasificar clases discretas mediante un corte rígido, sino ajustar un hiperplano continuo que minimice la discrepancia con datos numéricos reales (regresión lineal), se utiliza el **Adaptive Linear Element (ADALINE)**.

### 5.1. Función de Activación y Error Cuadrático

En ADALINE, la función de activación es la **identidad**:

$$g(h) = h \implies O^\mu = \sum_{i=0}^n w_i x_i^\mu$$

La función de costo o error acumulado es la suma del error cuadrático medio:

$$E(\mathbf{w}) = \frac{1}{2} \sum_{\mu=1}^p \left( \zeta^\mu - O^\mu \right)^2 = \frac{1}{2} \sum_{\mu=1}^p \left( \zeta^\mu - \sum_{i=0}^n w_i x_i^\mu \right)^2$$

### 5.2. Deducción del Gradiente y Regla Delta

Para minimizar $E(\mathbf{w})$ por el método del gradiente descendente, calculamos la derivada parcial respecto de cada peso $w_j$:

$$\frac{\partial E}{\partial w_j} = \frac{\partial}{\partial w_j} \left[ \frac{1}{2} \sum_{\mu=1}^p \left( \zeta^\mu - \sum_{i=0}^n w_i x_i^\mu \right)^2 \right]$$

Aplicando la regla de la cadena:

$$\frac{\partial E}{\partial w_j} = \sum_{\mu=1}^p \left( \zeta^\mu - O^\mu \right) \left( -x_j^\mu \right) = - \sum_{\mu=1}^p \left( \zeta^\mu - O^\mu \right) x_j^\mu$$

La regla de actualización en sentido contrario al gradiente para un patrón dado (formato *online*):

$$\Delta w_j = - \eta \frac{\partial E^\mu}{\partial w_j} = \eta \left( \zeta^\mu - O^\mu \right) x_j^\mu$$

$$\mathbf{w}(t+1) = \mathbf{w}(t) + \Delta \mathbf{w}$$

---

## 6. Perceptrón Simple No Lineal

Cuando la relación subyacente entre los datos y la salida no es lineal, se reemplaza la función de activación por una función continua, acotada y estrictamente diferenciable.

### 6.1. Funciones de Activación Típicas y sus Derivadas

1. **Tangente Hiperbólica ($\tanh$):** Imagen en $(-1, 1)$
   $$g(h) = \tanh(\beta h)$$
   $$g'(h) = \beta \left( 1 - g(h)^2 \right)$$

2. **Logística (Sigmoide):** Imagen en $(0, 1)$
   $$g(h) = \frac{1}{1 + e^{-2\beta h}} \quad \text{o} \quad g(h) = \frac{1}{1 + e^{-\beta h}}$$
   Para $g(h) = \frac{1}{1 + e^{-\beta h}}$:
   $$g'(h) = \beta \, g(h) \left( 1 - g(h) \right)$$

*Nota sobre normalización:* Dado que la imagen de estas funciones está acotada en $(0, 1)$ o $(-1, 1)$, los valores de salida deseados $\zeta^\mu$ del conjunto de datos deben normalizarse a dichos rangos antes de entrenar la red.

### 6.2. Regla Delta No Lineal

Para $O^\mu = g(h^\mu) = g\left( \sum_{i=0}^n w_i x_i^\mu \right)$ con $E^\mu = \frac{1}{2} (\zeta^\mu - O^\mu)^2$:

$$\frac{\partial E^\mu}{\partial w_j} = \frac{\partial E^\mu}{\partial O^\mu} \frac{\partial O^\mu}{\partial h^\mu} \frac{\partial h^\mu}{\partial w_j} = - (\zeta^\mu - O^\mu) \cdot g'(h^\mu) \cdot x_j^\mu$$

Por ende, la actualización de pesos resulta:

$$\Delta w_j = \eta (\zeta^\mu - O^\mu) g'(h^\mu) x_j^\mu$$

---

## 7. Perceptrón Multicapa (MLP)

El perceptrón simple lineal y no lineal sólo puede resolver problemas linealmente separables (fracasa ante funciones como la compuerta XOR). Para problemas no lineales complejos, se acoplan capas de neuronas intermedias u ocultas.

### 7.1. Teorema de Aproximación Universal

> **Teorema (Cybenko, 1989; Hornik, 1991):** Una red neuronal *feedforward* con al menos una capa oculta, un número finito de neuronas y funciones de activación no lineales continuas y acotadas (como sigmoide, tanh o ReLU) puede aproximar cualquier función continua sobre subconjuntos compactos de $\mathbb{R}^n$ con cualquier grado de precisión deseado $\varepsilon > 0$.

*Aclaración importante:* El teorema demuestra la existencia teórica de la red, pero no garantiza que el número de neuronas sea acotado en la práctica ni provee un algoritmo garantizado para encontrar dichos pesos en tiempo polinomial.

### 7.2. Propagación Hacia Adelante (*Feed-Forward Pass*)

Considerando una red organizada en capas $m = 0, 1, \dots, M$:
* $m = 0$: Capa de entrada ($V_i^0 = x_i^\mu$).
* $m = 1, \dots, M-1$: Capas ocultas.
* $m = M$: Capa de salida.

Para cada capa $m$ desde 1 hasta $M$:

$$h_i^m = \sum_{j=0}^{N_{m-1}} w_{ij}^m V_j^{m-1}$$

$$V_i^m = g(h_i^m) \quad (\text{con } V_0^m = 1 \text{ para el término independiente})$$

---

## 8. Algoritmo de Backpropagation (Retropropagación del Error)

El algoritmo de **Backpropagation** es una implementación sistemática y computacionalmente eficiente de la **Regla de la Cadena** del cálculo multivariado para determinar las derivadas parciales de la función de costo con respecto a cada peso de la red: $\frac{\partial E}{\partial w_{ij}^m}$.

Se define el error local de la neurona $i$ en la capa $m$ como:

$$\delta_i^m \equiv - \frac{\partial E^\mu}{\partial h_i^m}$$

De este modo, por regla de la cadena:

$$\frac{\partial E^\mu}{\partial w_{ij}^m} = \frac{\partial E^\mu}{\partial h_i^m} \frac{\partial h_i^m}{\partial w_{ij}^m} = - \delta_i^m V_j^{m-1}$$

$$\Delta w_{ij}^m = - \eta \frac{\partial E^\mu}{\partial w_{ij}^m} = \eta \delta_i^m V_j^{m-1}$$

### 8.1. Paso 1: Capa de Salida ($m = M$)

Para una neurona de salida $i$, con error $E^\mu = \frac{1}{2} \sum_k (\zeta_k^\mu - V_k^M)^2$:

$$\delta_i^M = - \frac{\partial E^\mu}{\partial V_i^M} \frac{\partial V_i^M}{\partial h_i^M} = (\zeta_i^\mu - V_i^M) g'(h_i^M)$$

### 8.2. Paso 2: Capas Ocultas ($m < M$)

El error de la neurona $i$ en la capa oculta $m$ depende de la influencia de su salida sobre todas las neuronas $k$ de la capa posterior $m+1$:

$$\delta_i^m = - \frac{\partial E^\mu}{\partial h_i^m} = - \sum_{k} \frac{\partial E^\mu}{\partial h_k^{m+1}} \frac{\partial h_k^{m+1}}{\partial V_i^m} \frac{\partial V_i^m}{\partial h_i^m}$$

Dado que $\frac{\partial h_k^{m+1}}{\partial V_i^m} = w_{ki}^{m+1}$ y $- \frac{\partial E^\mu}{\partial h_k^{m+1}} = \delta_k^{m+1}$:

$$\delta_i^m = g'(h_i^m) \sum_{k=1}^{N_{m+1}} \delta_k^{m+1} w_{ki}^{m+1}$$

### 8.3. Esquema Completo del Algoritmo Backpropagation

```text
1. Inicializar todos los pesos sinápticos w_{ij}^m con valores aleatorios pequeños.
2. Definir tasa de aprendizaje eta > 0.
3. Mientras no se alcance el criterio de parada:
     Para cada muestra (x^mu, zeta^mu):
       // Fase 1: Feed-Forward
       V_j^0 = x_j^mu
       Para m = 1 hasta M:
         Para cada neurona i de la capa m:
           h_i^m = sum_j (w_{ij}^m * V_j^{m-1})
           V_i^m = g(h_i^m)
       
       // Fase 2: Backpropagation de los deltas
       Para cada neurona i en la capa de salida M:
         delta_i^M = g'(h_i^M) * (zeta_i^mu - V_i^M)
         
       Para m = M-1 descendiendo hasta 1:
         Para cada neurona i en la capa m:
           delta_i^m = g'(h_i^m) * sum_k (delta_k^{m+1} * w_{ki}^{m+1})
           
       // Fase 3: Actualización de pesos
       Para m = 1 hasta M:
         w_{ij}^m <- w_{ij}^m + eta * delta_i^m * V_j^{m-1}
```

### 8.4. Variantes de Entrenamiento según el Lote

1. **Online (Estocástico / Incremental):** Los pesos se actualizan inmediatamente después de evaluar cada muestra individual $x^\mu$. Genera oscilaciones que ayudan a escapar de óptimos locales pero presenta un cómputo menos paralelizable.
2. **Batch:** Se acumulan los $\Delta w$ a lo largo de todo el conjunto de datos de entrenamiento ($p$ muestras) y se realiza una única actualización por época:
   $$\Delta w_{ij}^m = \eta \sum_{\mu=1}^p \delta_i^{m, \mu} V_j^{m-1, \mu}$$
3. **Mini-Batch:** Los datos se fraccionan en subconjuntos de tamaño $B$ ($1 < B < p$). La actualización se ejecuta promediando el gradiente sobre cada subconjunto. Es el estándar moderno en optimización profunda por su balance entre estabilidad y aceleración en GPU.

---

## 9. Métricas de Evaluación

Para analizar el rendimiento de un modelo de clasificación supervisada, los datos se separan al menos en:
* **Training Set:** Utilizado para actualizar los pesos.
* **Testing Set:** Evaluado una vez concluido el entrenamiento para medir la capacidad de generalización.
* *(Opcional)* **Validation Set:** Utilizado durante el entrenamiento para ajustar hiperparámetros y monitorear el sobreajuste.

### 9.1. Matriz de Confusión

Organiza las predicciones frente a los valores reales:

| | Predicho Positivo | Predicho Negativo |
|---|---|---|
| **Real Positivo** | Verdadero Positivo ($TP$) | Falso Negativo ($FN$) |
| **Real Negativo** | Falso Positivo ($FP$) | Verdadero Negativo ($TN$) |

### 9.2. Métricas Derivadas

* **Exactitud (*Accuracy*):** Fracción global de aciertos:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
  *Limitación:* Es engañosa en clases fuertemente desbalanceadas.

* **Precisión (*Precision*):** Calidad de las predicciones positivas:
  $$\text{Precision} = \frac{TP}{TP + FP}$$

* **Sensibilidad / Exhaustividad (*Recall* / True Positive Rate):** Fracción de positivos reales detectados:
  $$\text{Recall} = \text{TPR} = \frac{TP}{TP + FN}$$

* **Tasa de Falsos Positivos (*False Positive Rate*):**
  $$\text{FPR} = \frac{FP}{FP + TN}$$

* **F1-Score:** Media armónica entre Precisión y Recall:
  $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$

### 9.3. Validación Cruzada ($K$-Fold Cross Validation)

Para mitigar el sesgo introducido por una partición fija de entrenamiento/prueba:
1. Se particiona el dataset aleatoriamente en $K$ subconjuntos disjuntos de tamaño similar.
2. Para cada $j \in \{1, \dots, K\}$:
   * Se toman $K-1$ bloques para entrenamiento.
   * Se utiliza el bloque $j$ como prueba (*testing*).
   * Se calcula la métrica de desempeño sobre dicho bloque.
3. Se promedian las métricas obtenidas a lo largo de las $K$ iteraciones para obtener una estimación robusta.

### 9.4. Preprocesamiento de Datos

* **Feature Scaling (Min-Max en $[a, b]$):**
  $$x' = a + \frac{x - x_{\min}}{x_{\max} - x_{\min}} (b - a)$$

* **Estandarización ($Z$-Score):**
  $$x' = \frac{x - \mu}{\sigma} \quad \text{donde } \mu = \frac{1}{n} \sum_{i=1}^n x_i, \; \sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (x_i - \mu)^2}$$

* **Normalización L2 (Unit Length Scaling):**
  $$\mathbf{x}' = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} = \frac{\mathbf{x}}{\sqrt{\sum_{i=1}^n x_i^2}}$$

---

## 10. Diagnóstico del Entrenamiento: Sesgo, Varianza y Capacidad

* **Underfitting (Subajuste):** El modelo tiene poca **capacidad** (es demasiado simple) para capturar la estructura de los datos. Se traduce en un **alto sesgo** (*high bias*) y alto error tanto en entrenamiento como en test.
* **Overfitting (Sobreajuste):** El modelo tiene una capacidad excesiva y memoriza el ruido aleatorio del conjunto de entrenamiento en lugar del patrón general. Presenta un error de entrenamiento casi nulo pero un alto error en test (**alta varianza**, *high variance*).
* **Brecha de Generalización:** Es la diferencia entre el error de validación/prueba y el error de entrenamiento:
  $$\text{Gap} = E_{\text{test}} - E_{\text{train}}$$
  Un aumento progresivo del Gap durante el entrenamiento evidencia la presencia de sobreajuste.

---

## 11. Métodos Avanzados de Optimización

El gradiente descendente básico ($\Delta \mathbf{w} = -\eta \nabla E$) utiliza información estrictamente local y sufre oscilaciones en valles estrechos y lentitud en mesetas. Se introducen variantes para acelerar la convergencia:

### 11.1. Momentum

Agrega una fracción del cambio de pesos del paso anterior, otorgando inercia al movimiento:

$$\Delta w_{ij}(t+1) = - \eta \frac{\partial E}{\partial w_{ij}} + \alpha \, \Delta w_{ij}(t) \quad \text{con } \alpha \in [0.8, 0.95]$$

Amortigua oscilaciones transversales y acelera en direcciones de descenso consistente.

### 11.2. Eta Adaptativo

Ajusta la tasa de aprendizaje $\eta$ según la evolución de la función de costo:
* Si $E(\mathbf{w})$ desciende de manera consistente durante $k$ épocas: $\eta \leftarrow \eta + a$.
* Si $E(\mathbf{w})$ se incrementa: $\eta \leftarrow \eta \cdot (1 - b)$ o $\Delta w = 0$ para descartar el paso divergente.

### 11.3. RMSProp (Root Mean Square Propagation)

Adapta la tasa de aprendizaje individualmente para cada parámetro dividiendo por la raíz del promedio móvil exponencial de los cuadrados de las derivadas:

$$v_t = \beta v_{t-1} + (1 - \beta) g_t^2$$

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{v_t} + \varepsilon} g_t$$

Donde $g_t = \nabla_w E(w_t)$ y $\varepsilon \approx 10^{-8}$ evita divisiones por cero.

### 11.4. Adam (Adaptive Moment Estimation)

Combina las propiedades de **Momentum** (primer momento: estimador de la media del gradiente) y **RMSProp** (segundo momento: estimador de la varianza no centrada):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{Primer momento})$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{Segundo momento})$$

Corrección de sesgo para compensar la inicialización en cero ($m_0 = 0, v_0 = 0$):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Regla de actualización:

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \varepsilon} \hat{m}_t$$

Valores típicos estándar: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

---

## 12. Técnicas de Regularización

La regularización agrupa técnicas orientadas a disminuir el error de generalización sin comprometer la capacidad de aprendizaje esencial del modelo.

### 12.1. Early Stopping (Parada Temprana)

Consiste en monitorear la función de error sobre el conjunto de validación al final de cada época:
1. Durante las primeras fases, tanto $E_{\text{train}}$ como $E_{\text{val}}$ descienden.
2. Al comenzar el sobreajuste, $E_{\text{train}}$ continúa bajando pero $E_{\text{val}}$ se estanca o se incrementa.
3. Se define una ventana de tolerancia llamada **paciencia** (*patience*): si $E_{\text{val}}$ no mejora durante $P$ épocas consecutivas, se detiene el entrenamiento.
4. Se restablecen los pesos de la red correspondientes a la época donde $E_{\text{val}}$ alcanzó su mínimo histórico.

### 12.2. Penalización L2 / Decaimiento de Pesos (*Weight Decay*)

Modifica la función de costo agregando un término que penaliza normas de pesos grandes:

$$E_{\text{reg}}(\mathbf{w}) = E_0(\mathbf{w}) + \frac{\lambda}{2} \sum_i w_i^2$$

Al calcular el gradiente respecto de $w_i$:

$$\frac{\partial E_{\text{reg}}}{\partial w_i} = \frac{\partial E_0}{\partial w_i} + \lambda w_i$$

$$\Delta w_i = - \eta \frac{\partial E_0}{\partial w_i} - \eta \lambda w_i$$

En cada iteración, el peso se multiplica por un factor de decaimiento $(1 - \eta \lambda) < 1$ antes de sumar el gradiente del error, forzando a la red a mantener pesos pequeños y reduciendo la sensibilidad frente a fluctuaciones irrelevantes de las entradas.

### 12.3. Aumento de Datos (*Data Augmentation*)

Técnica consistente en expandir sintéticamente la diversidad del conjunto de entrenamiento aplicando transformaciones aleatorias que preservan la etiqueta:
* Inyecciones controladas de ruido gaussiano.
* Rotaciones, traslaciones, reflexiones y deformaciones elásticas.
* Escalados y recortes aleatorios (*cropping*).

Permite que el modelo aprenda invariancias geométricas y estadísticas fundamentales de la tarea.