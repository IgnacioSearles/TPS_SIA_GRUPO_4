# Ejercicio 1 — Opcionales

Tres opcionales del enunciado: ReLU (práctico), construcción y descarte de features (teórico) y calibración (teórico). Los números salen de `reports/` y del dataset completo; los comandos para reproducirlos están al final.

---

## 1. ReLU como activación del perceptrón simple (práctico)

Mismo pipeline que el sigmoide: misma semilla, mismo split desarrollo/test, mismos folds, SGD con lr 0,05, batch 128, 1000 épocas. Solo cambia la activación de salida.

| | lineal | sigmoide | ReLU |
|---|---:|---:|---:|
| RMSE entrenamiento (todas las muestras) | 0,162 | 0,105 | 0,161 |
| RMSE validación (K-Fold) | — | 0,105 ± 0,001 | 0,159 ± 0,002 |
| RMSE test | — | 0,105 | 0,169 |
| predicciones de test fuera de [0, 1] | — | 0 de 1500 | 74 de 1500 (hasta 2,16) |
| ROC AUC test | — | 0,992 | 0,983 |
| umbral recomendado (máx. F1 en validación) | — | 0,888 | 0,687 |
| F1 test con ese umbral | — | 0,875 | 0,883 |

**Qué pasa.** El perceptrón aprende $z = \mathbf{w}^\top\mathbf{x} + b$ y la ReLU devuelve $\max(0, z)$. Con bias +0,42, solo el 1,8 % de las transacciones cae en la zona $z < 0$: en la práctica **la ReLU se comporta como el perceptrón lineal** con el piso recortado en 0. Por eso su curva de aprendizaje queda superpuesta a la lineal (`reports/fraud_probability_relu/learning_curves.png`).

La figura `reports/fraud_activation_comparison/target_vs_pre_activation.png` muestra por qué: la probabilidad de BigModel graficada contra $z$ ya tiene forma de S. Una sigmoide la sigue; una ReLU solo puede trazar una recta que no satura y pasa de 1 en las transacciones más obvias (montos cercanos a 2000 USD con 20+ unidades).

**Efecto en las conclusiones anteriores**

- **a) Underfitting:** se mantiene y es igual de fuerte que con el lineal (0,161 vs 0,162). La no linealidad de la ReLU está en la zona equivocada: el problema necesita saturar en 1, no cortar en 0.
- **b) Saturación de capacidades:** la ReLU también llega a una meseta, en el mismo nivel que el lineal.
- **c) Selección:** no cambia. El sigmoide sigue siendo el adecuado: menor error y salidas siempre válidas como probabilidad.
- **Generalización:** la ReLU generaliza peor. El RMSE sube de 0,159 (validación) a 0,169 (test) porque extrapola sin techo en los valores extremos. Si se recorta la salida a [0, 1] después de predecir, el RMSE baja a 0,149 en validación y 0,150 en test: la brecha era justamente la extrapolación.

**Por qué el F1 es parecido.** Con una sola neurona y una activación monótona, "salida ≥ umbral" equivale a "$z$ ≥ otro umbral". La frontera de decisión es siempre un hiperplano y la activación no cambia el orden de las transacciones, solo sus valores. Por eso ambas activaciones detectan fraude casi igual (AUC 0,983 vs 0,992), pero solo el sigmoide da **probabilidades** utilizables. También por eso los umbrales no son comparables (0,687 vs 0,888): cada modelo tiene su propia escala. Esto conecta con la calibración (sección 3).

**El problema es tener una sola neurona.** Una ReLU aporta un único quiebre, en $z = 0$. Con una neurona ese quiebre está en el lugar equivocado: el problema necesita saturar en 1, y no hay otra unidad que corte arriba. En un perceptrón multicapa, en cambio, las ReLU de la capa oculta se combinan y arman cortes donde haga falta. Por ejemplo, dos unidades bastan para acotar a [0, 1]:

$$\mathrm{ReLU}(z) - \mathrm{ReLU}(z - 1) = \min(1, \max(0, z))$$

Con más unidades ocultas, una red puede ubicar varios de estos quiebres sobre distintas features: justamente los saltos de `quantity_purchased > 9` o `items_viewed_before_purchase > 14` que la sección 2 construye a mano. Por eso ReLU es la activación habitual en capas ocultas (ejercicios 2 y 3), donde además evita el desvanecimiento del gradiente. La capa de salida, en cambio, sigue necesitando una activación acotada (sigmoide) para producir una probabilidad.

---

## 2. Features que se pueden construir y features que se pueden descartar (teórico)

La EDA mostró que el techo del perceptrón simple viene de **saltos** en la relación features → probabilidad. Un perceptrón simple solo puede aprender $\sigma(\mathbf{w}^\top\mathbf{x} + b)$: monótono en cada feature y sin umbrales internos. La forma de superar ese techo sin agrandar el modelo es darle features que ya contengan esos saltos.

Las tasas de fraude de abajo están calculadas sobre el dataset completo (7500 filas) solo para fundamentar las propuestas. En un modelo real, los umbrales de cada regla se elegirían con las filas de desarrollo, nunca con el test.

### Features para construir

| feature | definición | evidencia en el dataset | por qué ayuda |
|---|---|---|---|
| `many_units` | `quantity_purchased > 9` | 524 transacciones, **100 %** fraude (resto: 4,9 %) | convierte un salto en una entrada binaria que el perceptrón puede ponderar |
| `many_items_viewed` | `items_viewed_before_purchase > 14` | 490 transacciones, **100 %** fraude (resto: 5,4 %) | ídem |
| `new_account` | `account_age_days <= 30` | 119 transacciones, **97,5 %** fraude | la relación con la antigüedad es muy no lineal cerca de 0 |
| `round_amount_100` | `amount_usd == 100` | 156 transacciones, **98,7 %** fraude (resto: 9,7 %) | montos redondos típicos de tarjetas de regalo o pruebas de tarjeta |
| `logged_in_this_session` | `time_since_last_login_s < session_duration_seconds` | 527 transacciones, 3,0 % fraude (resto: 12,2 %) | la columna cruda no correlaciona (r = 0,002), pero combinada con la sesión sí informa |
| `log_amount` | `log(amount_usd)` | Pearson 0,565 vs 0,557 crudo | reduce la cola larga (hasta 2000 USD) y la influencia de valores extremos |
| `items_per_minute` | `items_viewed / (session_duration / 60)` | Pearson 0,31 | velocidad de navegación: sesiones muy rápidas sugieren automatización |
| `unit_price` | `amount_usd / quantity_purchased` | Pearson 0,27 (en log) | distingue muchas unidades baratas de pocas caras |

Las tres primeras reglas juntas cubren el **78,5 %** de los fraudes con **99,6 %** de precisión. Eso sugiere que BigModel aplica reglas de ese tipo, y que un TinyModel con esos indicadores como entrada podría bajar el sesgo que hoy vemos en la zona media de probabilidades.

Con features derivadas hay dos riesgos:
- **Colinealidad:** `many_units` está muy correlacionada con `quantity_purchased`. En un perceptrón no rompe nada, pero hace menos interpretables los pesos.
- **Sobreajuste a este dataset:** reglas tan limpias (100 %) pueden ser un artefacto de cómo se generaron los datos. Hay que confirmarlas en validación y monitorearlas en producción.

### Features para descartar

| feature | motivo | evidencia |
|---|---|---|
| `timestamp` crudo | no es estacionario: crece con el tiempo, así que un peso aprendido hoy extrapola mal mañana | correlación 0,001; la probabilidad media es plana por hora (0,41–0,43), día de semana (0,41–0,44) y mes |
| features de calendario (hora, día, mes) | se probaron como candidatas y no muestran señal en este dataset | tasa de fraude entre 11,0 % y 12,4 % en todos los bloques de 4 horas |
| `device_screen_resolution` | es ancho × alto: un código numérico de una categoría, no una magnitud | los 5 grupos de resolución tienen fraude entre 10,4 % y 12,9 % (media 11,6 %) |
| `time_since_last_login_s` crudo | sin relación lineal ni monótona | Pearson 0,002, Spearman 0,001; solo sirve derivada (ver arriba) |
| `flagged_fraud` | **fuga de información**: es `big_model_fraud_probability > 0,85` | coincide en el 100 % de las filas |
| `big_model_fraud_probability` | es el objetivo; como entrada sería copiar la respuesta | — |

**Nota sobre montos de 1 USD.** Los 82 montos exactos de 1,00 USD tienen 0 % de fraude y coinciden con el mínimo de la columna. Pueden ser un recorte de los datos (piso en 1). Conviene preguntarle al cliente antes de usarlos como feature.

---

## 3. Calibración (teórico)

**Qué es.** Un modelo está calibrado si, entre todas las transacciones a las que asigna probabilidad $\hat{p}$, la fracción que efectivamente es fraude es $\hat{p}$. Por ejemplo, de 100 transacciones con $\hat{p} = 0{,}3$, unas 30 son fraude. Se evalúa con:
- un **diagrama de confiabilidad**: $\hat{p}$ agrupado en bins contra la frecuencia observada;
- el **ECE** (expected calibration error): el promedio ponderado de esas diferencias;
- el **Brier score**: $\frac{1}{n}\sum(\hat{p}_i - y_i)^2$.

Calibración y capacidad de ordenar son cosas distintas: un modelo puede tener AUC 0,99 y estar descalibrado. La sección 1 lo muestra: la ReLU ordena casi igual que el sigmoide, pero sus valores no son probabilidades.

### Por qué es apropiado analizarla y ajustarla en este caso

1. **El cliente pide una probabilidad, no un ranking.** El enunciado define 0 como 0 % y 1 como 100 %. Si $\hat{p} = 0{,}7$ no significa "70 % de chances", el número no se puede comunicar ni usar en cuentas de riesgo.
2. **El umbral óptimo depende de que $\hat{p}$ esté calibrada.** Si un fraude no detectado cuesta $C_{FN}$ y una falsa alarma $C_{FP}$, conviene marcar cuando $\hat{p} \cdot C_{FN} > (1 - \hat{p}) \cdot C_{FP}$, es decir cuando $\hat{p} > C_{FP} / (C_{FP} + C_{FN})$. Si perder un fraude cuesta 10 veces más que una falsa alarma, el umbral es 1/11 ≈ 0,09. Esa regla solo vale con probabilidades calibradas. Nuestro umbral de 0,888 se eligió empíricamente (máximo F1) justamente porque no podemos confiar en la escala.
3. **TinyModel está descalibrado respecto de BigModel.** En validación y test sobreestima las probabilidades bajas (≈ +0,05) y subestima las intermedias de 0,6 a 0,8 (hasta −0,08) (`reports/fraud_probability/validation/error_by_target.png`). Ese sesgo es sistemático y monótono, que es exactamente lo que corrige una recalibración.
4. **Los umbrales no se pueden transferir entre modelos.** BigModel usa 0,85, nuestro sigmoide 0,888 y la ReLU 0,687. Si cada versión de TinyModel se recalibra a la misma escala, el negocio puede fijar un único umbral y mantenerlo cuando se reentrena el modelo.
5. **La prevalencia cambia.** El fraude es 11,6 % de este dataset, pero en producción cambia con el tiempo (drift). Un modelo calibrado hoy se descalibra si cambia la tasa base; hay que monitorear la calibración y recalibrar periódicamente, lo cual es mucho más barato que reentrenar.

### Una advertencia sobre contra qué calibrar

Solo tenemos dos referencias, y ninguna es la verdad:
- `big_model_fraud_probability`: permite calibrar **TinyModel contra BigModel** (fidelidad de la destilación).
- `flagged_fraud`: es BigModel con umbral 0,85, así que no agrega información independiente.

Para saber si las probabilidades reflejan **fraude real** hacen falta etiquetas de resultado: contracargos, investigaciones confirmadas. Si BigModel está descalibrado, TinyModel hereda ese error por más fiel que sea. Sería lo primero para pedirle a CompanyX.

### Cómo se haría

- **Datos:** un subconjunto de calibración separado del entrenamiento (por ejemplo, las predicciones out-of-fold de desarrollo). El test se sigue usando una sola vez, para evaluar.
- **Método:** una función monótona $g$ tal que $g(\hat{p})$ quede calibrada. Al ser monótona, **no cambia el ranking** (el AUC queda igual); solo corrige los valores.
  - **Platt scaling:** $g(\hat{p}) = \sigma(a \cdot z + c)$ sobre la preactivación. Con nuestro perceptrón sigmoide aporta poco, porque el modelo ya tiene esa forma.
  - **Regresión isotónica:** una función escalonada monótona sin forma fija. Es la adecuada para el sesgo en forma de S que observamos, y con miles de muestras de calibración no tiene problemas de varianza.
- **Evaluación en test:** diagrama de confiabilidad, ECE y Brier, antes y después de calibrar.

**Por qué el sesgo aparece aunque entrenamos con MSE.** Minimizar MSE contra probabilidades es minimizar un Brier score, que es una regla de puntuación propia: en teoría incentiva predicciones calibradas. El descalibre que vemos no viene de la función de pérdida sino de la **capacidad** del modelo. Un solo perceptrón no puede representar la función de BigModel (los saltos de la sección 2), y el error se reparte de forma sistemática. Por eso la calibración y las features de la sección 2 son complementarias: las features atacan la causa y la calibración corrige lo que queda.

---

## Reproducir

```
python -m experiments.fraud.probability experiments/fraud/configs/probability.json
python -m experiments.fraud.validation_analysis
python -m experiments.fraud.probability experiments/fraud/configs/relu.json
python -m experiments.fraud.validation_analysis experiments/fraud/configs/relu_validation.json
python -m experiments.fraud.activation_comparison
```
