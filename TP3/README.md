# TP3 — Perceptrón simple y multicapa

Sistemas de Inteligencia Artificial, ITBA, 2026. Grupo 4: Ignacio Searles, Ivo
Vilamowski, Agustin Galan, Nicolas Koron y Toribio Viton Sconza.

Una librería mínima de redes neuronales escrita desde cero con NumPy (capas,
activaciones, pérdidas, optimizadores y bucle de entrenamiento) y los
experimentos de los tres ejercicios de la consigna (`enunciado.md`):

1. **Fraude:** un perceptrón aprende la probabilidad de fraude que da un modelo
   grande (destilación de conocimiento).
2. **Dígitos:** un perceptrón multicapa clasifica dígitos escritos a mano de
   28×28 píxeles.
3. **Dígitos, ≥ 98 %:** el mismo problema, sumando `more_digits.csv` y las
   técnicas necesarias para llegar al 98 % de accuracy en test.

## Instalación

Desde `TP3` (probado con Python 3.14):

```bash
cd TP3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Todos los comandos de este archivo se corren desde `TP3`.

Los CSV de la consigna no se suben al repositorio. Copiar `fraud_dataset.csv`,
`digits.csv`, `more_digits.csv` y `digits_test.csv` en `datasets/` antes de correr
cualquier experimento.

## Estructura

| Carpeta | Qué contiene |
| --- | --- |
| `nn/` | La librería: capas densas, activaciones, pérdidas, inicializadores, optimizadores y la red secuencial. |
| `training/` | El bucle de entrenamiento por mini-lotes, los callbacks y las métricas. |
| `data/` | Preprocesamiento, particiones (train/validación, K-Fold) y augmentación de imágenes. |
| `analysis/` | Utilidades de gráficos compartidas por los experimentos. |
| `datasets/` | El cargador de dígitos; acá van los CSV de la consigna. |
| `experiments/` | Un script por estudio, agrupados en `fraud/`, `digits/` y `validation/`, con sus configs JSON en `configs/`. |
| `reports/`, `results/` | Salidas de los experimentos: tablas, figuras, historiales y un `report.md` por estudio. |
| `tests/` | Tests de la librería y de los experimentos (`pytest`). |
| `presentation/` | La presentación (`index.html`, funciona sin conexión). |

Por tamaño, el repositorio no incluye los pesos de cada corrida ni las predicciones
por muestra: cualquier corrida los vuelve a generar. Sí incluye los pesos de los
modelos finales (`reports/fraud_probability/model_weights.npz`,
`results/digits_final/final_weights.npz` y `reports/digits_e3_final/entrenamiento/`),
que usan los scripts de test final, ruido e interpretabilidad.

## La librería

Cada componente se registra con un nombre y se construye desde un JSON, así que
cambiar la activación, la pérdida o el optimizador de un experimento es cambiar
una línea de su config:

```json
{
  "model": {"layers": [784, 128, 128, 10], "activation": "tanh",
            "output_activation": "softmax", "initializer": "xavier"},
  "loss": "cross_entropy",
  "optimizer": {"name": "momentum", "lr": 0.01, "momentum": 0.9},
  "training": {"epochs": 150, "batch_size": 128}
}
```

| Tipo | Nombres disponibles |
| --- | --- |
| Activación | `identity`, `step`, `sigmoid`, `tanh`, `relu`, `softmax` |
| Pérdida | `mse` (con factor ½), `cross_entropy` |
| Optimizador | `sgd`, `momentum`, `adam`; todos aceptan `weight_decay` (L2) |
| Inicializador | `uniform`, `xavier`, `he` |
| Callback | `progress`, `loss_threshold` |

Las claves que falten en un JSON toman el valor por defecto del experimento
(`experiments/config.py`).

## Cómo correr los experimentos

Cada estudio es un módulo y recibe su config como argumento. Por ejemplo:

```bash
# Validación: un perceptrón escalón aprende el AND lógico
python -m experiments.validation.and_perceptron experiments/validation/configs/and_step.json

# Ejercicio 1: entrenamiento, K-Fold y test; después, el análisis del umbral
python -m experiments.fraud.probability experiments/fraud/configs/probability.json
python -m experiments.fraud.validation_analysis

# Ejercicio 2: modelo base y estudios de η, optimizador y arquitectura
python -m experiments.digits.baseline experiments/digits/configs/baseline.json
python -m experiments.digits.learning_rates experiments/digits/configs/learning_rates_500.json
python -m experiments.digits.optimizer_study experiments/digits/configs/optimizers.json

# Ejercicio 3: entrenamiento de la arquitectura elegida y evaluación final en test
python -m experiments.digits.architecture_e3 experiments/digits/configs/e3_final.json
python -m experiments.digits.final_test_e3

# Opcionales
python -m experiments.fraud.activation_comparison         # E1: ReLU contra sigmoide
python -m experiments.digits.l2_study experiments/digits/configs/l2.json   # E2: regularización L2
python -m experiments.digits.relu_depth                    # E2: gradientes y ReLU en redes profundas
python -m experiments.digits.noise_figure                  # E3: ruido gaussiano contra accuracy
python -m experiments.digits.interpretability_maps         # E3: mapas de oclusión y pesos de la primera capa
python -m experiments.digits.ensemble --evaluate-test    # E3 extra: ensamble elegido en validación, una vez en test

# Figuras resumen de la presentación (no entrena; lee los CSV guardados)
python -m experiments.digits.presentation_figures
```

Los scripts escriben en la carpeta que indica `output.directory` de su config.
Los estudios largos guardan cada corrida terminada y, si se interrumpen, retoman
desde ahí. `experiments.digits.queue` corre una lista de estudios en orden.

`digits_test.csv` y el 20 % de test del ejercicio 1 se apartan antes de cualquier
decisión. Todas las comparaciones se hacen sobre validación; el test se consulta
solo para el resultado final y cada script lo dice en su reporte.

## Resultados principales

| Ejercicio | Resultado | Fuente |
| --- | --- | --- |
| 1 | Perceptrón sigmoide: RMSE de test 0,105 (el baseline constante da 0,302), ROC AUC 0,992. Umbral recomendado 0,888, elegido por máximo F1 en validación: en test, precisión 0,886 y recall 0,865. | `reports/fraud_probability/report.md`, `validation/threshold_summary.json` |
| 2 | 784-64-32-10, tanh + sigmoide, momentum: **85,90 %** en test. `digits.csv` no tiene ningún 8, así que el recall del 8 es 0 y el techo ronda el 90 %. | `results/digits_final/RESULTADOS_FINALES.md` |
| 3 | 784-128-128-10 con `more_digits.csv`, balanceo de clases y augmentación: **98,63 % ± 0,15** en test (3 semillas, todas ≥ 98 %). | `reports/digits_e3_final/report.md` |
| 3 | Softmax + entropía cruzada en la misma red (η 0,03, 5 semillas pareadas): gana en validación (98,78 % ± 0,04 contra 98,66 % ± 0,06) pero empata en test (98,57 % las dos; consulta adicional al test, declarada). El modelo final sigue siendo el de sigmoide. | `reports/digits_e3_softmax_tuned/report.md`, `reports/digits_e3_softmax_tuned_test/` |

En el ejercicio 3, agregar los datos que faltaban aporta unos 3 puntos y las
técnicas (balanceo y augmentación), unos 2 más
(`reports/digits_e3_datos_vs_tecnicas/report.md`).

### Opcionales

| Ejercicio | Resultado | Fuente |
| --- | --- | --- |
| 1 | Con una sola neurona, ReLU empeora la imitación: RMSE de test 0,169 contra 0,105 de la sigmoide, y 74 salidas mayores que 1. | `reports/fraud_activation_comparison/` |
| 2 | L2 con λ = 3·10⁻⁴: 96,96 % contra 96,25 % en validación, mejor en las tres semillas. En E3 no ayuda, porque la augmentación ya regulariza. | `results/digits_l2/`, `reports/digits_e3_l2/` |
| 2 | El gradiente se desvanece con sigmoide, no con tanh. Hasta 8 capas ocultas, ReLU no supera a tanh. | `results/digits_relu_depth/` |
| 3 | Con ruido gaussiano σ = 0,1 la accuracy de test se mantiene en 97,9 %; con σ = 0,2 cae a 86,0 % y con σ = 0,4, a 47,7 %. | `reports/digits_e3_noise/` |
| 3 | Mapas de oclusión por dígito y pesos de la primera capa (solo validación). | `reports/digits_e3_interpretability_maps/` |
| 3 (extra) | Promediar 12 redes ya entrenadas da **98,92 %** en test, contra 98,58 % ± 0,14 de cada red sola. El ensamble se eligió en validación entre 7 candidatos; bajar η con un coseno no mejoró. | `reports/digits_e3_ensemble/` |

## Tests

```bash
python -m pytest
```

Incluyen una verificación numérica de los gradientes de cada capa
(`tests/test_gradients.py`).

## Presentación

Abrir `presentation/index.html` en el navegador. Flechas para avanzar, `F`
para pantalla completa y `S` para las notas del orador. `presentation/README.md`
explica la estructura y `FUENTES.json`, de qué archivo sale cada figura.
