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

## Estructura

| Carpeta | Qué contiene |
| --- | --- |
| `nn/` | La librería: capas densas, activaciones, pérdidas, inicializadores, optimizadores y la red secuencial. |
| `training/` | El bucle de entrenamiento por mini-lotes, los callbacks y las métricas. |
| `data/` | Preprocesamiento, particiones (train/validación, K-Fold) y augmentación de imágenes. |
| `datasets/` | Los CSV de la consigna y su cargador. |
| `experiments/` | Un script por estudio, agrupados en `fraud/`, `digits/` y `validation/`, con sus configs JSON en `configs/`. |
| `reports/`, `results/` | Salidas de los experimentos: tablas, figuras, pesos y un `report.md` por estudio. |
| `tests/` | Tests de la librería y de los experimentos (`pytest`). |
| `docs/` | Bitácoras y notas de diseño de algunos experimentos. |
| `presentation_v3/` | La presentación vigente (`index.html`, funciona sin conexión). `presentation/` y `presentation_v2/` son versiones anteriores. |

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

## Tests

```bash
python -m pytest
```

Incluyen una verificación numérica de los gradientes de cada capa
(`tests/test_gradients.py`).

## Presentación

Abrir `presentation_v3/index.html` en el navegador. Flechas para avanzar, `F`
para pantalla completa y `S` para las notas del orador. `presentation_v3/README.md`
explica la estructura y `FUENTES.json`, de qué archivo sale cada figura.
