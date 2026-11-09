# TP5 — Base del ejercicio 1

Implementación pequeña y autónoma con NumPy. Incluye un autoencoder básico y
su variante denoising, configuraciones y exportación de métricas y figuras.
Es una base ejecutable para realizar los experimentos; las configuraciones
iniciales no garantizan alcanzar la meta del TP ni constituyen un estudio terminado.

## Ejecutar

Desde `TP5`, con Python 3.10 o posterior:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
# Copiar el font.h original en data/font.h antes de entrenar.
python3 main.py --config configs/basic.json
python3 main.py --config configs/denoising.json
```

Las rutas del JSON se resuelven respecto del propio archivo de configuración.
Cada corrida requiere una salida nueva: cambiar `output` para conservar resultados
al modificar arquitectura, semilla u optimizador. Los resultados quedan en `results/`.

## Organización

| Archivo | Responsabilidad |
| --- | --- |
| `tp5/data.py` | Leer `font.h` y generar ruido por inversión independiente de píxeles |
| `tp5/model.py` | Encoder, decoder, backpropagation y persistencia de pesos |
| `tp5/training.py` | Minibatches, SGD/Adam e historial |
| `tp5/plotting.py` | Gráficos sin lógica de entrenamiento |
| `main.py` | Coordinar configuración, entrenamiento y exportación |
| `configs/` | Ejemplos editables para ambas variantes |
| `tests/` | Gradientes numéricos, aprendizaje, carga de datos y persistencia |

No hay dependencia de TP3 ni jerarquías de clases. `encode`, `decode` y
`reconstruct` permiten usar el modelo directamente desde Python.

## Decisiones iniciales

La arquitectura básica es `35 → 16 → 8 → 2 → 8 → 16 → 35`.
`hidden` define la mitad del encoder; el decoder se construye en espejo.
Las capas ocultas usan tanh; las dos coordenadas latentes son lineales y la
salida sigmoide representa valores entre cero y uno. Se minimiza entropía cruzada
binaria media con gradientes implementados a mano. La salida se binariza en 0,5.

El denoising usa inicialmente `35 → 32 → 16 → 2 → 16 → 32 → 35` para dar más
capacidad de procesamiento manteniendo el cuello de botella comparable. Es una
hipótesis para evaluar. Se regenera ruido en cada minibatch y el objetivo siempre
es el patrón limpio. Para aislar el efecto del ruido, comparar además ambas
variantes con la misma arquitectura y presupuesto de entrenamiento.

## Relación con la consigna

- **1.a.1–2:** cambiar `hidden`, `optimizer` (`sgd` o `adam`), tasa, épocas y semillas.
  `summary.json` verifica el máximo error por patrón: la meta es **todos ≤ 1 píxel**.
  `history.csv` y `learning.png` permiten estudiar el entrenamiento.
- **1.a.3:** `latent.png` y `latent.csv` muestran las dos coordenadas y los índices.
  `reconstruction.png` y `reconstruction.csv` muestran qué caracteres fallan.
- **1.a.4:** `interpolation.png` explora once puntos entre los dos primeros patrones
  seleccionados. `interpolation.npz` guarda coordenadas, muestras y la marca `novel`,
  comparada contra los 32 originales. Novedad binaria no garantiza que la salida
  parezca una letra: inspeccionar y elegir manualmente. Puede que no haya ninguna
  muestra nueva; cambiar el par desde Python o explorar otros puntos latentes.
- **1.b:** `noise.csv` compara error de entrada y salida con ruido nuevo, en los
  niveles configurados y con varias repeticiones. Exporta ejemplos de cada nivel.
  Estas pruebas usan los caracteres entrenados: miden recuperación de patrones
  conocidos con nuevas corrupciones, no generalización a letras nunca vistas.

`subset: null` usa los 32 patrones. Una lista como `[0, 1, 2, 3]` permite pruebas
cortas; no demuestra que sea imposible aprender el conjunto completo. Para el
informe, comparar varias arquitecturas, optimizadores y semillas con el conjunto
completo antes de justificar un subconjunto. Registrar resultados sin seleccionar
únicamente la mejor semilla. El estudio y las conclusiones quedan por realizar.

## Uso del modelo guardado

```python
import numpy as np
from tp5.model import Autoencoder

model = Autoencoder.load("results/basic/model.npz")
new_pattern = (model.decode(np.array([[0.2, -0.3]])) >= 0.5).reshape(7, 5)
```

Los pesos guardados sirven para inferencia; no son checkpoints para reanudar Adam.
El ejercicio 2 (VAE) queda fuera de esta base.
