# TP5 — Autoencoders, Denoising Autoencoder y VAE

Redes, retropropagación y optimizadores implementados solo con NumPy.
Cada experimento se describe en un archivo JSON; nada se configura por código.

## Uso

Desde `TP5`, con Python 3.10 o posterior:

```bash
pip install -r requirements.txt
python -m pytest                                   # tests (incluye chequeo numérico de gradientes)

python main.py configs/ej1a_baseline.json          # una corrida
python main.py configs/ej1a_*.json                 # varios experimentos
python main.py configs/ej1a_baseline.json --force  # repetir aunque ya tenga resultados
python main.py configs/ej2_vae_nsynth.json --evaluate-only   # rehacer figuras/audio sin reentrenar

python -m src.analysis.compare results/ej1a_optimizers                     # comparar configuraciones
python -m src.analysis.compare results/ej1b_denoising --metric denoising_output_mean_error@0.1

python -m src.song "results/ej2_vae_nsynth/base__seed=7"   # tema techno de 1 minuto con el VAE entrenado
python -m src.explorer "results/ej2_vae_nsynth/base__seed=7"   # explorador interactivo del latente
```

El explorador abre una página local (solo accesible desde la propia máquina) con
el mapa del espacio latente de cualquier corrida entrenada. Clic o arrastre para
decodificar un punto y, si es audio, escucharlo; "Fijar A/B" y el deslizador
recorren el latente entre dos puntos. Con más de dos dimensiones latentes el
mapa es el plano de las dos componentes principales, y lo que queda fuera del
plano se completa interpolando entre las muestras vecinas.

Para el ejercicio 2 hay que descargar una vez el split de test de NSynth (349 MB,
no se versiona): `python scripts/download_nsynth.py`.

## Archivo de configuración

Todas las claves son opcionales; `{}` es un experimento válido (autoencoder
35‑16‑8‑2‑8‑16‑35 sobre el font). Las claves desconocidas son un error, para que
un tipeo no pase inadvertido. Las rutas son relativas a `TP5`.

```jsonc
{
  "name": "mi_experimento",          // por defecto, el nombre del archivo
  "seed": 42,                        // o "seeds": [1, 2, 3] para repetir cada configuración
  "dataset": {
    "type": "font",                  // "font" | "nsynth"
    "subset": ["a", "b", "c"]        // opcional: etiquetas o índices
  },
  "model": {
    "type": "autoencoder",           // "autoencoder" | "variational"
    "latent_size": 2,
    "hidden_layers": [16, 8],        // capas ocultas del codificador
    "decoder_hidden_layers": null,   // null = espejo del codificador
    "activation": "tanh",            // identity | tanh | sigmoid | relu | leaky_relu | softplus
    "latent_activation": "identity",
    "output_activation": "sigmoid",
    "initializer": "xavier",         // xavier | he | small_normal | uniform
    "loss": "bce",                   // bce | mse (sumada sobre píxeles, promediada sobre muestras)
    "precision": "float64",          // float32 entrena ~2x más rápido en redes grandes
    "kl": {"weight": 1.0, "warmup_epochs": 0}   // solo "variational"
  },
  "training": {
    "epochs": 5000,
    "batch_size": 32,
    "log_every": 100,
    "optimizer": {"type": "adam", "learning_rate": 0.001},   // sgd | momentum | rmsprop | adam
    "noise": {"type": "flip", "level": 0.1}   // su presencia activa el modo denoising
  },
  "evaluation": {                    // cada sección presente se ejecuta
    "reconstruction": {"max_examples": 32, "audio_examples": 4},
    "latent": {},
    "latent_grid": {"size": 12, "margin": 0.1},
    "interpolation": {"pairs": [["a", "e"]], "random_pairs": 0, "steps": 9},
    "sampling": {"count": 16},
    "denoising": {"type": "flip", "levels": [0, 0.1, 0.2], "trials": 20}
  },
  "grid": {                          // producto cartesiano; rutas con puntos
    "model.hidden_layers": [[16], [32, 16]],
    "training.optimizer.type": ["sgd", "adam"]
  }
}
```

Hiperparámetros propios de cada optimizador: `momentum` (momentum), `decay` y
`epsilon` (rmsprop), `beta1`, `beta2` y `epsilon` (adam).
Tipos de ruido: `flip`, `mask`, `salt_and_pepper` (el nivel es una probabilidad)
y `gaussian` (el nivel es el desvío).

Dataset `nsynth`: `families`, `sources` (`acoustic`, `electronic`, `synthetic`),
`pitch_range` (MIDI), `max_per_family`, `selection_seed` y `spectrogram`
(`duration_seconds`, `fft_size`, `hop_length`, `mel_bands`, `dynamic_range_db`,
`griffin_lim_iterations`). La primera lectura recorre el `.tar.gz` y deja los
espectrogramas en `data/cache/`.

## Resultados

```
results/<experimento>/summary.csv        una fila por corrida (grilla × semillas)
results/<experimento>/<corrida>/
    config.json      configuración exacta de esa corrida
    model.npz        pesos
    history.csv      pérdida y métricas cada log_every épocas
    learning.png
    training.json    tiempo y pérdida final
    summary.json     métricas finales de entrenamiento y evaluación
    ...              CSV, figuras y audio (audio/) de cada evaluación
```

Las corridas terminadas no se repiten: se puede agregar valores a `grid` o
semillas y volver a correr el mismo archivo. Métricas principales para el font:
`max_pixel_errors` (la consigna pide ≤ 1), `patterns_within_target` y
`first_epoch_within_target` (con la resolución de `log_every`).

Para usar un modelo entrenado desde Python:

```python
from pathlib import Path
import numpy as np
from src.experiments.runner import load_run

model, dataset = load_run(Path("results/ej1a_baseline/base__seed=42"))
nueva_letra = model.decode(np.array([[0.5, -1.0]])) >= 0.5
```

## Organización

| Módulo | Responsabilidad |
| --- | --- |
| `src/nn/` | Activaciones, inicializadores, capas densas con backprop, pérdidas y optimizadores |
| `src/models/` | `Autoencoder`, `VariationalAutoencoder` y su construcción desde la configuración |
| `src/data/` | Font, ruido, códec audio ↔ espectrograma mel (Griffin‑Lim) y lector de NSynth |
| `src/training/` | Bucle de entrenamiento por minilotes (limpio o denoising) |
| `src/evaluation/` | Métricas y evaluaciones que generan CSV, figuras y audio |
| `src/experiments/` | Expansión de la grilla y ejecución/almacenamiento de corridas |
| `src/analysis/` | Comparación entre configuraciones de un experimento |
| `src/options.py` | Lectura estricta de la configuración (defaults y claves desconocidas) |

## Relación con la consigna

| Punto | Configuración | Qué mirar |
| --- | --- | --- |
| 1.a.1 | `ej1a_baseline` | `summary.json`, `reconstruction.png` |
| 1.a.2 | `ej1a_architectures`, `ej1a_activations`, `ej1a_optimizers`, `ej1a_latent_size` | `summary.csv` y `python -m src.analysis.compare` |
| 1.a.3 | `ej1a_baseline` | `latent.png` |
| 1.a.4 | `ej1a_baseline` | `latent_grid.png`, `interpolation.png` e `interpolation.csv` (columna `novel`) |
| 1.b | `ej1b_denoising`, `ej1b_denoising_latent_size` | `denoising.png`, `denoising_<tipo>_<nivel>.png`, `denoising.csv` |
| 2 | `ej2_ae_nsynth`, `ej2_vae_nsynth` | `latent.png`, `samples.png`, `interpolation.png`, `audio/` |

Sobre el audio: la red solo ve magnitudes mel; la fase se estima con Griffin‑Lim,
lo que degrada el sonido aun sin red. Por eso `audio/reconstruction/` trae cada
ejemplo como `__original.wav` (original pasado por el códec) y `__model.wav`:
la diferencia entre ambos es lo atribuible al modelo.
