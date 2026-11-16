"""Control de muestreo de NSynth: N(0,I) contra gaussiana ajustada, para AE y VAE por igual.

Lee las salidas que escribe `python -m src.analysis.nsynth_sampling` en results/nsynth_sampling_control/.
La distancia es relativa a la energía de cada muestra, la misma medida que e2_lejos (ver far_sampling).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.presentacion.common import RESULTS, Array, save
from scripts.presentacion.far_sampling import FAR_PERCENTILE, RUNS, real_relative_distances, relative_distances
from src.experiments.runner import load_run

GENERATED = RESULTS / "nsynth_sampling_control"
SAMPLE_COUNT = 256
# (carpeta del método, rótulo, color): AE en rojo y VAE en azul, como en e2_distancia_muestras.
METHODS = (
    ("vae_fitted", "VAE · gaussiana ajustada", "C0"),
    ("vae_standard", "VAE · N(0, I)", "C0"),
    ("ae_fitted", "AE · gaussiana ajustada", "C3"),
    ("ae_standard", "AE · N(0, I)", "C3"),
)
REFERENCE_LABEL = "referencia: nota real\ny su vecina más parecida"


def generated_samples(method: str) -> Array:
    with np.load(GENERATED / method / "generated.npz") as stored:
        samples = stored["samples"]
    if len(samples) != SAMPLE_COUNT:
        raise ValueError(f"{method}: se esperaban {SAMPLE_COUNT} muestras y hay {len(samples)}")
    return samples


def draw_row(axis: plt.Axes, position: int, label: str, color: str, values: Array, jitter) -> None:
    filled = "ajustada" not in label
    axis.scatter(values, position + jitter.uniform(-0.18, 0.18, len(values)), s=14, alpha=0.45,
                 color=color if filled else "none", edgecolors="none" if filled else color, linewidths=0.8)
    median = np.median(values)
    axis.plot([median, median], [position - 0.3, position + 0.3], color="black", lw=2.5)
    axis.text(median, position + 0.34, f"mediana {median:.2f}".replace(".", ","), ha="center", va="bottom",
              fontsize=11)


def make_all() -> None:
    _, dataset = load_run(RUNS["vae"])
    references = dataset.samples
    reference = real_relative_distances(references)
    far = np.percentile(reference, FAR_PERCENTILE)
    rows = [(label, color, relative_distances(references, generated_samples(method)))
            for method, label, color in METHODS]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.axvline(far, color="0.3", ls="--", lw=1.2)
    jitter = np.random.default_rng(0)
    for position, (label, color, values) in enumerate(rows):
        draw_row(axis, position, label, color, values, jitter)
        beyond = int((values > far).sum())
        axis.text(1.01, position, f"{beyond}/{len(values)} lejos", transform=axis.get_yaxis_transform(),
                  va="center", fontsize=11)
        print(f"    {label}: mediana {np.median(values):.3f}, lejos {beyond}/{len(values)}")
    draw_row(axis, len(rows), REFERENCE_LABEL, "0.45", reference, jitter)
    axis.text(far, len(rows) + 0.55, f" lejos: más que el {FAR_PERCENTILE} % de las notas reales",
              fontsize=10, color="0.3", va="bottom")
    print(f"    umbral lejos: {far:.3f}")
    axis.set_yticks(range(len(rows) + 1), [label for label, _, _ in rows] + [REFERENCE_LABEL])
    axis.set_ylim(-0.5, len(rows) + 0.9)
    axis.set_xlim(left=0)
    axis.set_xlabel("diferencia con la nota real más parecida, relativa a la energía de la muestra")
    save(figure, "e2_sorteo_ajustado")
