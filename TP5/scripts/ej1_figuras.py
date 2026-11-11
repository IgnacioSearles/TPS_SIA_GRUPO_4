"""Figuras y tablas para la presentación de las pruebas extra del ejercicio 1.

Lee las corridas ya hechas (no entrena nada) y escribe PNG + resumen.md en
reports/ej1_pruebas/. Además copia a reports/ej1_resultados/ las figuras de
results/ que se muestran (results/ no se versiona: pesa ~70 MB con los pesos
y las figuras de cada corrida). Requiere haber corrido antes todas las
configs ej1a_* y ej1b_*, en particular:

    python main.py configs/ej1a_stability.json configs/ej1a_small_failures.json \
                   configs/ej1b_denoising_architectures.json configs/ej1b_noise_types.json

Uso, desde TP5:  python scripts/ej1_figuras.py
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "reports" / "ej1_pruebas"
EXPORT = ROOT / "reports" / "ej1_resultados"

# Figuras individuales que vale la pena versionar, además de las comparativas.
SELECTED_FIGURES = {
    "ej1a_baseline/base__seed=42": [
        "reconstruction.png", "latent.png", "latent_grid.png", "interpolation.png", "learning.png",
    ],
    "ej1a_small_failures/latent_size=2__hidden_layers=8__seed=1": ["reconstruction.png", "latent.png"],
    "ej1a_small_failures/latent_size=1__hidden_layers=32-16__seed=1": ["reconstruction.png"],
    "ej1b_denoising/level=0__seed=2": ["denoising_flip_0.1.png"],
    "ej1b_denoising/level=0.1__seed=3": ["denoising_flip_0.1.png", "denoising_flip_0.2.png"],
    "ej1b_denoising_architectures/latent_size=8__hidden_layers=32-16__seed=1": [
        "denoising.png", "denoising_flip_0.1.png", "denoising_flip_0.2.png",
    ],
}
TARGET = 1  # la consigna: a lo sumo un píxel incorrecto por letra

# Paleta categórica de referencia (orden fijo) y tinta.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.edgecolor": INK_SOFT,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


# ---------------------------------------------------------------- lectura


def runs(experiment: str) -> list[dict]:
    """Cada corrida: su config y la carpeta donde están sus CSV."""
    found = []
    for folder in sorted((RESULTS / experiment).iterdir()):
        config_file = folder / "config.json"
        if config_file.exists():
            found.append({"dir": folder, "config": json.loads(config_file.read_text())})
    if not found:
        raise SystemExit(f"No hay corridas en results/{experiment}; correr main.py primero")
    return found


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def hidden_name(layers: list[int]) -> str:
    return "-".join(map(str, layers)) if layers else "ninguna"


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(OUTPUT / name, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"  {OUTPUT.relative_to(ROOT) / name}")


# ------------------------------------------------- 1) estabilidad del objetivo


def stability(report: list[str]) -> None:
    groups: dict[tuple, list[list[dict]]] = defaultdict(list)
    for run in runs("ej1a_stability"):
        key = (
            tuple(run["config"]["model"]["hidden_layers"]),
            run["config"]["training"]["optimizer"]["learning_rate"],
        )
        groups[key].append(read_csv(run["dir"] / "history.csv"))

    architectures = sorted({key[0] for key in groups}, key=lambda h: (len(h), h))
    rates = sorted({key[1] for key in groups}, reverse=True)

    figure, axes = plt.subplots(1, len(architectures), figsize=(12, 3.8), sharey=True, layout="constrained")
    rows = []
    for axis, architecture in zip(axes, architectures):
        for color, rate in zip(SERIES, rates):
            histories = groups[(architecture, rate)]
            epochs = [int(row["epoch"]) for row in histories[0]]
            within = np.array(
                [[int(row["max_pixel_errors"]) <= TARGET for row in history] for history in histories]
            )
            axis.plot(epochs, within.mean(axis=0) * 100, color=color, label=f"η = {rate:g}")

            reached = [np.flatnonzero(seed) for seed in within]
            first = [int(epochs[hits[0]]) for hits in reached if len(hits)]
            # Después de cumplir por primera vez, ¿qué fracción de los controles vuelve a fallar?
            relapses = [
                1 - seed[hits[0] :].mean() for seed, hits in zip(within, reached) if len(hits)
            ]
            rows.append(
                {
                    "ocultas": hidden_name(list(architecture)),
                    "η": f"{rate:g}",
                    "alguna vez cumple": f"{len(first)}/{len(histories)}",
                    "cumple al final": f"{int(within[:, -1].sum())}/{len(histories)}",
                    "época mediana en que cumple": int(statistics.median(first)) if first else "—",
                    "controles fuera del objetivo después de cumplir": (
                        f"{100 * statistics.mean(relapses):.0f} %" if relapses else "—"
                    ),
                }
            )
        axis.set_title(f"Ocultas {hidden_name(list(architecture))}", color=INK)
        axis.set_xlabel("Época")
        axis.set_ylim(-3, 103)
    axes[0].set_ylabel("Semillas con ≤ 1 píxel mal (%)")
    axes[-1].legend(loc="lower right")
    figure.suptitle(
        "¿Cuántas de las 10 semillas cumplen el objetivo en cada momento? (latente 2, Adam)",
        color=INK,
    )
    save(figure, "estabilidad_semillas_en_objetivo.png")

    report += ["## 1. Estabilidad del ≤ 1 píxel (`ej1a_stability`)", "", _table(rows), ""]


# ---------------------------------------------- 2) por qué fallan las redes chicas


def small_failures(report: list[str]) -> None:
    by_config: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs("ej1a_small_failures"):
        model = run["config"]["model"]
        by_config[(model["latent_size"], tuple(model["hidden_layers"]))].append(run)

    configs = sorted(by_config, key=lambda key: (key[0], len(key[1]), key[1]))
    labels = [row["label"] for row in read_csv(by_config[configs[0]][0]["dir"] / "reconstruction.csv")]
    failures = np.zeros((len(configs), len(labels)), dtype=int)
    rows = []
    for i, key in enumerate(configs):
        runs_here = by_config[key]
        errors = np.array(
            [[float(row["error"]) for row in read_csv(run["dir"] / "reconstruction.csv")] for run in runs_here]
        )
        failures[i] = (errors > TARGET).sum(axis=0)
        worst = sorted(range(len(labels)), key=lambda j: -failures[i, j])[:6]
        always = int((failures[i] == len(runs_here)).sum())
        rows.append(
            {
                "latente": key[0],
                "ocultas": hidden_name(list(key[1])),
                "semillas que cumplen": f"{int((errors.max(axis=1) <= TARGET).sum())}/{len(runs_here)}",
                "letras mal (promedio)": f"{(errors > TARGET).sum(axis=1).mean():.1f}",
                "píxeles mal (promedio)": f"{errors.sum(axis=1).mean():.1f}",
                "letras que más fallan (semillas)": (
                    f"{always} de {len(labels)} fallan en todas las semillas"
                    if always > len(worst)
                    else " ".join(f"{labels[j]}({failures[i, j]})" for j in worst if failures[i, j])
                    or "—"
                ),
            }
        )

    figure, axis = plt.subplots(figsize=(12, 3.2))
    image = axis.imshow(failures, cmap="Blues", vmin=0, vmax=len(by_config[configs[0]]), aspect="auto")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(
        range(len(configs)), [f"latente {k[0]}, ocultas: {hidden_name(list(k[1]))}" for k in configs]
    )
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, pad=0.01)
    colorbar.set_label("Semillas en que la letra\ntiene > 1 píxel mal", color=INK)
    axis.set_title("Qué letras no aprende cada red (5 semillas)", color=INK)
    save(figure, "redes_chicas_letras_que_fallan.png")

    _latent_strip(by_config, labels)
    report += ["## 2. Por qué fallan las redes chicas (`ej1a_small_failures`)", "", _table(rows), ""]


def _latent_strip(by_config: dict, labels: list[str]) -> None:
    """Latente de 1 dimensión: dónde cae cada letra y cuáles salen mal."""
    key = next(k for k in by_config if k[0] == 1 and len(k[1]) == 2)
    run = by_config[key][0]
    codes = {row["label"]: float(row["z1"]) for row in read_csv(run["dir"] / "latent.csv")}
    errors = {row["label"]: float(row["error"]) for row in read_csv(run["dir"] / "reconstruction.csv")}

    ordered = sorted(labels, key=codes.get)
    figure, axis = plt.subplots(figsize=(12, 2.2))
    for index, label in enumerate(ordered):
        bad = errors[label] > TARGET
        axis.scatter(codes[label], 0, s=60, color=SERIES[1] if bad else SERIES[0], zorder=3,
                     edgecolors="white", linewidths=2)
        axis.annotate(
            label if not bad else f"{label} ({errors[label]:.0f})",
            (codes[label], 0),
            xytext=(0, 12 if index % 2 == 0 else -18),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
    axis.set_yticks([])
    axis.spines["left"].set_visible(False)
    axis.grid(axis="y", visible=False)
    axis.set_xlabel("z1 (única coordenada del latente)")
    axis.scatter([], [], color=SERIES[0], label="≤ 1 píxel mal")
    axis.scatter([], [], color=SERIES[1], label="> 1 píxel mal (entre paréntesis, cuántos)")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2)
    seed = run["config"]["seed"]
    axis.set_title(
        f"Latente de 1 dimensión (ocultas {hidden_name(list(key[1]))}, semilla {seed}): "
        "todas las letras en una sola recta",
        color=INK,
    )
    save(figure, "latente_1d_letras.png")


# ---------------------------------------------- 3) arquitectura del denoising


def denoising_architectures(report: list[str]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs("ej1b_denoising_architectures"):
        model = run["config"]["model"]
        groups[(tuple(model["hidden_layers"]), model["latent_size"])].append(run)

    architectures = sorted({k[0] for k in groups}, key=lambda h: (len(h), h))
    latents = sorted({k[1] for k in groups})
    figure, axes = plt.subplots(1, len(architectures), figsize=(10, 4.0), sharey=True, layout="constrained")
    rows = []
    for axis, architecture in zip(axes, architectures):
        noisy_reference = None
        for color, latent in zip(SERIES, latents):
            runs_here = groups[(architecture, latent)]
            tables = [read_csv(run["dir"] / "denoising.csv") for run in runs_here]
            levels = [float(row["level"]) for row in tables[0]]
            output = np.array([[float(row["output_mean_error"]) for row in table] for table in tables])
            noisy_reference = np.mean(
                [[float(row["input_mean_error"]) for row in table] for table in tables], axis=0
            )
            axis.plot(levels, output.mean(axis=0), color=color, marker="o", markersize=5,
                      label=f"latente {latent}")
            summaries = [json.loads((run["dir"] / "summary.json").read_text()) for run in runs_here]
            at = {level: output[:, i].mean() for i, level in enumerate(levels)}
            rows.append(
                {
                    "ocultas": hidden_name(list(architecture)),
                    "latente": latent,
                    "letras limpias ≤ 1 píxel": (
                        f"{sum(s['max_pixel_errors'] <= TARGET for s in summaries)}/{len(summaries)}"
                    ),
                    "salida @0,1": f"{at[0.1]:.2f}",
                    "salida @0,2": f"{at[0.2]:.2f}",
                    "salida @0,3": f"{at[0.3]:.2f}",
                }
            )
        axis.plot(levels, noisy_reference, color=INK_SOFT, linestyle="--", linewidth=1.5,
                  label="entrada con ruido (sin red)")
        axis.set_title(f"Ocultas {hidden_name(list(architecture))}", color=INK)
        axis.set_xlabel("Probabilidad de invertir cada píxel")
    axes[0].set_ylabel("Píxeles mal por letra (promedio)")
    axes[-1].legend(loc="upper left")
    figure.suptitle("Denoising: cuánto ruido queda a la salida (entrenado con 10 %, 5 semillas)",
                    color=INK)
    save(figure, "denoising_arquitecturas.png")

    noisy_row = {
        "ocultas": "(entrada sin red)",
        "latente": "—",
        "letras limpias ≤ 1 píxel": "—",
        "salida @0,1": f"{noisy_reference[levels.index(0.1)]:.2f}",
        "salida @0,2": f"{noisy_reference[levels.index(0.2)]:.2f}",
        "salida @0,3": f"{noisy_reference[levels.index(0.3)]:.2f}",
    }
    report += [
        "## 3. Arquitectura del denoising (`ej1b_denoising_architectures`)",
        "",
        "Píxeles mal por letra (de 35), promedio de 5 semillas y 50 repeticiones del font.",
        "",
        _table(rows + [noisy_row]),
        "",
    ]


# ---------------------------------------------- 4) tipo de ruido: entrenar vs probar


# Nivel de comparación de cada tipo: corrompen parecido una vez binarizada la entrada.
REFERENCE_LEVEL = {"flip": 0.1, "mask": 0.2, "salt_and_pepper": 0.2, "gaussian": 0.4}
NOISE_NAMES = {"flip": "invertir", "mask": "apagar", "salt_and_pepper": "sal y pimienta",
               "gaussian": "gaussiano"}


def noise_types(report: list[str]) -> None:
    cells: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for run in runs("ej1b_noise_types"):
        trained = run["config"]["training"]["noise"]["type"]
        tested = run["config"]["evaluation"]["denoising"]["type"]
        for row in read_csv(run["dir"] / "denoising.csv"):
            if abs(float(row["level"]) - REFERENCE_LEVEL[tested]) < 1e-9:
                cells[(trained, tested)].append(
                    (float(row["input_mean_error"]), float(row["output_mean_error"]))
                )

    kinds = list(REFERENCE_LEVEL)
    output = np.array([[np.mean([o for _, o in cells[(t, e)]]) for e in kinds] for t in kinds])
    noisy = np.array([np.mean([i for i, _ in cells[(kinds[0], e)]]) for e in kinds])

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    image = axis.imshow(output, cmap="Blues", vmin=0, aspect="auto")
    for i in range(len(kinds)):
        for j in range(len(kinds)):
            axis.text(j, i, f"{output[i, j]:.2f}", ha="center", va="center",
                      color="white" if output[i, j] > output.max() * 0.55 else INK, fontsize=10)
    axis.set_xticks(
        range(len(kinds)),
        [f"{NOISE_NAMES[k]} {REFERENCE_LEVEL[k]:g}\n(entrada: {noisy[j]:.1f})" for j, k in enumerate(kinds)],
        fontsize=8,
    )
    axis.set_yticks(range(len(kinds)), [f"{NOISE_NAMES[k]} {REFERENCE_LEVEL[k]:g}" for k in kinds])
    axis.set_xlabel("Ruido con el que se prueba")
    axis.set_ylabel("Ruido con el que se entrenó")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.colorbar(image, ax=axis, pad=0.02).set_label("Píxeles mal a la salida", color=INK)
    axis.set_title("Entrenar con un ruido y probar con otro\n(latente 8, ocultas 32-16, 3 semillas)",
                   color=INK)
    save(figure, "denoising_tipos_de_ruido.png")

    rows = [
        {"entrenado con": NOISE_NAMES[t],
         **{f"probado con {NOISE_NAMES[e]}": f"{output[i, j]:.2f}" for j, e in enumerate(kinds)}}
        for i, t in enumerate(kinds)
    ]
    rows.append({"entrenado con": "(entrada sin red)",
                 **{f"probado con {NOISE_NAMES[e]}": f"{noisy[j]:.2f}" for j, e in enumerate(kinds)}})
    report += [
        "## 4. Tipo de ruido (`ej1b_noise_types`)",
        "",
        "Píxeles mal por letra a la salida. Nivel de cada ruido: "
        + ", ".join(f"{NOISE_NAMES[k]} {v:g}" for k, v in REFERENCE_LEVEL.items())
        + ".",
        "",
        _table(rows),
        "",
    ]


def _table(rows: list[dict]) -> str:
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    # La comilla invertida (una letra del font) abriría código en Markdown.
    lines += ["| " + " | ".join(str(row[h]).replace("`", "\\`") for h in header) + " |" for row in rows]
    return "\n".join(lines)


def export_results() -> None:
    """Copia las comparativas de cada experimento y las figuras de SELECTED_FIGURES."""
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    copied = 0
    for summary in sorted(RESULTS.glob("ej1*/summary.csv")):
        for figure in sorted(summary.parent.glob("comparison*.png")):
            target = EXPORT / summary.parent.name / figure.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(figure, target)
            copied += 1
    for run, names in SELECTED_FIGURES.items():
        for name in names:
            source = RESULTS / run / name
            if not source.exists():
                print(f"  falta {source.relative_to(ROOT)}; se omite")
                continue
            target = EXPORT / run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    (EXPORT / "README.md").write_text(
        "# Ejercicio 1: figuras de results/\n\n"
        "Copia liviana de `results/` (que no se versiona): las comparativas de cada experimento\n"
        "(`<experimento>/comparison_*.png`) y algunas corridas de ejemplo (`<experimento>/<corrida>/`).\n"
        "Los números están en `../ej1_pruebas/resumen.md`.\n\n"
        "Se regenera con `python scripts/ej1_figuras.py` después de correr las configs `ej1a_*` y `ej1b_*`.\n"
    )
    print(f"  {copied} figuras copiadas a {EXPORT.relative_to(ROOT)}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = ["# Ejercicio 1: pruebas extra para la presentación", ""]
    for step in (stability, small_failures, denoising_architectures, noise_types):
        step(report)
    (OUTPUT / "resumen.md").write_text("\n".join(report))
    print(f"  {(OUTPUT / 'resumen.md').relative_to(ROOT)}")
    export_results()


if __name__ == "__main__":
    main()
