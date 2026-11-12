"""Figuras de las búsquedas de los ejercicios 1a y 1b.

Lee las corridas ya hechas (no entrena nada) y escribe PNG en
reports/ej1_busqueda/figuras/ y reports/ej1b_busqueda/figuras/, con el mismo
estilo que scripts/ej1_figuras.py. También copia las figuras de los modelos
finales (ej1a_final, ej1b_final), porque results/ no se versiona.

Uso, desde TP5:  python scripts/busqueda_figuras.py
"""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import sys

import ej1a_search_table as search_1a
import ej1b_search_table as search_1b
import matplotlib.pyplot as plt
import numpy as np
from ej1_figuras import INK, INK_SOFT  # aplica también el estilo de las otras figuras

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import plotting  # noqa: E402
from src.data.noise import Noise  # noqa: E402
from src.experiments import runner  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT_1A = ROOT / "reports" / "ej1_busqueda" / "figuras"
OUTPUT_1B = ROOT / "reports" / "ej1b_busqueda" / "figuras"

# Paleta categórica de referencia, en orden fijo; cada entidad conserva su color en todas las figuras.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ACTIVATIONS = {"tanh": BLUE, "relu": ORANGE, "leaky_relu": AQUA, "sigmoid": YELLOW}
NOISES = {"flip": BLUE, "mask": ORANGE, "salt_and_pepper": AQUA, "gaussian": YELLOW}
PAIRED = {"tanh": "xavier", "relu": "he", "leaky_relu": "he", "sigmoid": "xavier_sigmoid"}
FINAL_FIGURES = {
    OUTPUT_1A: ("ej1a_final/base__seed=1", [
        "reconstruction.png", "latent.png", "latent_grid.png", "interpolation.png",
    ]),
    OUTPUT_1B: ("ej1b_final/base__seed=1", [
        "denoising.png", "denoising_flip_0.1.png", "denoising_flip_0.2.png",
        "denoising_mask_0.6.png", "denoising_gaussian_0.59.png",
    ]),
}


def decimal(value: float) -> str:
    """Número con coma decimal, como en el resto de los reportes."""
    return f"{value:g}".replace(".", ",")


def save(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / name, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"  {(directory / name).relative_to(ROOT)}")


def label_end(axis: plt.Axes, x: float, y: float, text: str) -> None:
    """Etiqueta directa al final de una línea (tinta de texto, no el color de la serie)."""
    axis.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                  color=INK, fontsize=9)


# ------------------------------------------------------------------ 1a


def stage_one_width() -> None:
    """Etapa 1: con una capa, ¿cuánto ancho hace falta y cuánto tarda cada activación?"""
    rows = search_1a.collect(sorted(RESULTS.glob(search_1a.STAGE_ONE)))
    widths = [16, 32, 64, 128]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for activation, color in ACTIVATIONS.items():
        mine = [row for row in rows if row["activation"] == activation
                and row["initializer"] == PAIRED[activation]]
        reached = [sum(row["final"] for row in mine if row["hidden"] == (width,)) for width in widths]
        steps = [next(row["median"] for row in mine if row["hidden"] == (width,) and row["rate"] == 0.01)
                 for width in widths]
        left.plot(widths, reached, color=color, marker="o", label=activation)
        right.plot(widths, steps, color=color, marker="o", label=activation)
    for axis in (left, right):
        axis.set_xscale("log", base=2)
        axis.set_xticks(widths, [str(width) for width in widths])
        axis.set_xlabel("Neuronas de la capa oculta (una capa)")
    left.set_ylabel("Semillas que terminan en la meta\n(de 30: 10 semillas × 3 η)")
    left.set_ylim(0, 31)
    left.legend(loc="lower right")
    right.set_ylabel("Pasos hasta la meta (mediana, η 0,01)")
    right.set_ylim(bottom=0)
    figure.suptitle("Etapa 1: pasado un ancho de 64 no mejora; sigmoid es la más lenta", color=INK)
    save(figure, OUTPUT_1A, "etapa1_ancho_y_activacion.png")


def batch_and_rate() -> None:
    """Interacción batch × η: semillas en la meta sobre las 9 redes de la etapa 2."""
    networks = {(64,), (64, 32), (128, 64)}
    activations = {"tanh", "relu", "leaky_relu"}
    rows = search_1a.collect(sorted((RESULTS / "ej1a_etapa2").glob("batch__*")))
    rows += search_1a.collect(sorted(RESULTS.glob(search_1a.STAGE_ONE)))
    rows = [row for row in rows if row["hidden"] in networks and row["activation"] in activations
            and row["initializer"] == PAIRED[row["activation"]] and row["optimizer"] == "adam"
            and row["loss"] == "bce"]
    batches, rates = [4, 8, 16, 32], [0.01, 0.003, 0.001]
    grid = np.array([[sum(row["final"] for row in rows if row["batch"] == batch and row["rate"] == rate)
                      for rate in rates] for batch in batches])
    total = sum(row["seeds"] for row in rows if row["batch"] == 32 and row["rate"] == 0.01)

    figure, axis = plt.subplots(figsize=(5, 3.6), layout="constrained")
    image = axis.imshow(grid, cmap="Blues", vmin=0, vmax=total, aspect="auto")
    for i in range(len(batches)):
        for j in range(len(rates)):
            axis.text(j, i, f"{grid[i, j]}/{total}", ha="center", va="center", fontsize=10,
                      color="white" if grid[i, j] > 0.6 * total else INK)
    axis.set_xticks(range(len(rates)), [decimal(rate) for rate in rates])
    axis.set_yticks(range(len(batches)), [str(batch) for batch in batches])
    axis.set(xlabel="η", ylabel="Batch")
    axis.grid(False)
    figure.colorbar(image, ax=axis, pad=0.02).set_label("Semillas en la meta", color=INK)
    axis.set_title("Batch chico con η alto se vuelve inestable", color=INK)
    save(figure, OUTPUT_1A, "etapa2_batch_y_eta.png")


def depth_against_narrowness() -> None:
    """Ancho 16: ¿falla por tener más capas o por tener una capa angosta?"""
    rows = search_1a.collect(sorted((RESULTS / "ej1a_etapa2").glob("profundidad__*")))
    rows += search_1a.collect(sorted(RESULTS.glob(search_1a.STAGE_ONE)))
    architectures = [(16,), (16, 16), (16, 16, 16), (16, 8), (16, 4), (16, 8, 4)]
    activations = ["tanh", "relu", "leaky_relu"]
    figure, axis = plt.subplots(figsize=(8, 3.6), layout="constrained")
    width = 0.26
    for offset, activation in enumerate(activations):
        values = []
        for architecture in architectures:
            match = [row for row in rows if row["activation"] == activation and row["hidden"] == architecture
                     and row["initializer"] == PAIRED[activation] and row["rate"] == 0.01
                     and row["batch"] == 32 and row["optimizer"] == "adam" and row["loss"] == "bce"]
            values.append(match[0]["final"] if match else np.nan)
        positions = np.arange(len(architectures)) + (offset - 1) * width
        axis.bar(positions, values, width=width - 0.03, color=ACTIVATIONS[activation], label=activation)
        for position, value in zip(positions, values):
            if value == 0:  # una barra vacía no se distingue de un dato faltante
                axis.text(position, 0.15, "0", ha="center", color=INK, fontsize=9)
    axis.set_xticks(range(len(architectures)), ["-".join(map(str, a)) for a in architectures])
    axis.axvline(2.5, color=INK_SOFT, linewidth=1, linestyle=":")
    axis.text(1, 10.6, "misma anchura, más capas", ha="center", color=INK_SOFT, fontsize=9)
    axis.text(4, 10.6, "capas que se angostan", ha="center", color=INK_SOFT, fontsize=9)
    axis.set(xlabel="Capas ocultas (η 0,01)", ylabel="Semillas en la meta (de 10)", ylim=(0, 11.5))
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=3)
    axis.set_title("relu falla por angostura, tanh por profundidad", color=INK)
    save(figure, OUTPUT_1A, "etapa2_profundidad_y_angostura.png")


def candidates() -> None:
    """Pasos hasta la meta de cada semilla en las cuatro candidatas finales."""
    chosen = [("relu", [32]), ("tanh", [64]), ("tanh", [64, 32]), ("tanh", [128, 64])]
    figure, axis = plt.subplots(figsize=(7, 3.4), layout="constrained")
    rng = np.random.default_rng(0)  # solo para separar puntos superpuestos
    for position, (activation, hidden) in enumerate(chosen):
        steps = []
        for config_path in sorted(RESULTS.glob(f"ej1a_search_{activation}_{PAIRED[activation]}/*/config.json")):
            config = json.loads(config_path.read_text())
            if (config["model"]["hidden_layers"] == hidden
                    and config["training"]["optimizer"]["learning_rate"] == 0.01):
                steps.append(search_1a.seed_statistics(config_path.parent)["first_epoch"])
        jitter = rng.uniform(-0.12, 0.12, len(steps))
        axis.scatter(np.full(len(steps), position) + jitter, steps, s=36, color=ACTIVATIONS[activation],
                     edgecolors="white", linewidths=1.5, zorder=3)
        median = statistics.median(steps)
        axis.plot([position - 0.25, position + 0.25], [median, median], color=INK, linewidth=2)
        axis.annotate(f"{median:.0f}", (position + 0.27, median), va="center", color=INK, fontsize=9)
    parameters = [search_1a.parameter_count(hidden, 2) for _, hidden in chosen]
    axis.set_xticks(range(len(chosen)), [f"{a} {'-'.join(map(str, h))}\n{p} parámetros"
                                          for (a, h), p in zip(chosen, parameters)])
    axis.set(ylabel="Pasos hasta la meta (cada punto, una semilla)", ylim=(0, None))
    axis.set_title("Candidatas finales: todas cumplen 10/10; más grande, menos pasos", color=INK)
    save(figure, OUTPUT_1A, "candidatas_pasos.png")


# ------------------------------------------------------------------ 1b


def training_noise() -> None:
    """Red elegida: letras recuperadas según con qué ruido y cuánto se la entrenó."""
    rows = search_1b.collect(search_1b.STAGES[3], "*/config.json")
    clean = next(row for row in rows if row["trained"] == "sin ruido")["promedio"]["recovered"]
    inputs = [1.75, 3.5, 7, 10.5]
    figure, axis = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for kind, color in NOISES.items():
        name = search_1b.NAMES[kind]
        levels = sorted(float(row["trained"].rsplit(" ", 1)[1]) for row in rows
                        if row["trained"].startswith(name + " "))
        values = [100 * next(row for row in rows if row["trained"] == f"{name} {level:g}")["promedio"]["recovered"]
                  for level in levels]
        # flip, sal y pimienta y gaussiano se superponen (ese es el resultado): formas distintas.
        marker = {"flip": "o", "mask": "o", "salt_and_pepper": "s", "gaussian": "^"}[kind]
        size = {"flip": 9}.get(kind, 6)
        axis.plot(inputs, values, color=color, marker=marker, markersize=size, label=f"entrenada con {name}")
    axis.axhline(100 * clean, color=INK_SOFT, linestyle="--", linewidth=1.5, label="entrenada sin ruido")
    optimal = statistics.mean(row["promedio"]["optimal_recovered"] for row in rows)
    axis.axhline(100 * optimal, color=INK, linestyle=":", linewidth=1.5, label="óptimo")
    axis.set_xticks(inputs, ["1,75", "3,5", "7", "10,5"])
    axis.set(xlabel="Píxeles mal a la entrada durante el entrenamiento (nivel de ruido)",
             ylabel="Letras recuperadas (%)\npromedio de los 4 tipos de prueba", ylim=(0, 100))
    axis.legend(loc="lower left", fontsize=8.5)
    axis.annotate("flip, sal y pimienta y gaussiano\nse superponen", (7, 90), xytext=(4.6, 72),
                  color=INK, fontsize=9, arrowprops={"arrowstyle": "-", "color": INK_SOFT})
    axis.set_title("Entrenar con ruido es lo que más importa; mask solo sirve para mask", color=INK)
    save(figure, OUTPUT_1B, "ruido_de_entrenamiento.png")


def architecture_and_rate() -> None:
    """Etapa 2: letras recuperadas por tamaño de red, relu contra tanh, en cada η."""
    rows = search_1b.collect(search_1b.STAGES[2])
    sizes = ["32", "64", "64-32", "128-64"]
    rates = [0.01, 0.003, 0.001]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, layout="constrained")
    for axis, rate in zip(axes, rates):
        for activation in ("relu", "tanh"):
            values = [100 * next(row for row in rows if row["network"] == f"{activation} {size}"
                                 and row["latent"] == 8 and row["rate"] == rate
                                 and row["epochs"] == 20000)["promedio"]["recovered"] for size in sizes]
            axis.plot(range(len(sizes)), values, color=ACTIVATIONS[activation], marker="o", label=activation)
            label_end(axis, len(sizes) - 1, values[-1], activation)
        axis.set_xticks(range(len(sizes)), sizes)
        axis.set_title(f"η = {decimal(rate)}", color=INK)
        axis.set_xlabel("Capas ocultas")
    axes[0].set_ylabel("Letras recuperadas (%)\npromedio de los 4 tipos")
    axes[0].legend(loc="lower left")
    figure.suptitle("Latente 8, 20000 épocas: con η ≤ 0,003 más capacidad ayuda; relu supera a tanh",
                    color=INK)
    save(figure, OUTPUT_1B, "arquitectura_y_eta.png")


def types_matrix() -> None:
    """Entrenada con × probada con, al nivel de 7 píxeles mal, y el óptimo."""
    rows = search_1b.collect(search_1b.STAGES[3], "*/config.json")
    trained = ["flip 0.2", "sal y pimienta 0.4", "gaussiano 0.59", "mask 0.6", "sin ruido"]
    kinds = list(search_1b.RANKED)
    grid = [[100 * next(row for row in rows if row["trained"] == name)[kind]["recovered"] for kind in kinds]
            for name in trained]
    grid.append([100 * statistics.mean(row[kind]["optimal_recovered"] for row in rows) for kind in kinds])
    grid = np.array(grid)
    figure, axis = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    image = axis.imshow(grid, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            axis.text(j, i, f"{grid[i, j]:.0f} %", ha="center", va="center", fontsize=10,
                      color="white" if grid[i, j] > 60 else INK)
    axis.axhline(len(trained) - 0.5, color="white", linewidth=3)
    axis.set_xticks(range(len(kinds)), [search_1b.NAMES[kind] for kind in kinds])
    axis.set_yticks(range(len(grid)), [*(name.replace(".", ",") for name in trained), "óptimo"])
    axis.set(xlabel="Probada con", ylabel="Entrenada con")
    axis.grid(False)
    figure.colorbar(image, ax=axis, pad=0.02).set_label("Letras recuperadas (%)", color=INK)
    axis.set_title("Cada ruido limpia a los demás, salvo mask", color=INK)
    save(figure, OUTPUT_1B, "tipos_entrenada_probada.png")


def against_optimal() -> None:
    """Red elegida entrenada con flip 0,2: letras recuperadas por nivel, contra el óptimo."""
    per_level: dict[tuple, list[dict]] = defaultdict(list)
    for config_path in sorted(search_1b.STAGES[3].glob("*/config.json")):
        noise = json.loads(config_path.read_text())["training"]["noise"]
        if (noise["type"], noise["level"]) != ("flip", 0.2):
            continue
        for row in csv.DictReader((config_path.parent / "denoising.csv").open()):
            per_level[(row["noise_type"], float(row["level"]))].append(row)
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True, layout="constrained")
    for axis, (kind, color) in zip(axes, NOISES.items()):
        keys = sorted(key for key in per_level if key[0] == kind and key[1] > 0)
        mean = lambda key, name: statistics.mean(float(r[name]) for r in per_level[key])  # noqa: E731
        inputs = [mean(key, "input_mean_error") for key in keys]
        network = [100 * mean(key, "fraction_within_target") for key in keys]
        optimal = [100 * mean(key, "optimal_fraction_within_target") for key in keys]
        axis.fill_between(inputs, network, optimal, color=color, alpha=0.15, linewidth=0)
        axis.plot(inputs, optimal, color=INK, linestyle=":", marker="o", markersize=4, label="óptimo")
        axis.plot(inputs, network, color=color, marker="o", label="red")
        axis.set_title(search_1b.NAMES[kind], color=INK)
        axis.set_xlabel("Píxeles mal a la entrada")
        axis.set_xticks([0, 3.5, 7, 10.5], ["0", "3,5", "7", "10,5"])
        axis.legend(loc="lower left", fontsize=8.5)
    axes[0].set_ylabel("Letras recuperadas (%)")
    axes[0].set_ylim(0, 102)
    figure.suptitle("Entrenada con flip 0,2: casi óptima con poco ruido; la franja es lo que pierde la red",
                    color=INK)
    save(figure, OUTPUT_1B, "capacidad_contra_optimo.png")


# Denoising anterior (configs/ej1b_anterior_comparable.json, la config de antes con la evaluación nueva) y nuevo.
COMPARED = {
    "anterior: tanh 32-16, latente 2": (RESULTS / "ej1b_anterior_comparable", "latent_size=2__*", ORANGE),
    "mejor anterior: latente 8": (RESULTS / "ej1b_anterior_comparable", "latent_size=8__*", AQUA),
    "nueva: relu 128-64, latente 8": (search_1b.STAGES[3], "noise=flip,level=0.2__*", BLUE),
}


def comparison_with_previous() -> None:
    """Letras recuperadas por tipo y nivel: denoising anterior, nuevo y óptimo."""
    def per_level(directory: Path, pattern: str) -> dict[tuple, dict[str, float]]:
        values: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for config_path in sorted(directory.glob(f"{pattern}/config.json")):
            for row in csv.DictReader((config_path.parent / "denoising.csv").open()):
                for name in ("input_mean_error", "fraction_within_target", "optimal_fraction_within_target"):
                    values[(row["noise_type"], float(row["level"]))][name].append(float(row[name]))
        return {key: {name: statistics.mean(v) for name, v in named.items()} for key, named in values.items()}

    measured = {label: per_level(directory, pattern) for label, (directory, pattern, _) in COMPARED.items()}
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True, layout="constrained")
    for axis, kind in zip(axes, NOISES):
        newest = measured[next(reversed(measured))]
        keys = sorted(key for key in newest if key[0] == kind and key[1] > 0)
        inputs = [newest[key]["input_mean_error"] for key in keys]
        axis.plot(inputs, [100 * newest[key]["optimal_fraction_within_target"] for key in keys],
                  color=INK, linestyle=":", marker="o", markersize=4, label="óptimo")
        for label, (_, _, color) in COMPARED.items():
            axis.plot(inputs, [100 * measured[label][key]["fraction_within_target"] for key in keys],
                      color=color, marker="o", label=label)
        axis.set_title(search_1b.NAMES[kind], color=INK)
        axis.set_xlabel("Píxeles mal a la entrada")
        axis.set_xticks([0, 3.5, 7, 10.5], ["0", "3,5", "7", "10,5"])
    axes[0].set_ylabel("Letras recuperadas (%)")
    axes[0].set_ylim(0, 102)
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("Denoising anterior contra el nuevo (misma evaluación, 10 semillas cada uno)", color=INK)
    save(figure, OUTPUT_1B, "comparacion_anterior.png")


def comparison_example(kind: str = "flip", level: float = 0.2) -> None:
    """Las tres redes (semilla 1) limpiando exactamente las mismas letras ruidosas."""
    models = {}
    for label, (directory, pattern, _) in COMPARED.items():
        run = next(path for path in sorted(directory.glob(pattern)) if path.name.endswith("seed=1"))
        models[label], dataset = runner.load_run(run)
    noisy = Noise(kind, level).apply(dataset.samples, np.random.default_rng(7))
    rows = [dataset.samples, noisy] + [(model.reconstruct(noisy) >= 0.5).astype(float) for model in models.values()]
    names = ["Original", "Con ruido", "Anterior", "Mejor anterior", "Nueva"]
    OUTPUT_1B.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_1B / f"comparacion_ejemplo_{kind}_{level:g}.png"
    plotting.save_image_rows(rows, names, dataset.labels, dataset.image_shape, True, path)
    errors = {name: int(np.sum(np.count_nonzero(row != dataset.samples, axis=1) > 1))
              for name, row in zip(names[2:], rows[2:])}
    print(f"  {path.relative_to(ROOT)} (letras con > 1 píxel mal: {errors})")


def copy_final_figures() -> None:
    for directory, (run, names) in FINAL_FIGURES.items():
        target = directory / "modelo_final"
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(RESULTS / run / name, target / name)
            print(f"  {(target / name).relative_to(ROOT)}")


def main() -> None:
    for step in (stage_one_width, batch_and_rate, depth_against_narrowness, candidates,
                 training_noise, architecture_and_rate, types_matrix, against_optimal,
                 copy_final_figures):
        step()
    # Opcional: solo si se corrió configs/ej1b_anterior_comparable.json.
    if any((RESULTS / "ej1b_anterior_comparable").glob("*/denoising.csv")):
        comparison_with_previous()
        comparison_example()
    else:
        print("  (sin results/ej1b_anterior_comparable: se omite la comparación con el denoising anterior)")


if __name__ == "__main__":
    main()
