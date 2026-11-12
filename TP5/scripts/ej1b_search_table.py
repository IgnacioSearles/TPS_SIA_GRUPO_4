"""Tablas de la búsqueda del denoising autoencoder del ejercicio 1b.

Agrupa las semillas de cada configuración (red, latente, ruido de
entrenamiento, η, épocas) y las ordena, para cada tipo de ruido de prueba y para el
promedio de los cuatro, por los criterios definidos antes de correr:
  1. letras recuperadas (≤ 1 píxel mal), promedio de los niveles con 1,75,
     3,5 y 7 píxeles mal a la entrada
  2. píxeles mal promedio en esos niveles
El nivel más alto (10,5 píxeles mal) se informa pero no ordena.

Uso, desde TP5:
    python scripts/ej1b_search_table.py             # etapa 1 -> tabla_<tipo>.md
    python scripts/ej1b_search_table.py --stage 2   # etapa 2 -> etapa2_<tipo>.md
    python scripts/ej1b_search_table.py --stage 3   # tipos de ruido con la red elegida -> tipos.md
En reports/ej1b_busqueda/, una tabla por tipo de prueba y una del promedio.
La etapa 2 incluye como referencia las filas de la etapa 1 de las mismas redes.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = {
    1: ROOT / "results" / "ej1b_busqueda",
    2: ROOT / "results" / "ej1b_etapa2",
    3: ROOT / "results" / "ej1b_tipos",
}
OUTPUT = ROOT / "reports" / "ej1b_busqueda"
# Niveles de cada tipo con 1,75, 3,5 y 7 píxeles mal a la entrada (ordenan) y con 10,5 (no).
RANKED = {
    "flip": (0.05, 0.1, 0.2),
    "mask": (0.15, 0.3, 0.6),
    "salt_and_pepper": (0.1, 0.2, 0.4),
    "gaussian": (0.3, 0.39, 0.59),
}
HARDEST = {"flip": 0.3, "mask": 0.91, "salt_and_pepper": 0.6, "gaussian": 0.95}
AVERAGE = "promedio"
NAMES = {"flip": "flip", "mask": "mask", "salt_and_pepper": "sal y pimienta", "gaussian": "gaussiano"}


def seed_scores(run_directory: Path) -> dict[str, dict[str, float]]:
    """Por tipo de prueba: letras recuperadas, píxeles mal, óptimos y nivel más alto."""
    rows = list(csv.DictReader((run_directory / "denoising.csv").open()))
    scores = {}
    for kind, levels in RANKED.items():
        ranked = [row for row in rows if row["noise_type"] == kind and float(row["level"]) in levels]
        hardest = next(
            row for row in rows if row["noise_type"] == kind and float(row["level"]) == HARDEST[kind]
        )
        scores[kind] = {
            "recovered": statistics.mean(float(row["fraction_within_target"]) for row in ranked),
            "pixels": statistics.mean(float(row["output_mean_error"]) for row in ranked),
            "optimal_recovered": statistics.mean(
                float(row["optimal_fraction_within_target"]) for row in ranked
            ),
            "optimal_pixels": statistics.mean(float(row["optimal_mean_error"]) for row in ranked),
            "hardest": float(hardest["fraction_within_target"]),
        }
    clean = next(row for row in rows if row["noise_type"] == "flip" and float(row["level"]) == 0)
    scores[AVERAGE] = {
        name: statistics.mean(scores[kind][name] for kind in RANKED) for name in scores["flip"]
    }
    for kind_scores in scores.values():
        kind_scores["clean"] = float(clean["fraction_within_target"])
    return scores


def collect(results: Path, runs: str = "*/*/config.json") -> list[dict]:
    """`runs`: patrón de las corridas dentro de `results` (por defecto <experimento>/<corrida>)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for config_path in sorted(results.glob(runs)):
        if not (config_path.parent / "denoising.csv").is_file():
            continue  # divergió o no terminó
        config = json.loads(config_path.read_text())
        model, training = config["model"], config["training"]
        noise = training["noise"]
        network = f"{model['activation']} {'-'.join(map(str, model['hidden_layers']))}"
        trained = "sin ruido" if noise["level"] == 0 else f"{NAMES[noise['type']]} {noise['level']:g}"
        key = (network, model["latent_size"], trained, training["optimizer"]["learning_rate"], training["epochs"])
        groups[key].append(seed_scores(config_path.parent))

    rows = []
    for (network, latent, trained, rate, epochs), seeds in groups.items():
        row = {
            "network": network, "latent": latent, "trained": trained,
            "rate": rate, "epochs": epochs, "seeds": len(seeds),
        }
        for kind in (*RANKED, AVERAGE):
            per_seed = [seed[kind] for seed in seeds]
            row[kind] = {name: statistics.mean(s[name] for s in per_seed) for name in per_seed[0]}
            row[kind]["spread"] = statistics.stdev(s["recovered"] for s in per_seed)
        rows.append(row)
    return rows


def as_markdown(rows: list[dict], kind: str) -> list[str]:
    ranked = sorted(rows, key=lambda row: (-row[kind]["recovered"], row[kind]["pixels"]))
    # Cada corrida estima el óptimo con su propio ruido; el encabezado muestra el promedio.
    optimal = {name: statistics.mean(row[kind][name] for row in rows) for name in ("optimal_recovered", "optimal_pixels")}
    lines = [
        f"Óptimo: {100 * optimal['optimal_recovered']:.1f} % de letras recuperadas, "
        f"{optimal['optimal_pixels']:.2f} píxeles mal (la brecha de cada fila usa su mismo ruido).",
        "",
        "| # | red | latente | entrenada con | η | épocas | letras recuperadas (desvío entre semillas) | "
        "brecha con el óptimo | píxeles mal | letras @10,5 | letras sin ruido | semillas |",
        "|" + "---|" * 12,
    ]
    for position, row in enumerate(ranked, start=1):
        scores = row[kind]
        lines.append(
            f"| {position} | {row['network']} | {row['latent']} | {row['trained']} | "
            f"{row['rate']:g} | {row['epochs']} | "
            f"{100 * scores['recovered']:.1f} % (± {100 * scores['spread']:.1f}) | "
            f"{100 * (scores['optimal_recovered'] - scores['recovered']):.1f} | "
            f"{scores['pixels']:.2f} | {100 * scores['hardest']:.1f} % | "
            f"{100 * scores['clean']:.0f} % | {row['seeds']} |"
        )
    return lines


def noise_types_report(results: Path) -> list[str]:
    """Entrenada con cada ruido × probada con cada ruido, y el detalle por nivel."""
    rows = sorted(collect(results, "*/config.json"), key=lambda row: -row[AVERAGE]["recovered"])
    lines = [
        "## Entrenada con × probada con",
        "",
        "Letras recuperadas (≤ 1 píxel mal), promedio de los niveles con 1,75, 3,5 y 7 píxeles mal.",
        "",
        "| entrenada con | " + " | ".join(NAMES[kind] for kind in RANKED) + " | promedio | sin ruido |",
        "|" + "---|" * (len(RANKED) + 3),
    ]
    for row in rows:
        cells = [f"{100 * row[kind]['recovered']:.1f} %" for kind in (*RANKED, AVERAGE)]
        lines.append(f"| {row['trained']} | " + " | ".join(cells) + f" | {100 * row[AVERAGE]['clean']:.0f} % |")
    optimal = [
        f"{100 * statistics.mean(row[kind]['optimal_recovered'] for row in rows):.1f} %"
        for kind in (*RANKED, AVERAGE)
    ]
    lines += [f"| **óptimo** | " + " | ".join(optimal) + " | 100 % |", ""]
    return lines


def level_detail(results: Path, trained: str) -> list[str]:
    """Por tipo y nivel de prueba: píxeles mal y letras recuperadas, de la red y del óptimo."""
    sums: dict[tuple, list[dict]] = defaultdict(list)
    for config_path in sorted(results.glob("*/config.json")):
        noise = json.loads(config_path.read_text())["training"]["noise"]
        if f"{noise['type']} {noise['level']:g}" != trained:
            continue
        for row in csv.DictReader((config_path.parent / "denoising.csv").open()):
            sums[(row["noise_type"], float(row["level"]))].append(row)
    lines = [
        f"## Detalle por nivel, entrenada con {trained}",
        "",
        "| prueba | nivel | píxeles mal a la entrada | píxeles mal a la salida | óptimo | "
        "letras recuperadas | óptimo |",
        "|" + "---|" * 7,
    ]
    for (kind, level), runs in sums.items():
        mean = lambda name: statistics.mean(float(run[name]) for run in runs)  # noqa: E731
        lines.append(
            f"| {NAMES[kind]} | {level:g} | {mean('input_mean_error'):.2f} | "
            f"{mean('output_mean_error'):.2f} | {mean('optimal_mean_error'):.2f} | "
            f"{100 * mean('fraction_within_target'):.1f} % | "
            f"{100 * mean('optimal_fraction_within_target'):.1f} % |"
        )
    return [*lines, ""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=int, choices=tuple(STAGES), default=1)
    stage = parser.parse_args().stage
    if stage == 3:
        results = STAGES[3]
        report = [
            "# Ejercicio 1b: tipos de ruido con la red elegida",
            "",
            "relu 128-64, latente 8, Adam η 0,001, 20000 épocas, 10 semillas por fila.",
            "",
            *noise_types_report(results),
            *level_detail(results, "flip 0.2"),
            *level_detail(results, "mask 0.6"),
        ]
        path = OUTPUT / "tipos.md"
        path.write_text("\n".join(report), encoding="utf-8")
        print(f"-> {path.relative_to(ROOT)}")
        return
    rows = collect(STAGES[stage])
    if not rows:
        raise SystemExit(f"No hay corridas en {STAGES[stage].relative_to(ROOT)}; correr main.py primero")
    if stage == 2:
        compared = {(row["network"], row["latent"], row["trained"]) for row in rows}
        rows += [
            row for row in collect(STAGES[1])
            if (row["network"], row["latent"], row["trained"]) in compared
        ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for kind in (*RANKED, AVERAGE):
        tested = "el promedio de los cuatro tipos" if kind == AVERAGE else NAMES[kind]
        report = [
            f"# Ejercicio 1b: probadas con {tested}",
            "",
            "Letras recuperadas = ≤ 1 píxel mal; promedio de los niveles con 1,75, 3,5 y 7 píxeles",
            "mal a la entrada, de 50 repeticiones del font y de las semillas. Orden: letras",
            "recuperadas ↓, píxeles mal ↑. La brecha está en puntos porcentuales. @10,5: nivel más",
            "alto, no ordena. Sin ruido: letras limpias reconstruidas con ≤ 1 píxel mal.",
            "",
            *as_markdown(rows, kind),
            "",
        ]
        name = "promedio" if kind == AVERAGE else kind
        path = OUTPUT / f"{'tabla' if stage == 1 else 'etapa2'}_{name}.md"
        path.write_text("\n".join(report), encoding="utf-8")
        print(f"{len(rows)} configuraciones -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
