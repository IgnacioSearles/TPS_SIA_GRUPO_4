"""Tablas de la búsqueda de configuraciones del ejercicio 1a.

Agrupa las semillas de cada configuración y las ordena por los criterios
definidos antes de correr, en este orden:
  1. semillas que terminan cumpliendo (≤ 1 píxel mal en las 32 letras);
     una corrida que divergió cuenta como que no cumple
  2. recaídas: fracción de controles fuera del objetivo después de cumplir,
     redondeada a puntos porcentuales enteros
  3. pasos de gradiente medianos hasta cumplir por primera vez
     (con batch 32 = el font entero, un paso por época)
  4. cantidad de parámetros (desempate: la más simple)

Uso, desde TP5:
    python scripts/ej1a_search_table.py                   # etapa 1 (ej1a_search_*)
    python scripts/ej1a_search_table.py --stage 2         # una tabla por prueba de la etapa 2
Escribe reports/ej1_busqueda/etapa<N>[_<prueba>].md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "reports" / "ej1_busqueda"
STAGE_ONE = "ej1a_search_*"
STAGE_TWO = RESULTS / "ej1a_etapa2"
TARGET = 1  # la consigna: a lo sumo un píxel incorrecto por letra
INPUT_SIZE = 35
LATE_FRACTION = 0.9  # cumplir después del 90 % de las épocas sugiere que faltaron épocas
DIVERGED_KEY = "diverged_at_epoch"
KEY_FIELDS = ("activation", "initializer", "hidden", "optimizer", "rate", "loss", "batch")


def parameter_count(hidden: list[int], latent: int) -> int:
    sizes = [INPUT_SIZE, *hidden, latent, *hidden[::-1], INPUT_SIZE]
    return sum(inputs * outputs + outputs for inputs, outputs in zip(sizes, sizes[1:]))


def seed_statistics(run_directory: Path) -> dict:
    """Resultado de una semilla: cumple al final, primera época que cumple, recaídas, divergencia."""
    summary = json.loads((run_directory / "summary.json").read_text())
    history_path = run_directory / "history.csv"
    rows = list(csv.DictReader(history_path.open())) if history_path.is_file() else []
    within = [int(row["max_pixel_errors"]) <= TARGET for row in rows]
    diverged = DIVERGED_KEY in summary
    first = within.index(True) if True in within else None
    return {
        "final": bool(within) and within[-1] and not diverged,
        "first_epoch": None if first is None else int(rows[first]["epoch"]),
        # Divergir después de cumplir es la peor recaída: el último tramo cuenta como fuera.
        "relapse": None if first is None else 1 - sum(within[first:]) / (len(within[first:]) + diverged),
        "diverged": diverged,
    }


def collect(directories: list[Path]) -> list[dict]:
    groups: dict[tuple, list[Path]] = defaultdict(list)
    configs: dict[tuple, dict] = {}
    for directory in directories:
        for config_path in sorted(directory.glob("*/config.json")):
            config = json.loads(config_path.read_text())
            model, training = config["model"], config["training"]
            key = (
                model["activation"],
                model["initializer"],
                tuple(model["hidden_layers"]),
                training["optimizer"]["type"],
                training["optimizer"]["learning_rate"],
                model["loss"],
                training["batch_size"],
            )
            groups[key].append(config_path.parent)
            configs[key] = config

    rows = []
    for key, run_directories in groups.items():
        row = dict(zip(KEY_FIELDS, key))
        config = configs[key]
        steps_per_epoch = math.ceil(32 / row["batch"])
        stats = [seed_statistics(directory) for directory in run_directories]
        firsts = sorted(s["first_epoch"] * steps_per_epoch for s in stats if s["first_epoch"] is not None)
        relapses = [s["relapse"] for s in stats if s["relapse"] is not None]
        quartiles = statistics.quantiles(firsts, n=4) if len(firsts) >= 2 else [None] * 3
        late_step = LATE_FRACTION * config["training"]["epochs"] * steps_per_epoch
        row |= {
            "seeds": len(stats),
            "final": sum(s["final"] for s in stats),
            "ever": len(firsts),
            "diverged": sum(s["diverged"] for s in stats),
            "relapse": statistics.mean(relapses) if relapses else None,
            "median": statistics.median(firsts) if firsts else None,
            "q1": quartiles[0],
            "q3": quartiles[2],
            "late": any(step > late_step for step in firsts),
            "parameters": parameter_count(list(row["hidden"]), config["model"]["latent_size"]),
        }
        rows.append(row)
    return rank(rows)


def rank(rows: list[dict]) -> list[dict]:
    """Ordena por los criterios del docstring del módulo."""
    return sorted(
        rows,
        key=lambda row: (
            -row["final"],
            # En puntos porcentuales enteros: un control fallido entre ~1000 (0,01 %)
            # no debe pesar más que una convergencia varias veces más lenta.
            round(100 * row["relapse"]) if row["relapse"] is not None else 100,
            row["median"] if row["median"] is not None else float("inf"),
            row["parameters"],
        ),
    )


def as_markdown(rows: list[dict]) -> list[str]:
    header = (
        "| # | activación | inicializador | ocultas | optimizador | η | pérdida | batch | "
        "cumple al final | alguna vez | divergió | recaídas | pasos mediana (Q1–Q3) | parámetros |"
    )
    lines = [header, "|" + "---|" * 14]
    for position, row in enumerate(rows, start=1):
        steps = "—"
        if row["median"] is not None:
            spread = f" ({row['q1']:.0f}–{row['q3']:.0f})" if row["q1"] is not None else ""
            steps = f"{row['median']:.0f}{spread}" + (" ⚠" if row["late"] else "")
        relapse = "—" if row["relapse"] is None else f"{100 * row['relapse']:.2f} %"
        lines.append(
            f"| {position} | {row['activation']} | {row['initializer']} | "
            f"{'-'.join(map(str, row['hidden']))} | {row['optimizer']} | {row['rate']:g} | "
            f"{row['loss']} | {row['batch']} | {row['final']}/{row['seeds']} | "
            f"{row['ever']}/{row['seeds']} | {row['diverged']} | {relapse} | {steps} | "
            f"{row['parameters']} |"
        )
    return lines


def write_report(name: str, title: str, rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = [
        f"# {title}",
        "",
        "Latente 2, 10000 épocas, controles cada 10 épocas, 10 semillas por fila.",
        "Orden: cumple al final ↓, recaídas (redondeadas al 1 %) ↑, pasos mediana ↑, parámetros ↑.",
        "Pasos = pasos de gradiente hasta cumplir por primera vez (batch 32: un paso por época).",
        "⚠: alguna semilla cumplió recién en el último 10 % del entrenamiento (pueden haber faltado épocas).",
        "",
        *as_markdown(rows),
        "",
    ]
    path = OUTPUT / f"{name}.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print(f"{len(rows)} configuraciones -> {path.relative_to(ROOT)}")


def stage_two() -> None:
    """Una tabla por prueba, con las filas de la etapa 1 de las mismas redes como referencia."""
    reference = collect(sorted(RESULTS.glob(STAGE_ONE)))
    tests: dict[str, list[Path]] = defaultdict(list)
    for directory in sorted(STAGE_TWO.iterdir()):
        test = directory.name.rsplit("__", 1)[0]  # <prueba>__<activación>
        tests[test].append(directory)
    for test, directories in tests.items():
        rows = collect(directories)
        networks = {(row["activation"], row["hidden"]) for row in rows}
        rows += [row for row in reference if (row["activation"], row["hidden"]) in networks]
        write_report(f"etapa2_{test}", f"Ejercicio 1a, etapa 2: {test}", rank(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    if parser.parse_args().stage == 1:
        rows = collect(sorted(RESULTS.glob(STAGE_ONE)))
        if not rows:
            raise SystemExit("No hay corridas en results/ej1a_search_*; correr main.py primero")
        write_report("etapa1", "Ejercicio 1a, etapa 1: búsqueda de configuraciones", rows)
    else:
        stage_two()


if __name__ == "__main__":
    main()
