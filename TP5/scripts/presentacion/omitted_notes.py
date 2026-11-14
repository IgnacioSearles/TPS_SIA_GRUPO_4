"""Estudio de notas omitidas (reports/ej2_notas_omitidas/): aciertos al interpolar, con su dispersión entre semillas.

Lee puntos_medios.csv, que guarda cada punto medio evaluado (experimento, corrida,
semilla, nota objetivo, variante y si acertó). Cada barra es el porcentaje de
aciertos; la barra de error, el desvío de ese porcentaje entre las 3 semillas.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from scripts.presentacion.common import ROOT, save

STUDY = ROOT / "reports" / "ej2_notas_omitidas" / "puntos_medios.csv"
METHOD = "vae_interpolation"
Row = dict[str, str]


@dataclass(frozen=True)
class Criterion:
    name: str
    is_hit: Callable[[Row], bool]


def _flag(row: Row, column: str) -> bool:
    return row[column] == "True"


CRITERIA = [
    Criterion("afinación ≤ 50 cents", lambda row: _flag(row, "hit")),
    Criterion("cercanía espectral", lambda row: _flag(row, "spectral_hit")),
    Criterion("ambos", lambda row: _flag(row, "hit") and _flag(row, "spectral_hit")),
]


@dataclass(frozen=True)
class Group:
    label: str
    experiment: str
    configuration: str = "base"


def make_all() -> None:
    rows = _read_study()
    hit_rate_figure(rows, [Group("Control completo", "ej2_pitch_microtonal_complete"),
                           Group("Omisión normal", "ej2_pitch_microtonal_omitted"),
                           Group("Omisión estricta", "ej2_pitch_microtonal_strict_omitted")],
                    "e2_omitidas_microtonal")
    hit_rate_figure(rows, [Group(f"Latente {size}", "ej2_pitch_latent_validation_omitted", f"latent_size={size}")
                           for size in (2, 4, 16)], "e2_omitidas_latentes")
    hit_rate_figure(rows, [Group(f"λ = {weight}".replace(".", ","), "ej2_pitch_interploss_validation_omitted",
                                 f"weight={weight}") for weight in ("0", "0.1", "1")], "e2_omitidas_perdida")


def _read_study() -> list[Row]:
    if not STUDY.is_file():
        raise FileNotFoundError(f"Falta {STUDY}: es la evidencia guardada del estudio de notas omitidas")
    with STUDY.open(encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["method"] == METHOD]


def hit_rate_figure(rows: list[Row], groups: list[Group], name: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.6))
    positions = np.arange(len(groups))
    width = 0.8 / len(CRITERIA)
    for index, criterion in enumerate(CRITERIA):
        offset = (index - (len(CRITERIA) - 1) / 2) * width
        for position, group in zip(positions, groups):
            by_seed = _rows_by_seed(rows, group)
            rates = [100 * np.mean([criterion.is_hit(row) for row in seed_rows]) for seed_rows in by_seed.values()]
            hits = sum(criterion.is_hit(row) for seed_rows in by_seed.values() for row in seed_rows)
            total = sum(len(seed_rows) for seed_rows in by_seed.values())
            mean, deviation = np.mean(rates), np.std(rates, ddof=1)
            # Un porcentaje no sale de [0, 100]: la barra de error se recorta en esos bordes.
            error = [[min(deviation, mean)], [min(deviation, 100 - mean)]]
            axis.bar(position + offset, mean, width, yerr=error, capsize=4,
                     color=f"C{index}", label=criterion.name if position == 0 else None)
            axis.text(position + offset, mean + error[1][0] + 2, f"{hits}/{total}",
                      ha="center", va="bottom", fontsize=10)
    seeds = len(_rows_by_seed(rows, groups[0]))
    axis.set_xticks(positions, [group.label for group in groups])
    axis.set_ylim(0, 112)
    axis.set_ylabel(f"aciertos (%)\nmedia ± desvío, {seeds} semillas")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(CRITERIA))
    save(figure, name)


def _rows_by_seed(rows: list[Row], group: Group) -> dict[str, list[Row]]:
    by_seed: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row["experiment"] == group.experiment and row["run"].split("__")[0] == group.configuration:
            by_seed[row["seed"]].append(row)
    if len(by_seed) < 2:
        raise ValueError(f"{group.experiment}/{group.configuration}: hacen falta al menos dos semillas")
    return dict(by_seed)
