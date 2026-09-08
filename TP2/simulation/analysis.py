"""Agregación estadística de los resultados de una matriz de experimentos.

Este módulo es puro: lee filas y devuelve resúmenes. No dibuja ni escribe nada,
para que la estadística se pueda testear sin depender de matplotlib.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from scipy.stats import t as student_t

DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class GroupSummary:
    """Las mediciones de un grupo (por ejemplo, todas las corridas de `ssim`)."""

    label: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"el grupo '{self.label}' no tiene mediciones.")

    @property
    def runs(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values)

    @property
    def standard_deviation(self) -> float:
        """Dispersión entre corridas; 0 cuando hay una sola medición."""
        if self.runs < 2:
            return 0.0
        return statistics.stdev(self.values)

    @property
    def standard_error(self) -> float:
        """Incertidumbre de la media, que es lo que comparan las barras de error."""
        if self.runs < 2:
            return 0.0
        return self.standard_deviation / (self.runs ** 0.5)

    def confidence_margin(self, confidence: float = DEFAULT_CONFIDENCE) -> float:
        """Semiancho del intervalo de confianza de la media (t de Student).

        Con pocas corridas la t corrige el subestimado que daría una normal: con
        cinco mediciones el factor es 2.78, no 1.96.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence debe estar entre 0 y 1 (se recibió {confidence}).")
        if self.runs < 2:
            return 0.0
        quantile = float(student_t.ppf(0.5 + confidence / 2.0, self.runs - 1))
        return quantile * self.standard_error


def read_results(path: str | Path) -> list[dict[str, str]]:
    """Lee el `results.csv` que escribe la matriz de experimentos."""
    results_path = Path(path)
    if not results_path.is_file():
        raise FileNotFoundError(f"no existe el archivo de resultados: {results_path}")
    with results_path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize_by(
    rows: Sequence[Mapping[str, str]], group_column: str, value_column: str
) -> list[GroupSummary]:
    """Agrupa las filas por `group_column` y junta los valores de `value_column`.

    Los grupos salen ordenados de menor a mayor media, que es el orden en que se
    lee un gráfico de barras de magnitud.
    """
    if not rows:
        raise ValueError("no hay filas para resumir.")
    _require_column(rows[0], group_column)
    _require_column(rows[0], value_column)
    grouped: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row[group_column], []).append(
            _parse_number(row[value_column], value_column, index)
        )
    summaries = [GroupSummary(label, tuple(values)) for label, values in grouped.items()]
    return sorted(summaries, key=lambda summary: summary.mean)


def _require_column(row: Mapping[str, str], column: str) -> None:
    if column not in row:
        raise ValueError(
            f"la columna '{column}' no está en los resultados. "
            f"Disponibles: {', '.join(sorted(row))}."
        )


def _parse_number(raw: str, column: str, index: int) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"la fila {index} tiene un valor no numérico en '{column}': {raw!r}."
        ) from error
