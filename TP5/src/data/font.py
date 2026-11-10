"""Lectura de los caracteres de 7 filas x 5 bits del font.h de la cátedra."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.data.dataset import Dataset

FONT_ROWS = 7
FONT_COLUMNS = 5
MAX_ROW_VALUE = (1 << FONT_COLUMNS) - 1
DELETE_CODE = 0x7F

# Una entrada: {siete enteros}, seguida opcionalmente de "// 0x61, a".
_CHARACTER_PATTERN = re.compile(r"\{([^{}]+)\}[ \t,;}]*(?://[ \t]*(0x[0-9a-fA-F]+))?")


def load_font(path: Path) -> Dataset:
    """Cada carácter queda aplanado por filas, con el bit más significativo a la izquierda."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de fuentes: {path}")
    patterns: list[list[int]] = []
    labels: list[str] = []
    for row_text, character_code in _CHARACTER_PATTERN.findall(path.read_text(encoding="utf-8")):
        row_values = [int(token, 0) for token in row_text.split(",") if token.strip()]
        if len(row_values) != FONT_ROWS:
            raise ValueError(f"Se esperaban {FONT_ROWS} filas por carácter: {{{row_text}}}")
        if any(not 0 <= value <= MAX_ROW_VALUE for value in row_values):
            raise ValueError(f"Cada fila debe caber en {FONT_COLUMNS} bits: {{{row_text}}}")
        patterns.append(
            [
                (value >> bit) & 1
                for value in row_values
                for bit in range(FONT_COLUMNS - 1, -1, -1)
            ]
        )
        labels.append(_label(character_code, fallback_index=len(labels)))
    if not patterns:
        raise ValueError(f"{path} no contiene caracteres")
    return Dataset(
        samples=np.asarray(patterns, dtype=float),
        labels=labels,
        groups=["carácter"] * len(labels),
        image_shape=(FONT_ROWS, FONT_COLUMNS),
        is_binary=True,
    )


def _label(character_code: str, fallback_index: int) -> str:
    if not character_code:
        return str(fallback_index)
    code = int(character_code, 16)
    return "DEL" if code == DELETE_CODE else chr(code)
