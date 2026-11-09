"""Lectura de filas de 5 bits del font.h de la cátedra."""

import re
from pathlib import Path

import numpy as np


def load_font(path, expected_patterns=32):
    """Devuelve (patrones N x 35, etiquetas) en orden de filas, MSB a la izquierda.

    Admite arreglos C con siete enteros por carácter (decimal, hex o binario).
    No ejecuta código C ni completa caracteres ausentes.
    """
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    rows = re.findall(r"\{([^{}]+)\}", text)
    patterns = []
    for row in rows:
        tokens = [token.strip() for token in row.split(",") if token.strip()]
        if len(tokens) != 7:
            continue
        try:
            values = [int(token, 0) if token.lower().startswith(("0x", "0b"))
                      else int(token, 10) for token in tokens]
        except ValueError as error:
            raise ValueError("Cada carácter debe contener siete enteros literales") from error
        if any(value < 0 or value > 31 for value in values):
            raise ValueError("Cada fila de font.h debe tener solamente 5 bits (0 a 31)")
        patterns.append([(value >> bit) & 1 for value in values for bit in range(4, -1, -1)])
    if len(patterns) != expected_patterns:
        raise ValueError(f"Se esperaban {expected_patterns} caracteres; se leyeron {len(patterns)}")
    # Índices en lugar de asumir un orden ASCII que el archivo podría no tener.
    return np.asarray(patterns, dtype=float), [str(i) for i in range(len(patterns))]


def flip_noise(x, probability, rng):
    """Invierte cada píxel con probabilidad p sin modificar el original."""
    if not 0 <= probability <= 1:
        raise ValueError("La probabilidad de ruido debe estar entre 0 y 1")
    return np.where(rng.random(x.shape) < probability, 1 - x, x)
