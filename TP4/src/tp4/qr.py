"""Conversión entre imágenes de QR de 37×37 y patrones de ±1 (negro = +1, blanco = −1).

El vector se arma fila por fila: el módulo (fila, columna) es el elemento fila·37 + columna.
"""
from pathlib import Path

import numpy as np
from PIL import Image

SIDE = 37
BLACK, WHITE = 0, 255


def pattern_from_image(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert('L'))
    if pixels.shape != (SIDE, SIDE):
        raise ValueError(f'Se esperaba un QR de {SIDE}×{SIDE} módulos y la imagen es de '
                         f'{pixels.shape[1]}×{pixels.shape[0]}')
    if not np.isin(pixels, (BLACK, WHITE)).all():
        raise ValueError('La imagen no es binaria: sólo se admiten píxeles negros (0) y blancos (255)')
    return np.where(pixels == BLACK, 1, -1).ravel()


def load_qr(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return pattern_from_image(image)


def pattern_to_image(pattern) -> Image.Image:
    grid = np.asarray(pattern).reshape(SIDE, SIDE)
    return Image.fromarray(np.where(grid > 0, BLACK, WHITE).astype(np.uint8))


def pattern_to_scan_image(pattern, scale=8, margin=4) -> Image.Image:
    """Imagen para un lector de QR: cada módulo ocupa `scale`×`scale` píxeles y se
    agrega alrededor un margen blanco de `margin` módulos (la zona de silencio)."""
    pixels = np.kron(np.asarray(pattern_to_image(pattern)), np.ones((scale, scale), dtype=np.uint8))
    return Image.fromarray(np.pad(pixels, margin * scale, constant_values=WHITE))
