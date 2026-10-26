"""Daños realistas sobre un QR de 37×37 (negro = +1, blanco = −1).

A diferencia del ruido aleatorio de `queries.py`, que invierte cada módulo por igual,
estos daños imitan cómo se arruina un QR impreso: la tinta se gasta (negro → blanco,
nunca al revés), aparecen manchas de forma irregular, rayones y, en una foto, poco
contraste, desenfoque y una mancha semitransparente que después se binarizan.

Cada función recibe un patrón de N = 1369 valores y un generador; devuelve otro
patrón ±1 y no modifica el recibido.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .qr import SIDE

BLACK, WHITE = 1., -1.


def _bipolar(pattern):
    pattern = np.asarray(pattern, dtype=float)
    if pattern.shape != (SIDE * SIDE,) or not np.isin(pattern, (BLACK, WHITE)).all():
        raise ValueError('Se requiere un QR de 37×37 con valores ±1')
    return pattern


def fade(pattern, probability, rng):
    """Desgaste: cada módulo negro pasa a blanco con probabilidad `probability`."""
    pattern = _bipolar(pattern)
    lost = (pattern == BLACK) & (rng.random(pattern.size) < probability)
    return np.where(lost, WHITE, pattern)


def edge_fade(pattern, probability, rng):
    """Desgaste mayor sobre un borde elegido al azar, como un papel que se agarra de un
    lado: la probabilidad vale `probability` en ese borde y baja linealmente a cero en
    el opuesto (en promedio, la mitad)."""
    pattern = _bipolar(pattern)
    ramp = 1 - np.arange(SIDE) / (SIDE - 1)
    edge = int(rng.integers(4))
    weights = [ramp[:, None], ramp[::-1, None], ramp[None, :], ramp[None, ::-1]][edge]
    chance = probability * np.broadcast_to(weights, (SIDE, SIDE)).ravel()
    lost = (pattern == BLACK) & (rng.random(pattern.size) < chance)
    return np.where(lost, WHITE, pattern)


def blob_mask(coverage, rng, max_blobs=3):
    """Máscara de 1 a `max_blobs` manchas irregulares que cubren aproximadamente
    `coverage` del QR. Cada mancha crece sumando discos chicos alrededor de un centro
    que se desplaza al azar, así el borde no queda redondo ni cuadrado."""
    target = round(coverage * SIDE * SIDE)
    rows, cols = np.mgrid[:SIDE, :SIDE]
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    centers = rng.uniform(0, SIDE - 1, size=(int(rng.integers(1, max_blobs + 1)), 2))
    while mask.sum() < target:
        blob = int(rng.integers(len(centers)))
        centers[blob] = np.clip(centers[blob] + rng.normal(0, 1.5, 2), 0, SIDE - 1)
        radius = rng.uniform(1, 2.5)
        mask |= (rows - centers[blob, 0]) ** 2 + (cols - centers[blob, 1]) ** 2 <= radius ** 2
    return mask.ravel()


def stain(pattern, coverage, rng, color=BLACK):
    """Manchas irregulares de un color: negras (comida, grasa) o blancas (rasgado,
    papel decolorado)."""
    pattern = _bipolar(pattern)
    return np.where(blob_mask(coverage, rng), color, pattern)


def scratches(pattern, count, rng):
    """`count` rayones de un módulo de ancho que cruzan el QR con ángulo al azar.
    Un rayón sobre papel impreso saca la tinta: los módulos que toca quedan blancos."""
    pattern = _bipolar(pattern)
    rows, cols = np.mgrid[:SIDE, :SIDE]
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    for _ in range(int(count)):
        angle = rng.uniform(0, np.pi)
        row, col = rng.uniform(0, SIDE - 1, size=2)
        mask |= np.abs((cols - col) * np.sin(angle) - (rows - row) * np.cos(angle)) <= .5
    return np.where(mask.ravel(), WHITE, pattern)


def fade_and_stain(pattern, probability, rng, coverage=.05):
    """Desgaste con probabilidad `probability` más una mancha negra que cubre `coverage`."""
    return stain(fade(pattern, probability, rng), coverage, rng)


def photo(pattern, severity, rng, scale=8):
    """Foto de un QR gastado, binarizada después por módulo.

    Se dibuja el QR en grises (negro 0, blanco 1) con `scale` píxeles por módulo y se
    le aplica, en proporción a `severity`: tinta descolorida (el negro se acerca al
    blanco), una mancha semitransparente que oscurece, desenfoque y ruido del sensor.
    Cada módulo se vuelve a leer como el promedio de sus píxeles, con umbral en el
    punto medio entre los módulos más claros y más oscuros (el contraste cambia).
    """
    pattern = _bipolar(pattern)
    paper = np.kron((pattern.reshape(SIDE, SIDE) == WHITE).astype(float), np.ones((scale, scale)))
    ink = .7 * severity
    image = paper * (1 - ink) + ink
    spot = np.kron(blob_mask(.05 + .25 * severity, rng).reshape(SIDE, SIDE).astype(float),
                   np.ones((scale, scale)))
    image *= 1 - .6 * gaussian_filter(spot, sigma=scale)
    image = gaussian_filter(image, sigma=1.2 * scale * severity)
    image += rng.normal(0, .05 + .1 * severity, size=image.shape)
    modules = image.reshape(SIDE, scale, SIDE, scale).mean(axis=(1, 3))
    low, high = np.percentile(modules, [5, 95])
    return np.where(modules.ravel() < (low + high) / 2, BLACK, WHITE)


DAMAGES = {
    'desgaste': (fade, (.05, .1, .2, .3, .4, .5)),
    'desgaste_borde': (edge_fade, (.1, .2, .4, .6, .8, 1.)),
    'mancha_negra': (lambda p, level, rng: stain(p, level, rng, BLACK), (.02, .05, .1, .15, .2, .3)),
    'mancha_blanca': (lambda p, level, rng: stain(p, level, rng, WHITE), (.02, .05, .1, .15, .2, .3)),
    'rayones': (scratches, (1, 2, 3, 4, 6, 8)),
    'desgaste_y_mancha': (fade_and_stain, (.05, .1, .15, .2, .3, .4)),
    'foto': (photo, (.1, .2, .3, .4, .5, .6)),
}
"""Tipo de daño → (función, niveles). El nivel es la probabilidad de desgaste, la
fracción cubierta por manchas, la cantidad de rayones o la severidad de la foto."""
