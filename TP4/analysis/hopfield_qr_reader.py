"""Ej. 2, experimento secundario: ¿un lector de QR real (OpenCV) lee el original,
el patrón de consulta y la salida de la red?

Cada patrón se amplía y se le agrega un margen blanco antes de leerlo, así un fallo
de lectura se debe al contenido y no al formato. Lo llama `hopfield_qr.py`.
"""
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd

from src.tp4.qr import pattern_to_scan_image
from src.tp4.queries import noisy_query

# Niveles de ruido por debajo del primero del barrido (0,05), para ubicar dónde deja
# de leer el lector por sí solo. Sólo consultas: no pasan por la red.
FINE_NOISE_LEVELS = (.005, .01, .015, .02, .03, .04)
FINE_SEEDS = range(20)


class Reading(NamedTuple):
    """`found`: el lector ubicó los patrones localizadores. `text`: lo decodificado,
    o None si no pudo decodificar (se haya ubicado el QR o no)."""
    found: bool
    text: str | None


class Reader:
    """Lector de QR con memoria: un mismo patrón se lee una sola vez. Cada patrón
    pasa por `pattern_to_scan_image` con sus valores por defecto: 8 píxeles por
    módulo y 4 módulos de margen blanco (la zona de silencio del estándar QR)."""

    def __init__(self):
        self._detector = cv2.QRCodeDetector()
        self._cache = {}

    def read(self, pattern) -> Reading:
        key = np.asarray(pattern, dtype=np.int8).tobytes()
        if key not in self._cache:
            image = np.asarray(pattern_to_scan_image(pattern))
            text, points, _ = self._detector.detectAndDecode(image)
            self._cache[key] = Reading(points is not None, text or None)
        return self._cache[key]


def _columns(prefix, reading, original):
    return {f'{prefix}_found': reading.found,
            f'{prefix}_decoded': reading.text is not None,
            f'{prefix}_matches': original is not None and reading.text == original}


def read_all(names, patterns, queries, results, finals):
    """Lee original, consulta y salida de cada recuperación de `results`.
    `finals[i]` es el estado final de la fila i."""
    reader = Reader()
    originals = {name: reader.read(pattern).text for name, pattern in zip(names, patterns)}
    query_of = {(names[q.origin], round(float(q.noise), 2), q.seed): q.pattern for q in queries}
    rows = []
    for record, final in zip(results.itertuples(index=False), finals):
        original = originals[record.qr]
        query = reader.read(query_of[record.qr, round(float(record.noise), 2), record.seed])
        output = reader.read(final)
        rows.append({'qr': record.qr, 'noise': record.noise, 'seed': record.seed,
                     'original_decoded': original is not None,
                     **_columns('query', query, original), **_columns('output', output, original),
                     'output_text': output.text})
    return originals, pd.DataFrame(rows)


def summarize(readings):
    """Cantidad de lecturas exitosas por nivel de ruido."""
    columns = ['original_decoded', 'query_found', 'query_decoded', 'query_matches',
               'output_found', 'output_decoded', 'output_matches']
    summary = readings.groupby('noise')[columns].sum()
    summary.insert(0, 'recalls', readings.groupby('noise').size())
    return summary


def reader_threshold(names, patterns):
    """Lee consultas con poco ruido. El generador de cada una se siembra con
    (índice del QR, 100 + índice del nivel fino, semilla), distinto del barrido."""
    reader = Reader()
    rows = []
    for origin, (name, pattern) in enumerate(zip(names, patterns)):
        original = reader.read(pattern).text
        for level, noise in enumerate(FINE_NOISE_LEVELS):
            for seed in FINE_SEEDS:
                query = noisy_query(pattern, noise, np.random.default_rng((origin, 100 + level, seed)))
                rows.append({'qr': name, 'noise': noise, 'seed': seed,
                             'query_hamming': int(np.sum(query != pattern)),
                             **_columns('query', reader.read(query), original)})
    readings = pd.DataFrame(rows)
    return readings.groupby('noise').agg(
        queries=('seed', 'size'), mean_query_hamming=('query_hamming', 'mean'),
        query_found=('query_found', 'sum'), query_decoded=('query_decoded', 'sum'),
        query_matches=('query_matches', 'sum'))


def run(names, patterns, queries, results, finals, out):
    originals, readings = read_all(names, patterns, queries, results, finals)
    readings.to_csv(out / 'lector_qr.csv', index=False)
    summary = summarize(readings)
    summary.to_csv(out / 'lector_qr_resumen.csv')

    print('\nLector de QR (OpenCV):')
    for name, text in originals.items():
        print(f'  original {name}: {text!r}')
    print(summary.to_string())
    threshold = reader_threshold(names, patterns)
    threshold.to_csv(out / 'lector_qr_umbral.csv')
    print(threshold.to_string())
    decoded = readings[readings.output_decoded]
    print(f'salidas decodificadas: {len(decoded)}; textos: '
          f'{decoded.output_text.value_counts().to_dict()}')
