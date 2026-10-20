"""Ej. 2, intentos de mejora: pseudoinversa, dos redes y daño por zonas.

Métodos que reciben un QR dañado (completo, 1369 módulos) y devuelven uno completo:
- Hebb con los cinco QR (el informe principal).
- Pseudoinversa con los cinco QR (no está en el apunte; como comparación).
- Pseudoinversa sin módulos compartidos.
- Dos redes de Hebb sin módulos compartidos: los cinco QR repartidos en una terna y
  un par; la consulta pasa por las dos y se elige la salida.
- Vecino más cercano (sin red): el QR original a menor distancia de Hamming.

Daños: ruido aleatorio (el del barrido principal), esquina rota (un bloque de una
esquina en blanco, como dice el enunciado) y mancha (un bloque negro al azar).

Uso desde TP4: python3 analysis/hopfield_qr_mejoras.py
Salidas (regenerables, fuera de git) en reports/ejercicio2/mejoras/.
"""
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Callable, NamedTuple

from analysis.hopfield_qr import MAX_ITERATIONS, NOISE_LEVELS, QRS, SEEDS, TITLES, _decimal, _save, load_patterns
from analysis.hopfield_qr_reader import Reader
from src.tp4.hopfield import Hopfield
from src.tp4.qr import SIDE, pattern_to_image
from src.tp4.queries import make_queries
from src.tp4.shared_modules import SharedModules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ejercicio2' / 'mejoras'

BLOCK_SIZES = (3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 29)
CORNERS = {'superior izquierda': (0, 0), 'superior derecha': (0, 1),
           'inferior izquierda': (1, 0), 'inferior derecha': (1, 1)}
EXAMPLE_BLOCK = 11
WHITE, BLACK = -1., 1.
INK, MUTED = '#1f1f1f', '#888888'


class Method(NamedTuple):
    label: str
    color: str
    linestyle: str
    recover: Callable  # QR dañado completo → QR completo


def _stripped_network(patterns, shared, rule):
    network = Hopfield(shared.strip(patterns), rule=rule)
    return lambda query: shared.restore(network.recall(shared.strip(query), max_iterations=MAX_ITERATIONS).state)


def best_partition(patterns, shared):
    """Terna y par (complementarios) con el menor solapamiento máximo dentro de cada
    grupo, medido en los módulos que difieren."""
    stripped = shared.strip(patterns)
    overlaps = np.abs(stripped @ stripped.T / stripped.shape[1])

    def worst(group):
        return max(overlaps[a, b] for a, b in combinations(group, 2))

    triples = combinations(range(len(patterns)), 3)
    options = [(t, tuple(i for i in range(len(patterns)) if i not in t)) for t in triples]
    return min(options, key=lambda option: max(worst(option[0]), worst(option[1])))


def two_networks(patterns, shared, groups):
    """La consulta pasa por una red de Hebb (sin compartidos) por grupo. Se queda con
    las salidas que son un QR guardado en su red y, entre ellas (o entre todas si no
    hay ninguna), con la más cercana a la consulta."""
    networks = [(patterns[list(group)], _stripped_network(patterns[list(group)], shared, 'hebb'))
                for group in groups]

    def recover(query):
        outputs = [(recall(query), stored) for stored, recall in networks]
        stored_outputs = [out for out, stored in outputs if (stored == out).all(axis=1).any()]
        candidates = stored_outputs or [out for out, _ in outputs]
        return min(candidates, key=lambda out: np.sum(out != query))
    return recover


def methods(patterns, shared, groups):
    hebb = Hopfield(patterns)
    pseudoinverse = Hopfield(patterns, rule='pseudoinverse')
    return {
        'hebb': Method('Hebb, 5 QR completos', '#2a78d6', '-',
                       lambda q: hebb.recall(q, max_iterations=MAX_ITERATIONS).state),
        'pseudoinversa': Method('Pseudoinversa, 5 QR completos', '#eb6834', '-',
                                lambda q: pseudoinverse.recall(q, max_iterations=MAX_ITERATIONS).state),
        'pseudoinversa_sin_compartidos': Method('Pseudoinversa, sin módulos compartidos', '#1baf7a', '-',
                                                _stripped_network(patterns, shared, 'pseudoinverse')),
        'dos_redes': Method('Dos redes de Hebb sin compartidos', '#eda100', '-',
                            two_networks(patterns, shared, groups)),
        'vecino': Method('Sin red: QR original más cercano', MUTED, '--',
                         lambda q: patterns[np.argmin(np.sum(patterns != q, axis=1))]),
    }


# --- Daños -----------------------------------------------------------------------

class Damaged(NamedTuple):
    kind: str
    level: float  # nivel de ruido, o lado del bloque en módulos
    origin: int
    seed: int
    detail: str
    pattern: np.ndarray


def block(pattern, top, left, size, value):
    grid = pattern.reshape(SIDE, SIDE).copy()
    grid[top:top + size, left:left + size] = value
    return grid.ravel()


def damaged_queries(patterns):
    queries = [Damaged('ruido', q.noise, q.origin, q.seed, '', q.pattern)
               for q in make_queries(patterns, NOISE_LEVELS, SEEDS)]
    for origin, pattern in enumerate(patterns):
        for size in BLOCK_SIZES:
            for corner, (bottom, right) in CORNERS.items():
                top, left = bottom * (SIDE - size), right * (SIDE - size)
                queries.append(Damaged('esquina', size, origin, 0, corner,
                                       block(pattern, top, left, size, WHITE)))
            for seed in SEEDS:
                rng = np.random.default_rng((3, origin, size, seed))
                top, left = rng.integers(0, SIDE - size + 1, size=2)
                queries.append(Damaged('mancha', size, origin, seed, f'({top}, {left})',
                                       block(pattern, top, left, size, BLACK)))
    return queries


def evaluate(patterns, queries, all_methods, reader):
    originals = [reader.read(p).text for p in patterns]
    rows = []
    for query in queries:
        base = {'kind': query.kind, 'level': query.level, 'qr': list(QRS)[query.origin], 'seed': query.seed,
                'detail': query.detail, 'query_hamming': int(np.sum(query.pattern != patterns[query.origin]))}
        rows.append({**base, 'method': 'consulta',
                     'correct': base['query_hamming'] == 0,
                     'read': reader.read(query.pattern).text == originals[query.origin]})
        for key, method in all_methods.items():
            output = method.recover(query.pattern)
            rows.append({**base, 'method': key, 'correct': np.array_equal(output, patterns[query.origin]),
                         'read': reader.read(output).text == originals[query.origin]})
    return pd.DataFrame(rows)


# --- Gráficos --------------------------------------------------------------------

QUERY_STYLE = ('Sin red: el lector lee la consulta dañada', '#e87ba4', ':')


def _series(ax, summary, column, all_methods, with_query):
    for key, method in all_methods.items():
        rows = summary.xs(key, level='method')[column]
        ax.plot(rows.index, rows, color=method.color, linestyle=method.linestyle, linewidth=2,
                marker='o', markersize=5, label=method.label)
    if with_query:
        rows = summary.xs('consulta', level='method')[column]
        label, color, style = QUERY_STYLE
        ax.plot(rows.index, rows, color=color, linestyle=style, linewidth=2, marker='^', markersize=5, label=label)
    ax.set_ylim(-.03, 1.03)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.8)
    ax.set_axisbelow(True)


def plot_damage(summary, kind, all_methods, xlabel, ticks, title, path):
    rows = summary.xs(kind, level='kind')
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    _series(left, rows, 'correct', all_methods, with_query=False)
    left.set(ylabel='Fracción', title='Devuelve exactamente el QR original', xlabel=xlabel)
    _series(right, rows, 'read', all_methods, with_query=True)
    right.set(title='Un lector de QR (OpenCV) lee la URL correcta', xlabel=xlabel)
    for ax in (left, right):
        ax.set_xticks(*ticks)
    others = rows.unstack('method').correct.drop(columns=['hebb', 'consulta'])
    if others.nunique(axis=1).max() == 1:
        left.text(.5, .55, 'Las cuatro curvas que no son Hebb\nson iguales y se superponen',
                  transform=left.transAxes, ha='center', fontsize=8, color=INK)
    handles, labels = right.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_examples(patterns, all_methods, path, origin=0):
    corner = 'superior izquierda'
    bottom, right = CORNERS[corner]
    query = block(patterns[origin], 0, 0, EXAMPLE_BLOCK, WHITE)
    panels = [('Original', patterns[origin]),
              (f'Esquina {corner} rota\n{EXAMPLE_BLOCK}×{EXAMPLE_BLOCK} módulos en blanco', query)]
    for method in all_methods.values():
        output = method.recover(query)
        name = method.label.split(' (')[0]
        panels.append((f'{name}\n{int(np.sum(output != patterns[origin]))} módulos mal', output))
    columns = (len(panels) + 1) // 2
    fig, axes = plt.subplots(2, columns, figsize=(2.8 * columns, 6.4))
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (title, pattern) in zip(axes.ravel(), panels):
        ax.imshow(pattern_to_image(pattern), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(title, fontsize=7)
        ax.axis('off')
    fig.suptitle(f'QR {TITLES[list(QRS)[origin]]} con una esquina rota: salida de cada método', fontsize=10)
    _save(fig, path)


def energy_check(patterns):
    """Energía de cada QR y del estado mezcla con cada regla, y si los QR son estables."""
    mixture = np.sign(patterns.sum(axis=0))
    rows = []
    for rule in ('hebb', 'pseudoinverse'):
        network = Hopfield(patterns, rule=rule)
        for name, pattern in zip(QRS, patterns):
            rows.append({'rule': rule, 'state': name, 'energy': network.energy(pattern),
                         'stable': network.recall(pattern, max_iterations=MAX_ITERATIONS).iterations == 0})
        rows.append({'rule': rule, 'state': 'estado_mezcla', 'energy': network.energy(mixture),
                     'stable': network.recall(mixture, max_iterations=MAX_ITERATIONS).iterations == 0})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns().astype(float)
    shared = SharedModules(patterns)
    groups = best_partition(patterns, shared)
    all_methods = methods(patterns, shared, groups)
    print('Partición de las dos redes:', [[list(QRS)[i] for i in g] for g in groups])

    energies = energy_check(patterns)
    energies.to_csv(OUT / 'energia.csv', index=False)
    print(energies.round(2).to_string(index=False))

    results = evaluate(patterns, damaged_queries(patterns), all_methods, Reader())
    results.to_csv(OUT / 'resultados.csv', index=False)
    summary = results.groupby(['kind', 'method', 'level'])[['correct', 'read']].mean()
    summary.to_csv(OUT / 'resumen.csv')
    print(summary.unstack('method').round(3).to_string())

    noise_ticks = (NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS])
    labelled = (3, 9, 15, 21, 25, 29)  # el resto queda sin rótulo para que no se pisen
    block_ticks = (BLOCK_SIZES, [f'{s}×{s}\n{s * s / SIDE ** 2:.0%}' if s in labelled else ''
                                 for s in BLOCK_SIZES])
    plot_damage(summary, 'ruido', all_methods, 'Nivel de ruido p (fracción de módulos invertidos)', noise_ticks,
                'Experimento 1: ruido aleatorio', OUT / '1_ruido.png')
    plot_damage(summary, 'esquina', all_methods, 'Lado del bloque roto (módulos) y fracción del QR', block_ticks,
                'Experimento 2: una esquina rota (bloque en blanco, las cuatro esquinas)', OUT / '2_esquina.png')
    plot_damage(summary, 'mancha', all_methods, 'Lado de la mancha (módulos) y fracción del QR', block_ticks,
                'Experimento 3: una mancha (bloque negro en un lugar al azar)', OUT / '3_mancha.png')
    plot_examples(patterns, all_methods, OUT / '4_ejemplo_esquina.png')


if __name__ == '__main__':
    main()
