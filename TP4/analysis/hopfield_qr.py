"""Ej. 2: recuperación de los QR de la Trattoria Aurora con una red de Hopfield.

Uso desde TP4: python3 analysis/hopfield_qr.py
Salidas (regenerables, fuera de git) en reports/ejercicio2/.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.tp4.hopfield import Hopfield
from src.tp4.qr import load_qr, pattern_to_image
from src.tp4.recall_outcome import classify

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'hopfield'
OUT = ROOT / 'reports' / 'ejercicio2'

# Orden fijo de los patrones almacenados: el índice es el de la tabla de resultados.
QRS = {
    'menu_principal': 'food_menu_qr.png',
    'vinos': 'wine_menu_qr.png',
    'cocteles_y_bebidas': 'beverages_menu_qr.png',
    'postres': 'dessert_menu_qr.png',
    'especiales': 'specials_menu_qr.png',
}
TITLES = {
    'menu_principal': 'menú principal', 'vinos': 'vinos', 'cocteles_y_bebidas': 'cócteles y otras bebidas',
    'postres': 'postres', 'especiales': 'especiales del día',
}
NOISE_LEVELS =np.round(np.arange(0, .51, .05), 2)
SEEDS = range(20)
MAX_ITERATIONS = 100

CATEGORIES = ['correct', 'other_pattern', 'inverse', 'spurious', 'cycle', 'max_iterations']
LABELS = {
    'correct': 'Recuperación correcta', 'other_pattern': 'Otro patrón almacenado',
    'inverse': 'Patrón inverso', 'spurious': 'Estado espurio',
    'cycle': 'Ciclo', 'max_iterations': 'Tope',
}
# Paleta categórica de referencia (dataviz), en orden fijo por categoría.
COLORS = dict(zip(CATEGORIES, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']))


def load_patterns():
    return np.vstack([load_qr(DATA / name) for name in QRS.values()])


def noisy_query(pattern, noise, rng):
    """Invierte cada módulo, de forma independiente, con probabilidad `noise`."""
    flips = rng.random(len(pattern)) < noise
    return np.where(flips, -pattern, pattern)


def make_queries(patterns):
    """Los 5 × 11 × 20 patrones de consulta, generados una sola vez.

    La semilla del generador es (índice del QR, índice del nivel de ruido, semilla),
    así cada consulta se puede rehacer a partir de su fila en la tabla.
    """
    queries = []
    for origin, pattern in enumerate(patterns):
        for level, noise in enumerate(NOISE_LEVELS):
            for seed in SEEDS:
                rng = np.random.default_rng((origin, level, seed))
                queries.append((origin, noise, seed, noisy_query(pattern, noise, rng)))
    return queries


def sweep(network, queries, mode):
    rows, finals = [], []
    names = list(QRS)
    for origin, noise, seed, query in queries:
        result = network.recall(query, mode=mode, max_iterations=MAX_ITERATIONS)
        outcome = classify(result.state, result.stop_reason, network.patterns_, origin)
        rows.append({
            'qr': names[origin], 'noise': noise, 'seed': seed, 'mode': mode,
            'category': outcome.category, 'hamming': outcome.hamming,
            'query_hamming': int(np.sum(query != network.patterns_[origin])),
            'iterations': result.iterations, 'stop_reason': result.stop_reason,
            'energy_initial': result.energies[0], 'energy_final': result.energies[-1],
            'energy_rises': result.energy_rises,
        })
        finals.append(result.state)
    return pd.DataFrame(rows), np.array(finals)


def label_final_states(finals):
    """Numera los estados finales distintos (0, 1, …) en orden de primera aparición."""
    _, first, inverse = np.unique(finals, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inverse.ravel()]


def describe_final_states(results, finals):
    labels = results.final_state.to_numpy()
    states = [finals[np.flatnonzero(labels == k)[0]] for k in range(labels.max() + 1)]
    print(f'{len(states)} estados finales distintos')
    for k, state in enumerate(states):
        opposite = [j for j, other in enumerate(states) if np.array_equal(other, -state)]
        print(f'  estado {k}: {np.sum(labels == k)} recuperaciones; '
              f'opuesto del estado {opposite or "ninguno"}')
    print(pd.crosstab(results.noise, results.final_state).to_string())


def category_counts(results):
    counts = pd.crosstab(results.noise, results.category)
    return counts.reindex(columns=CATEGORIES, fill_value=0)


def plot_categories(counts, mode, path):
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bottom = np.zeros(len(fractions))
    x = np.arange(len(fractions))
    for category in CATEGORIES:
        values = fractions[category].to_numpy()
        if values.any():
            ax.bar(x, values, bottom=bottom, width=.7, color=COLORS[category],
                   edgecolor='white', linewidth=2, label=LABELS[category])
        bottom += values
    ax.set_xticks(x, [f'{p:.2f}'.replace('.', ',') for p in fractions.index])
    ax.set(xlabel='Nivel de ruido p', ylabel='Fracción de las 100 recuperaciones',
           title=f'Categoría de resultado según el nivel de ruido (modo {mode})', ylim=(0, 1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    _save(fig, path)


def plot_examples(patterns, queries, results, finals, path, origin=0, noises=(.05, .2, .4), seed=0):
    names = list(QRS)
    fig, axes = plt.subplots(len(noises), 3, figsize=(7.5, 2.7 * len(noises)))
    for row, noise in zip(axes, noises):
        index = next(i for i, (o, p, s, _) in enumerate(queries)
                     if o == origin and np.isclose(p, noise) and s == seed)
        record = results.iloc[index]
        panels = [
            ('Patrón almacenado', patterns[origin]),
            (f'Patrón de consulta\np = {noise:.2f} · Hamming {record.query_hamming}', queries[index][3]),
            (f'Salida: {LABELS[record.category].lower()}\n'
             f'Hamming {record.hamming} · {record.iterations} '
             f'{"iteración" if record.iterations == 1 else "iteraciones"}', finals[index]),
        ]
        for ax, (title, pattern) in zip(row, panels):
            ax.imshow(pattern_to_image(pattern), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_title(title.replace('.', ','), fontsize=9)
            ax.axis('off')
    fig.suptitle(f'QR {TITLES[names[origin]]}: original, consulta y salida (semilla {seed})')
    _save(fig, path)


# --- Iteraciones y energía -------------------------------------------------------
# Las funciones de esta sección separan todo por la columna `mode` de la tabla de
# resultados, así un barrido con los dos modos de actualización se compara solo.

MODE_LABELS = {'synchronous': 'sincrónico', 'asynchronous': 'asincrónico'}
MODE_COLORS = {'synchronous': '#2a78d6', 'asynchronous': '#eb6834'}
ENERGY_EXAMPLE_NOISES = (0, .2, .45, .5)


def _decimal(value):
    return f'{value:g}'.replace('.', ',')


def iteration_counts(results):
    """Recuperaciones por (modo, nivel de ruido) según la cantidad de iteraciones."""
    return pd.crosstab([results['mode'], results.noise], results.iterations)


def energy_rise_counts(results):
    """Por (modo, nivel de ruido): recuperaciones con al menos una subida de energía
    entre dos iteraciones, y subidas en total."""
    grouped = results.assign(rose=results.energy_rises > 0).groupby(['mode', 'noise'])
    return grouped.agg(recuperaciones=('rose', 'size'), con_subida=('rose', 'sum'),
                       subidas=('energy_rises', 'sum'))


def plot_iterations(results, path):
    """Iteraciones medias según el nivel de ruido, con el rango mín.–máx., por modo."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for mode, rows in results.groupby('mode', sort=False):
        stats = rows.groupby('noise').iterations.agg(['mean', 'min', 'max'])
        color = MODE_COLORS.get(mode, '#555555')
        ax.fill_between(stats.index, stats['min'], stats['max'], color=color, alpha=.15, linewidth=0)
        ax.plot(stats.index, stats['mean'], color=color, linewidth=2, marker='o', markersize=5,
                label=f'Modo {MODE_LABELS.get(mode, mode)}: media (banda: mín.–máx.)')
    ax.set_xticks(NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS])
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set(xlabel='Nivel de ruido p', ylabel='Iteraciones hasta detenerse',
           title='Iteraciones según el nivel de ruido (20 semillas × 5 QR por nivel)', ylim=(0, None))
    ax.grid(axis='y', color='#dddddd', linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', frameon=False)
    _save(fig, path)


def plot_energy_examples(network, queries, results, path, origin=0, seed=0):
    """Energía por iteración de una consulta del QR `origin` en cada nivel de
    ENERGY_EXAMPLE_NOISES, una curva por modo, con la energía del QR original como
    referencia. Las recuperaciones se rehacen con la misma consulta y semilla."""
    names = list(QRS)
    reference = network.energy(network.patterns_[origin])
    fig, axes = plt.subplots(1, len(ENERGY_EXAMPLE_NOISES), figsize=(3 * len(ENERGY_EXAMPLE_NOISES), 3.6),
                             sharey=True)
    for ax, noise in zip(axes, ENERGY_EXAMPLE_NOISES):
        query = next(q for o, p, s, q in queries if o == origin and np.isclose(p, noise) and s == seed)
        for mode in results['mode'].unique():
            energies = network.recall(query, mode=mode, max_iterations=MAX_ITERATIONS).energies
            ax.plot(range(len(energies)), energies, color=MODE_COLORS.get(mode, '#555555'),
                    linewidth=2, marker='o', markersize=6, label=f'Modo {MODE_LABELS.get(mode, mode)}')
        ax.axhline(reference, color='#888888', linewidth=1, linestyle='--', label='H del QR original')
        ax.set_title(f'p = {_decimal(noise)}', fontsize=10)
        ax.set_xlabel('Iteración')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(axis='y', color='#dddddd', linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Energía H')
    axes[0].legend(loc='upper right', frameon=False, fontsize=8)
    fig.suptitle(f'Energía por iteración: QR {TITLES[names[origin]]}, semilla {seed}')
    _save(fig, path)


def analyze_iterations_and_energy(network, queries, results):
    counts = iteration_counts(results)
    counts.to_csv(OUT / 'iteraciones.csv')
    rises = energy_rise_counts(results)
    rises.to_csv(OUT / 'subidas_energia.csv')
    plot_iterations(results, OUT / 'iteraciones.png')
    plot_energy_examples(network, queries, results, OUT / 'energia_ejemplos.png')

    print('Iteraciones por nivel de ruido:')
    print(counts.to_string())
    print('Subidas de energía entre iteraciones:')
    print(rises.to_string())
    for mode, rows in results.groupby('mode', sort=False):
        print(f'  modo {MODE_LABELS.get(mode, mode)}: {int((rows.energy_rises > 0).sum())} de '
              f'{len(rows)} recuperaciones con alguna subida')
    final = results.groupby('mode').energy_final.agg(['min', 'max'])
    print('Energía final (mín., máx.) por modo:')
    print(final.to_string())
    print('Energía de cada QR original:',
          [round(network.energy(pattern), 2) for pattern in network.patterns_])


def _save(fig, path):
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns()
    network = Hopfield(patterns)
    queries = make_queries(patterns)

    results, finals = sweep(network, queries, 'synchronous')
    results['final_state'] = label_final_states(finals)
    results.to_csv(OUT / 'resultados.csv', index=False)
    counts = category_counts(results)
    counts.to_csv(OUT / 'categorias_sincronico.csv')
    plot_categories(counts, 'sincrónico', OUT / 'categorias_sincronico.png')
    plot_examples(patterns, queries, results, finals, OUT / 'ejemplos_sincronico.png')

    print(f'{len(results)} recuperaciones en modo sincrónico')
    print(results.stop_reason.value_counts().to_string())
    describe_final_states(results, finals)
    print(counts.to_string())
    print(results.groupby('noise')[['hamming', 'iterations']].agg(['min', 'median', 'max']).to_string())

    analyze_iterations_and_energy(network, queries, results)


if __name__ == '__main__':
    main()
