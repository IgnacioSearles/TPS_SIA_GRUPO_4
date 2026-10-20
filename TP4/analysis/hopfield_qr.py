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


# --- Ortogonalidad y colapso (antes de recuperar) ---------------------------

CAPACITY_FRACTION = .15  # capacidad del apunte: p_max ≈ 0,15·N
SEQUENTIAL = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']  # azul, claro → oscuro
UNSTABLE_COLOR = '#eb6834'


def mixture_state(patterns):
    """Estado mezcla sign(ξ₁ + … + ξ₅); con un número impar de patrones no hay ceros."""
    return np.sign(patterns.sum(axis=0))


def orthogonality(network):
    """Solapamientos, módulos compartidos y crosstalk de cada QR sin ruido.

    Una neurona es inestable cuando el crosstalk tiene signo opuesto a ξᵢ^ν y lo
    supera en módulo (ξᵢ^ν·Cᵢ^ν < −1), porque hᵢ^ν = ξᵢ^ν + Cᵢ^ν. Se contrasta con
    lo que hace la red: las neuronas que cambian en la primera iteración.
    """
    patterns = network.patterns_
    names = list(QRS)
    overlaps = pd.DataFrame(network.overlaps(), index=names, columns=names)
    shared = (patterns == patterns[0]).all(axis=0)
    mixture = mixture_state(patterns)
    rows, unstable = [], []
    for index, pattern in enumerate(patterns):
        crosstalk = network.crosstalk(index)
        aligned = pattern * crosstalk
        first = network.recall(pattern, max_iterations=1)
        final = network.recall(pattern, max_iterations=MAX_ITERATIONS)
        unstable.append(aligned < -1)
        rows.append({
            'qr': names[index],
            'max_overlap_with_other': np.delete(overlaps.iloc[index].to_numpy(), index).max(),
            'mean_abs_crosstalk': np.abs(crosstalk).mean(),
            'max_abs_crosstalk': np.abs(crosstalk).max(),
            'crosstalk_on_shared': aligned[shared].mean(),
            'opposing_neurons': int(np.sum(aligned < 0)),
            'unstable_neurons': int(np.sum(aligned < -1)),
            'unstable_on_shared': int(np.sum((aligned < -1) & shared)),
            'changed_first_iteration': int(np.sum(first.state != pattern)),
            'stable_without_noise': final.iterations == 0 and final.stop_reason == 'stable',
            'iterations_without_noise': final.iterations,
            'falls_into_mixture': bool(np.array_equal(final.state, mixture)),
            'hamming_to_mixture': int(np.sum(pattern != mixture)),
        })
    return overlaps, shared, pd.DataFrame(rows), np.array(unstable)


def report_orthogonality(network, overlaps, shared, table):
    neurons = network.patterns_.shape[1]
    capacity = CAPACITY_FRACTION * neurons
    off_diagonal = overlaps.to_numpy()[~np.eye(len(overlaps), dtype=bool)]
    mixture = mixture_state(network.patterns_)
    check = network.recall(mixture, max_iterations=MAX_ITERATIONS)
    print('Solapamientos entre los QR (m^μν):')
    print(overlaps.round(3).to_string())
    print(f'Solapamiento fuera de la diagonal: mín {off_diagonal.min():.3f}, máx {off_diagonal.max():.3f}')
    print(f'Módulos iguales en los cinco QR: {shared.sum()} de {neurons} '
          f'({shared.mean():.1%}; {np.sum(shared & (network.patterns_[0] == 1))} negros)')
    print(f'Patrones almacenados: {len(overlaps)}; capacidad 0,15·N = {capacity:.1f}')
    print(f'Estado mezcla sign(ξ₁ + … + ξ₅): {check.stop_reason}, {check.iterations} iteraciones')
    print(table.round(3).to_string(index=False))


def check_mixture(network, finals):
    """Cuántas salidas del barrido son el estado mezcla o su inverso."""
    mixture = mixture_state(network.patterns_)
    is_mixture = (finals == mixture).all(axis=1)
    is_inverse = (finals == -mixture).all(axis=1)
    print(f'Salidas iguales al estado mezcla: {is_mixture.sum()}; a su inverso: {is_inverse.sum()}; '
          f'a otro estado: {np.sum(~is_mixture & ~is_inverse)} (de {len(finals)})')


def plot_overlaps(overlaps, shared, path):
    names = list(overlaps.index)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('azul', SEQUENTIAL)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={'width_ratios': [1.25, 1]})
    image = left.imshow(overlaps.to_numpy(), cmap=cmap, vmin=.5, vmax=1)
    for (i, j), value in np.ndenumerate(overlaps.to_numpy()):
        left.text(j, i, f'{value:.3f}'.replace('.', ','), ha='center', va='center', fontsize=9,
                  color='white' if value > .8 else '#1f1f1f')
    labels = [TITLES[name] for name in names]
    left.set_xticks(range(len(names)), labels, rotation=35, ha='right', fontsize=8)
    left.set_yticks(range(len(names)), labels, fontsize=8)
    left.set_title(r'Solapamiento $m^{\mu\nu}$ entre los QR', fontsize=10)
    fig.colorbar(image, ax=left, fraction=.046, pad=.04)
    right.imshow(np.where(shared, 1, 0).reshape(37, 37), cmap=matplotlib.colors.ListedColormap(['#f0efec', '#2a78d6']),
                 vmin=0, vmax=1, interpolation='nearest')
    right.set_title(f'Módulos iguales en los cinco QR: {shared.sum()} de {shared.size}\n'
                    '(azul: igual en los cinco; gris: difiere en alguno)', fontsize=10)
    right.axis('off')
    _save(fig, path)


def plot_unstable(network, table, unstable, path):
    names = list(QRS)
    panels = [(f'{TITLES[name]}\n{row.unstable_neurons} neuronas inestables', pattern, mask)
              for name, row, pattern, mask in zip(names, table.itertuples(), network.patterns_, unstable)]
    panels.append(('Estado mezcla\nsign(ξ₁ + … + ξ₅)', mixture_state(network.patterns_), None))
    fig, axes = plt.subplots(2, 3, figsize=(9, 7.4))
    for ax, (title, pattern, mask) in zip(axes.ravel(), panels):
        image = np.asarray(pattern_to_image(pattern), dtype=float) / 255
        rgb = np.dstack([.35 + .55 * image] * 3)  # QR atenuado para que resalten las neuronas inestables
        if mask is not None:
            rgb[mask.reshape(37, 37)] = matplotlib.colors.to_rgb(UNSTABLE_COLOR)
        else:
            rgb = np.dstack([image] * 3)
        ax.imshow(rgb, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig.suptitle('Neuronas donde el crosstalk se opone a ξᵢ y lo supera (en naranja)', fontsize=10)
    _save(fig, path)


def analyze_orthogonality(network):
    overlaps, shared, table, unstable = orthogonality(network)
    overlaps.to_csv(OUT / 'solapamientos.csv')
    table.to_csv(OUT / 'crosstalk.csv', index=False)
    plot_overlaps(overlaps, shared, OUT / 'solapamientos.png')
    plot_unstable(network, table, unstable, OUT / 'crosstalk.png')
    report_orthogonality(network, overlaps, shared, table)


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
    analyze_orthogonality(network)
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
    check_mixture(network, finals)
    print(counts.to_string())
    print(results.groupby('noise')[['hamming', 'iterations']].agg(['min', 'median', 'max']).to_string())


if __name__ == '__main__':
    main()
