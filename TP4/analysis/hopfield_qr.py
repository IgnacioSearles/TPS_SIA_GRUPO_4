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

from typing import get_args

from analysis import hopfield_qr_reader
from src.tp4.hopfield import Hopfield
from src.tp4.qr import SIDE, load_qr, pattern_to_image
from src.tp4.queries import make_queries
from src.tp4.recall_outcome import Category, classify

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
# En los gráficos cada QR se nombra con la palabra en inglés que trae su URL
# (food = menú principal, wine = vinos, beverages = cócteles y otras bebidas,
# dessert = postres, specials = especiales del día).
TITLES = {
    'menu_principal': 'food', 'vinos': 'wine', 'cocteles_y_bebidas': 'beverages',
    'postres': 'dessert', 'especiales': 'specials',
}
NOISE_LEVELS = np.round(np.arange(0, .51, .05), 2)
SEEDS = range(20)
RECALLS_PER_LEVEL = len(SEEDS) * len(QRS)  # recuperaciones por nivel de ruido
MAX_ITERATIONS = 100
CAPACITY_FRACTION = .15  # capacidad del apunte: p_max ≈ 0,15·N


CATEGORIES = list(get_args(Category))
LABELS = {
    'correct': 'Recuperación correcta', 'other_pattern': 'Otro patrón almacenado',
    'inverse': 'Patrón inverso', 'spurious': 'Estado espurio',
    'cycle': 'Ciclo', 'max_iterations': 'Tope',
}
# Paleta categórica de referencia (dataviz), en orden fijo por categoría.
COLORS = dict(zip(CATEGORIES, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']))
SEQUENTIAL = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']  # azul, claro → oscuro
UNSTABLE_COLOR = '#eb6834'
LINE_COLOR = '#2a78d6'
ENERGY_EXAMPLE_NOISES = (0, .2, .45, .5)


def load_patterns():
    return np.vstack([load_qr(DATA / name) for name in QRS.values()])


def find_query(queries, origin, noise, seed):
    return next(q for q in queries if q.origin == origin and np.isclose(q.noise, noise) and q.seed == seed)


def recall_query(network, query):
    """Recupera un patrón de consulta con el tope del barrido."""
    return network.recall(query.pattern, max_iterations=MAX_ITERATIONS)


def sweep(network, queries):
    rows, finals = [], []
    names = list(QRS)
    for query in queries:
        result = recall_query(network, query)
        outcome = classify(result.state, result.stop_reason, network.patterns_, query.origin)
        rows.append({
            'qr': names[query.origin], 'noise': query.noise, 'seed': query.seed,
            'category': outcome.category, 'hamming': outcome.hamming,
            'query_hamming': int(np.sum(query.pattern != network.patterns_[query.origin])),
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
    print('Hamming de cada estado final al QR de origen:')
    print(results.groupby(['final_state', 'qr'], sort=False).hamming.unique().to_string())


def category_counts(results):
    """Recuperaciones por nivel de ruido en cada categoría de resultado."""
    counts = pd.crosstab(results.noise, results.category)
    return counts.reindex(columns=CATEGORIES, fill_value=0)


def plot_categories(counts, path):
    """Distribución de categorías según el nivel de ruido."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fractions = counts.div(counts.sum(axis=1), axis=0)
    bottom = np.zeros(len(fractions))
    x = np.arange(len(fractions))
    for category in CATEGORIES:
        values = fractions[category].to_numpy()
        ax.bar(x, values, bottom=bottom, width=.7, color=COLORS[category],
               edgecolor='white', linewidth=2, label=LABELS[category])
        bottom += values
    ax.set_xticks(x, [_decimal(p) for p in fractions.index], fontsize=8)
    ax.set(xlabel='Nivel de ruido p', ylabel=f'Fracción de las {RECALLS_PER_LEVEL} recuperaciones', ylim=(0, 1),
           title='Categoría de resultado según el nivel de ruido')
    ax.spines[['top', 'right']].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    keep = [i for i, c in enumerate(CATEGORIES) if counts[c].sum() > 0]
    ax.legend([handles[i] for i in keep], [labels[i] for i in keep],
              loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    _save(fig, path)


def plot_examples(patterns, queries, results, finals, path, origin=0, noises=(.05, .2, .4, .5), seed=0):
    """Patrón almacenado, patrón de consulta y salida de la red, por nivel de ruido.

    `finals[i]` es el estado final de la fila con índice i de `results`."""
    name = list(QRS)[origin]
    fig, axes = plt.subplots(len(noises), 3, figsize=(7.8, 2.8 * len(noises)))
    for row, noise in zip(axes, noises):
        query = find_query(queries, origin, noise, seed).pattern
        query_hamming = int(np.sum(query != patterns[origin]))
        match = results[(results.qr == name) & np.isclose(results.noise, noise) & (results.seed == seed)]
        (index, record), = match.iterrows()
        panels = [
            ('Patrón almacenado', patterns[origin]),
            (f'Patrón de consulta\np = {_decimal(noise)} · Hamming {query_hamming}', query),
            (f'Salida: {LABELS[record.category].lower()}\n'
             f'Hamming {record.hamming} · {record.iterations} '
             f'{"iteración" if record.iterations == 1 else "iteraciones"}', finals[index]),
        ]
        for ax, (title, pattern) in zip(row, panels):
            ax.imshow(pattern_to_image(pattern), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_title(title, fontsize=8)
            ax.axis('off')
    fig.suptitle(f'QR {TITLES[name]}: original, consulta y salida de la red (semilla {seed})')
    _save(fig, path)


# --- Ortogonalidad y colapso (antes de recuperar) ---------------------------

def mixture_state(patterns):
    """Estado mezcla sign(ξ₁ + … + ξ₅); con un número impar de patrones no hay ceros."""
    return np.sign(patterns.sum(axis=0))


def orthogonality(network):
    """Solapamientos, módulos compartidos y crosstalk de cada QR sin ruido.

    Con diagonal cero, el campo de la neurona i ante el QR ν es
    hᵢ^ν = ξᵢ^ν·(1 − p/N) + Cᵢ^ν (Hebb pondría p/N en la diagonal y la red la
    anula), y la neurona es inestable cuando hᵢ^ν tiene signo opuesto a ξᵢ^ν. Se
    verifica que son exactamente las neuronas que cambian en la primera iteración.
    """
    patterns = network.patterns_
    stored, neurons = patterns.shape
    names = list(QRS)
    overlaps = pd.DataFrame(network.overlaps(), index=names, columns=names)
    shared = (patterns == patterns[0]).all(axis=0)
    mixture = mixture_state(patterns)
    rows, unstable = [], []
    for index, pattern in enumerate(patterns):
        crosstalk = network.crosstalk(index)
        aligned = pattern * crosstalk
        field = pattern * (1 - stored / neurons) + crosstalk
        is_unstable = pattern * field < 0
        final = network.recall(pattern, max_iterations=MAX_ITERATIONS, return_states=True)
        changed = final.states[min(1, final.iterations)] != pattern
        assert np.array_equal(is_unstable, changed), f'{names[index]}: inestables ≠ cambian en la 1.ª iteración'
        unstable.append(is_unstable)
        rows.append({
            'qr': names[index],
            'max_overlap_with_other': np.delete(overlaps.iloc[index].to_numpy(), index).max(),
            'mean_abs_crosstalk': np.abs(crosstalk).mean(),
            'max_abs_crosstalk': np.abs(crosstalk).max(),
            'crosstalk_on_shared': aligned[shared].mean(),
            'opposing_neurons': int(np.sum(aligned < 0)),
            'unstable_neurons': int(is_unstable.sum()),
            'unstable_on_shared': int(np.sum(is_unstable & shared)),
            'changed_first_iteration': int(changed.sum()),
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
    right.imshow(np.where(shared, 1, 0).reshape(SIDE, SIDE), cmap=matplotlib.colors.ListedColormap(['#f0efec', '#2a78d6']),
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
            rgb[mask.reshape(SIDE, SIDE)] = matplotlib.colors.to_rgb(UNSTABLE_COLOR)
        else:
            rgb = np.dstack([image] * 3)
        ax.imshow(rgb, interpolation='nearest')
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig.suptitle('Neuronas inestables: el campo hᵢ tiene signo opuesto a ξᵢ (en naranja)', fontsize=10)
    _save(fig, path)


def analyze_orthogonality(network):
    overlaps, shared, table, unstable = orthogonality(network)
    overlaps.to_csv(OUT / 'solapamientos.csv')
    table.to_csv(OUT / 'crosstalk.csv', index=False)
    plot_overlaps(overlaps, shared, OUT / 'solapamientos.png')
    plot_unstable(network, table, unstable, OUT / 'crosstalk.png')
    report_orthogonality(network, overlaps, shared, table)


# --- Iteraciones y energía -------------------------------------------------------

def _decimal(value):
    return f'{value:g}'.replace('.', ',')


def iteration_counts(results):
    """Recuperaciones por nivel de ruido según la cantidad de iteraciones."""
    return pd.crosstab(results.noise, results.iterations)


def energy_rises(grouped):
    """Por grupo de la tabla de resultados: recuperaciones, cuántas tuvieron al menos
    una subida de energía entre dos iteraciones, y subidas en total."""
    return grouped.energy_rises.agg(recuperaciones='size', con_subida=lambda r: int((r > 0).sum()),
                                    subidas='sum')


def plot_iterations(results, path):
    """Iteraciones medias según el nivel de ruido, con el rango mín.–máx."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    stats = results.groupby('noise').iterations.agg(['mean', 'min', 'max'])
    ax.fill_between(stats.index, stats['min'], stats['max'], color=LINE_COLOR, alpha=.15, linewidth=0)
    ax.plot(stats.index, stats['mean'], color=LINE_COLOR, linewidth=2, marker='o', markersize=5,
            label='Media (banda: mín.–máx.)')
    ax.set_xticks(NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS])
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set(xlabel='Nivel de ruido p', ylabel='Iteraciones hasta detenerse',
           title=f'Iteraciones según el nivel de ruido ({len(SEEDS)} semillas × {len(QRS)} QR por nivel)',
           ylim=(0, None))
    ax.grid(axis='y', color='#dddddd', linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', frameon=False)
    _save(fig, path)


def plot_energy_examples(network, queries, results, path, origin=0, seed=0):
    """Energía por iteración de una consulta del QR `origin` en cada nivel de
    ENERGY_EXAMPLE_NOISES, con la energía del QR original como referencia. Las
    recuperaciones se rehacen con la misma consulta."""
    names = list(QRS)
    reference = network.energy(network.patterns_[origin])
    fig, axes = plt.subplots(1, len(ENERGY_EXAMPLE_NOISES), figsize=(3 * len(ENERGY_EXAMPLE_NOISES), 3.6),
                             sharey=True)
    for ax, noise in zip(axes, ENERGY_EXAMPLE_NOISES):
        query = find_query(queries, origin, noise, seed)
        energies = recall_query(network, query).energies
        ax.plot(range(len(energies)), energies, color=LINE_COLOR, linewidth=2, marker='o', markersize=6,
                label='Energía de la red')
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
    rises = energy_rises(results.groupby('noise'))
    rises.to_csv(OUT / 'subidas_energia.csv')
    plot_iterations(results, OUT / 'iteraciones.png')
    plot_energy_examples(network, queries, results, OUT / 'energia_ejemplos.png')

    print('Iteraciones por nivel de ruido:')
    print(counts.to_string())
    print('Subidas de energía entre iteraciones:')
    print(rises.to_string())
    print(f'  {int((results.energy_rises > 0).sum())} de {len(results)} recuperaciones con alguna subida')
    print(f'Energía final: mín. {results.energy_final.min():.2f}, máx. {results.energy_final.max():.2f}')
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
    analyze_orthogonality(network)
    queries = make_queries(patterns, NOISE_LEVELS, SEEDS)

    results, finals = sweep(network, queries)
    results['final_state'] = label_final_states(finals)
    results.to_csv(OUT / 'resultados.csv', index=False)
    counts = category_counts(results)
    counts.to_csv(OUT / 'categorias.csv')
    plot_categories(counts, OUT / 'categorias.png')
    plot_examples(patterns, queries, results, finals, OUT / 'ejemplos.png')

    print(f'{len(results)} recuperaciones')
    print(results.stop_reason.value_counts().to_string())
    describe_final_states(results, finals)
    check_mixture(network, finals)
    print(counts.to_string())
    print(results.groupby('noise')[['hamming', 'iterations']]
          .agg(['min', 'median', 'max']).to_string())

    analyze_iterations_and_energy(network, queries, results)

    # Experimento secundario: lee original, consulta y salida de la red.
    hopfield_qr_reader.run(list(QRS), patterns, queries, results, finals, OUT)


if __name__ == '__main__':
    main()
