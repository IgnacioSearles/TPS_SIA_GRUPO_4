"""Ej. 2, experimentos de la variante sin módulos compartidos.

Antes: la red guarda los QR completos (N = 1369). Después: guarda sólo los módulos
que difieren en algún QR y, al terminar, repone los compartidos con su valor conocido.
Cada experimento compara las dos con los mismos patrones de consulta que el barrido
principal (`hopfield_qr.py`).

Uso desde TP4: python3 analysis/hopfield_qr_variante.py
Salidas (regenerables, fuera de git) en reports/ejercicio2/variante/.
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

from analysis.hopfield_qr import (MAX_ITERATIONS, NOISE_LEVELS, QRS, SEEDS, TITLES, _decimal, _save,
                                  find_query, load_patterns)
from analysis.hopfield_qr_reader import Reader
from src.tp4.hopfield import Hopfield
from src.tp4.qr import pattern_to_image
from src.tp4.queries import make_queries
from src.tp4.recall_outcome import classify
from src.tp4.shared_modules import SharedModules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ejercicio2' / 'variante'

SUBSET_SIZE = 3  # QR guardados en el barrido de subconjuntos
EXAMPLE_SUBSET = (0, 1, 2)  # menú principal, vinos y cócteles: incluye el par más parecido
EXAMPLE_NOISES = (.1, .25, .4)

# Paleta categórica de referencia (dataviz): antes en el slot 1, después en el 2 y la
# consulta sin pasar por la red en el 3.
BEFORE, AFTER, QUERY = '#2a78d6', '#eb6834', '#1baf7a'
DIVERGING = ['#2a78d6', '#f0efec', '#e34948']  # azul (−1) · gris (0) · rojo (+1)
SEQUENTIAL = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
INK, MUTED = '#1f1f1f', '#888888'


class Variant(NamedTuple):
    """Cómo pasa un patrón completo a la red (`strip`) y cómo vuelve (`restore`)."""
    label: str
    color: str
    strip: Callable
    restore: Callable


def variants(shared):
    return {
        'completo': Variant(f'antes: QR completo (N = {shared.differing.size})', BEFORE,
                            np.asarray, np.asarray),
        'sin_compartidos': Variant(f'después: sin módulos compartidos (N = {shared.count})', AFTER,
                                   shared.strip, shared.restore),
    }


def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.8)
    ax.set_axisbelow(True)


# --- Experimento 1: solapamientos -----------------------------------------------

def overlaps_by_variant(patterns, spaces):
    names = list(QRS)
    return {key: pd.DataFrame(Hopfield(space.strip(patterns)).overlaps(), index=names, columns=names)
            for key, space in spaces.items()}


def plot_overlaps(overlaps, spaces, path):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('divergente', DIVERGING)
    labels = [TITLES[name] for name in QRS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (key, table) in zip(axes, overlaps.items()):
        values = table.to_numpy()
        masked = np.ma.masked_where(np.eye(len(values), dtype=bool), values)
        image = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, '—' if i == j else f'{value:.2f}'.replace('.', ','), ha='center', va='center',
                    fontsize=9, color='white' if abs(value) > .7 and i != j else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.tick_params(axis='y', labelleft=ax is axes[0])
        off = values[~np.eye(len(values), dtype=bool)]
        ax.set_title(f'{spaces[key].label}\nfuera de la diagonal: {_decimal(round(off.min(), 2))} a '
                     f'{_decimal(round(off.max(), 2))}', fontsize=9)
    fig.colorbar(image, ax=axes, fraction=.025, pad=.02, label=r'Solapamiento $m^{\mu\nu}$ (0 = ortogonales)')
    fig.suptitle('Experimento 1: solapamiento entre los QR, antes y después')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


# --- Experimento 2: crosstalk e inestabilidad con los cinco QR ---------------------

def stability_table(patterns, spaces):
    """Por QR y variante, con los cinco guardados: crosstalk, neuronas inestables sin
    ruido y Hamming del estado final (con los compartidos repuestos) al QR."""
    rows = []
    for key, space in spaces.items():
        network = Hopfield(space.strip(patterns))
        for index, (name, pattern) in enumerate(zip(QRS, network.patterns_)):
            crosstalk = network.crosstalk(index)
            fields = network.weights_ @ pattern
            final = space.restore(network.recall(pattern, max_iterations=MAX_ITERATIONS).state)
            rows.append({'variant': key, 'qr': name, 'neurons': len(pattern),
                         'mean_abs_crosstalk': np.abs(crosstalk).mean(),
                         'unstable_neurons': int(np.sum(pattern * fields < 0)),
                         'final_hamming': int(np.sum(final != patterns[index]))})
    return pd.DataFrame(rows)


def _paired_bars(ax, table, column, spaces, label_format='{:d}'):
    names = list(QRS)
    y = np.arange(len(names))
    height = .38
    for offset, (key, space) in zip((-height / 2, height / 2), spaces.items()):
        values = table[table.variant == key].set_index('qr').loc[names, column].to_numpy()
        bars = ax.barh(y + offset, values, height=height - .04, color=space.color, label=space.label)
        ax.bar_label(bars, [label_format.format(v).replace('.', ',') for v in values],
                     padding=3, fontsize=8, color=INK)
    ax.set_yticks(y, [TITLES[name] for name in names], fontsize=8)
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#dddddd', linewidth=.8)
    ax.set_axisbelow(True)


def plot_stability(table, spaces, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    _paired_bars(axes[0], table, 'mean_abs_crosstalk', spaces, '{:.2f}')
    axes[0].axvline(1, color=MUTED, linestyle='--', linewidth=1)
    axes[0].text(1.03, 4.45, 'señal = 1', color=MUTED, fontsize=8)
    axes[0].set(xlabel='|crosstalk| medio por neurona', title='Interferencia de los otros QR')
    _paired_bars(axes[1], table, 'unstable_neurons', spaces)
    axes[1].set(xlabel='Neuronas inestables sin ruido', title='Módulos que la red da vuelta en el 1.er paso')
    _paired_bars(axes[2], table, 'final_hamming', spaces)
    axes[2].set(xlabel='Módulos mal en el estado final (de 1369)', title='Distancia del estado final al QR')
    for ax in axes:
        ax.set_xlim(0, ax.get_xlim()[1] * 1.15)  # lugar para las etiquetas de las barras
        ax.title.set_fontsize(10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Experimento 2: cada QR sin ruido, con los cinco guardados')
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --- Experimento 3: capacidad -------------------------------------------------

def stable_subsets(patterns, spaces):
    """Por variante y tamaño k: cuántos subconjuntos de k QR quedan con todos sus
    patrones estables (ninguna neurona cambia al recuperarlos sin ruido)."""
    rows = []
    for key, space in spaces.items():
        for size in range(2, len(patterns) + 1):
            subsets = list(combinations(range(len(patterns)), size))
            stable = sum(all(network.recall(pattern, max_iterations=MAX_ITERATIONS).iterations == 0
                             for pattern in network.patterns_)
                         for network in (Hopfield(space.strip(patterns[list(s)])) for s in subsets))
            rows.append({'variant': key, 'stored': size, 'subsets': len(subsets), 'all_stable': stable})
    return pd.DataFrame(rows)


def plot_subsets(table, spaces, path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sizes = sorted(table.stored.unique())
    x = np.arange(len(sizes))
    width = .38
    for offset, (key, space) in zip((-width / 2, width / 2), spaces.items()):
        rows = table[table.variant == key].set_index('stored').loc[sizes]
        bars = ax.bar(x + offset, rows.all_stable / rows.subsets, width=width - .04, color=space.color,
                      label=space.label)
        ax.bar_label(bars, [f'{s}/{n}' for s, n in zip(rows.all_stable, rows.subsets)],
                     padding=3, fontsize=8, color=INK)
    ax.set_xticks(x, [f'{k} QR' for k in sizes])
    ax.set(xlabel='QR guardados a la vez', ylabel='Fracción de subconjuntos\ncon todos los QR estables',
           ylim=(0, 1.15), title='Experimento 3: cuántos QR puede guardar Hebb')
    _style(ax)
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    _save(fig, path)


# --- Experimentos 4 a 6: barrido de ruido y lector ---------------------------------

def sweep(patterns, queries, spaces, reader):
    """Barrido sincrónico: los cinco QR juntos y cada terna, en las dos variantes.

    `hamming` es la distancia al QR completo después de reponer los módulos
    compartidos; `output_matches` dice si el lector lee la URL original en esa salida.
    Las columnas `repaired_query_*` miden la consulta con sólo los módulos compartidos
    repuestos, sin pasar por la red: lo que la variante corrige gratis, sin Hopfield.
    """
    names = list(QRS)
    repair = spaces['sin_compartidos']
    originals = [reader.read(pattern).text for pattern in patterns]
    groups = [tuple(range(len(patterns)))] + list(combinations(range(len(patterns)), SUBSET_SIZE))
    rows = []
    for key, space in spaces.items():
        for subset in groups:
            network = Hopfield(space.strip(patterns[list(subset)]))
            for query in (q for q in queries if q.origin in subset):
                result = network.recall(space.strip(query.pattern), max_iterations=MAX_ITERATIONS)
                outcome = classify(result.state, result.stop_reason, network.patterns_,
                                   subset.index(query.origin))
                output = space.restore(result.state)
                repaired = repair.restore(repair.strip(query.pattern))
                rows.append({
                    'variant': key, 'stored': len(subset), 'subset': '+'.join(names[i] for i in subset),
                    'qr': names[query.origin], 'noise': query.noise, 'seed': query.seed,
                    'category': outcome.category, 'iterations': result.iterations,
                    'query_hamming': int(np.sum(query.pattern != patterns[query.origin])),
                    'hamming': int(np.sum(output != patterns[query.origin])),
                    'repaired_query_hamming': int(np.sum(repaired != patterns[query.origin])),
                    'query_matches': reader.read(query.pattern).text == originals[query.origin],
                    'repaired_query_matches': reader.read(repaired).text == originals[query.origin],
                    'output_matches': reader.read(output).text == originals[query.origin],
                })
    return pd.DataFrame(rows)


def noise_summary(results):
    """Por variante, QR guardados y nivel de ruido: fracción correcta, Hamming medio
    de la consulta y de la salida, y fracción que el lector decodifica."""
    grouped = results.groupby(['variant', 'stored', 'noise'])
    return grouped.agg(recalls=('seed', 'size'), correct=('category', lambda c: (c == 'correct').mean()),
                       query_hamming=('query_hamming', 'mean'),
                       repaired_query_hamming=('repaired_query_hamming', 'mean'), hamming=('hamming', 'mean'),
                       query_read=('query_matches', 'mean'), repaired_query_read=('repaired_query_matches', 'mean'),
                       output_read=('output_matches', 'mean'))


def _series(ax, summary, column, spaces):
    for (key, stored), rows in summary.groupby(level=['variant', 'stored'], sort=False):
        rows = rows.droplevel(['variant', 'stored'])
        ax.plot(rows.index, rows[column], color=spaces[key].color, linewidth=2,
                linestyle='-' if stored == SUBSET_SIZE else ':', marker='o' if stored == SUBSET_SIZE else 's',
                markersize=5, label=f'{spaces[key].label.split(":")[0]}, {stored} QR guardados')
    ax.set_xticks(NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS], fontsize=8)
    ax.set_xlabel('Nivel de ruido p (fracción de módulos invertidos)')
    _style(ax)


def _baselines(ax, summary, query_column, repaired_column):
    """Referencias sin red: la consulta tal cual y con sólo los compartidos repuestos."""
    rows = summary.xs(('completo', SUBSET_SIZE), level=['variant', 'stored'])
    ax.plot(rows.index, rows[query_column], color=MUTED, linewidth=1.5, linestyle='--',
            label='sin red: consulta tal cual')
    ax.plot(rows.index, rows[repaired_column], color=QUERY, linewidth=2, linestyle='--', marker='^',
            markersize=5, label='sin red: consulta con los compartidos repuestos')


def plot_recovery(summary, spaces, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6))
    _series(left, summary, 'correct', spaces)
    left.set(ylabel='Fracción de recuperaciones correctas', ylim=(-.03, 1.03),
             title='Recupera exactamente el QR de origen')
    left.annotate('antes (3 y 5 QR) y después con 5 QR:\nninguna correcta en ningún nivel', xy=(.05, 0),
                  xytext=(.02, .25), fontsize=8, color=INK, arrowprops={'arrowstyle': '->', 'color': MUTED})
    _series(right, summary, 'hamming', spaces)
    _baselines(right, summary, 'query_hamming', 'repaired_query_hamming')
    right.set(ylabel='Módulos distintos del QR de origen (media, de 1369)', ylim=(0, None),
              title='Cuánto se aleja la salida del QR')
    right.legend(loc='upper left', frameon=False, fontsize=8)
    fig.suptitle('Experimento 4: barrido de ruido, antes y después')
    _save(fig, path)


def plot_triples(results, overlaps, path):
    """Fracción correcta de cada terna en la variante, ordenada por el mayor
    solapamiento entre dos de sus QR (en los módulos que difieren)."""
    names = list(QRS)
    table = overlaps['sin_compartidos']
    rows = results[(results.variant == 'sin_compartidos') & (results.stored == SUBSET_SIZE)]
    correct = rows.assign(ok=rows.category == 'correct').pivot_table(index='subset', columns='noise',
                                                                      values='ok', aggfunc='mean')
    worst = {subset: max(table.loc[a, b] for a, b in combinations(subset.split('+'), 2))
             for subset in correct.index}
    correct = correct.loc[sorted(correct.index, key=worst.get)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('azul', ['#ffffff'] + SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.imshow(correct.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for (i, j), value in np.ndenumerate(correct.to_numpy()):
        ax.text(j, i, f'{value:.2f}'.replace('.', ','), ha='center', va='center', fontsize=7,
                color='white' if value > .6 else INK)
    ax.set_xticks(range(len(correct.columns)), [_decimal(p) for p in correct.columns], fontsize=8)
    ax.set_yticks(range(len(correct)),
                  [f'{" · ".join(TITLES[n] for n in s.split("+"))}  (máx. m = {worst[s]:.2f})'.replace('0.', '0,')
                   for s in correct.index], fontsize=8)
    ax.set_xlabel('Nivel de ruido p')
    ax.set_title('Experimento 5: fracción correcta por terna, después (sin módulos compartidos)\n'
                 'ordenadas por el solapamiento del par más parecido', fontsize=10)
    _save(fig, path)


def plot_reader(summary, spaces, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    _series(ax, summary, 'output_read', spaces)
    _baselines(ax, summary, 'query_read', 'repaired_query_read')
    ax.set(ylabel='Fracción que el lector lee con la URL correcta', ylim=(-.03, 1.03),
           title='Experimento 6: ¿un lector de QR real (OpenCV) lee el resultado?')
    ax.legend(loc='center right', frameon=False, fontsize=8)
    _save(fig, path)


# --- Experimento 7: ejemplos visuales --------------------------------------------

def plot_examples(patterns, queries, spaces, path, origin=0, seed=0):
    subset = list(EXAMPLE_SUBSET)
    networks = {key: Hopfield(space.strip(patterns[subset])) for key, space in spaces.items()}
    fig, axes = plt.subplots(len(EXAMPLE_NOISES), 2 + len(spaces), figsize=(10.4, 3.4 * len(EXAMPLE_NOISES)))
    for row, noise in zip(axes, EXAMPLE_NOISES):
        query = find_query(queries, origin, noise, seed).pattern
        panels = [('Original', patterns[origin]),
                  (f'Consulta, p = {_decimal(noise)}\n{int(np.sum(query != patterns[origin]))} módulos mal', query)]
        for key, space in spaces.items():
            output = space.restore(networks[key].recall(space.strip(query), max_iterations=MAX_ITERATIONS).state)
            panels.append((f'{space.label.split(":")[0].capitalize()}\n'
                           f'{int(np.sum(output != patterns[origin]))} módulos mal', output))
        for ax, (title, pattern) in zip(row, panels):
            ax.imshow(pattern_to_image(pattern), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_title(title, fontsize=9)
            ax.axis('off')
    stored = ', '.join(TITLES[list(QRS)[i]] for i in subset)
    fig.suptitle(f'Experimento 7: QR {TITLES[list(QRS)[origin]]} guardado junto a {stored.split(", ", 1)[1]} '
                 f'(semilla {seed})\nantes = QR completo · después = sin módulos compartidos', fontsize=10)
    _save(fig, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns()
    shared = SharedModules(patterns)
    spaces = variants(shared)
    queries = make_queries(patterns, NOISE_LEVELS, SEEDS)
    print(f'Módulos compartidos: {shared.differing.size - shared.count}; que difieren: {shared.count}')

    overlaps = overlaps_by_variant(patterns, spaces)
    pd.concat(overlaps, names=['variant']).to_csv(OUT / 'solapamientos.csv')
    plot_overlaps(overlaps, spaces, OUT / '1_solapamientos.png')
    for key, table in overlaps.items():
        print(f'Solapamientos ({key}):\n{table.round(3).to_string()}')

    stability = stability_table(patterns, spaces)
    stability.to_csv(OUT / 'estabilidad.csv', index=False)
    plot_stability(stability, spaces, OUT / '2_estabilidad.png')
    print(stability.round(3).to_string(index=False))

    subsets = stable_subsets(patterns, spaces)
    subsets.to_csv(OUT / 'subconjuntos.csv', index=False)
    plot_subsets(subsets, spaces, OUT / '3_capacidad.png')
    print(subsets.pivot(index='stored', columns='variant', values='all_stable').to_string())

    results = sweep(patterns, queries, spaces, Reader())
    results.to_csv(OUT / 'resultados.csv', index=False)
    summary = noise_summary(results)
    summary.to_csv(OUT / 'resumen_ruido.csv')
    categories = pd.crosstab([results.variant, results.stored, results.noise], results.category)
    categories.to_csv(OUT / 'categorias.csv')
    plot_recovery(summary, spaces, OUT / '4_barrido_ruido.png')
    plot_triples(results, overlaps, OUT / '5_ternas.png')
    plot_reader(summary, spaces, OUT / '6_lector.png')
    print(summary.round(3).to_string())
    print(categories.to_string())

    plot_examples(patterns, queries, spaces, OUT / '7_ejemplos.png')


if __name__ == '__main__':
    main()
