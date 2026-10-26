"""Ej. 2, a fondo: repartir los QR en varias redes de Hebb (`SplitHopfield`).

1. Esquemas: una red con los 5 QR, terna + par sin módulos compartidos (mejor y peor
   reparto), dos pares + uno con los QR completos, y una red por QR.
2. Todos los repartos posibles: ¿importa cómo se agrupan los QR?
3. Diagnóstico de la elección: cuántas redes devuelven un QR válido y si se elige bien.
4. Tasa de recuperación por QR con el reparto elegido.
5. Quién hace el trabajo: la red o la regla que elige. Con el reparto elegido, cada
   regla de elección contra comparar la consulta con los originales (Hamming, sin red).
6. Los mismos gráficos del informe principal, una red contra dos redes (c1 a c7).

Uso desde TP4: python3 analysis/hopfield_qr_dos_redes.py
Salidas (regenerables, fuera de git) en reports/ejercicio2/dos_redes/.
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

from typing import NamedTuple

from analysis.hopfield_qr import (CATEGORIES, COLORS, LABELS, MAX_ITERATIONS, NOISE_LEVELS, QRS, SEEDS, TITLES,
                                  _decimal, _save, find_query, load_patterns)
from analysis.hopfield_qr_mejoras import BLOCK_SIZES, CORNERS, WHITE, block
from analysis.hopfield_qr_reader import Reader
from src.tp4.hopfield import Hopfield
from src.tp4.qr import pattern_to_image
from src.tp4.queries import make_queries
from src.tp4.recall_outcome import classify
from src.tp4.shared_modules import SharedModules
from src.tp4.split_hopfield import SplitHopfield

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ejercicio2' / 'dos_redes'

HIGH_NOISES = (.35, .4, .45)  # donde los repartos se empiezan a diferenciar
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED = '#1f1f1f', '#888888'

# Resultado de una consulta según lo que devolvieron las redes, en orden fijo.
OUTCOMES = {
    'una_correcta': ('Una sola red devuelve un QR, y es el correcto', '#2a78d6'),
    'varias_bien': ('Varias redes devuelven un QR; el más cercano es el correcto', '#1baf7a'),
    'varias_mal': ('Varias redes devuelven un QR; el más cercano es otro', '#eb6834'),
    'una_otro': ('Una sola red devuelve un QR, pero es otro', '#eda100'),
    'ninguna': ('Ninguna red devuelve un QR', '#e87ba4'),
}


class Scheme(NamedTuple):
    label: str
    groups: tuple
    stripped: bool
    color: str
    linestyle: str


def partitions(items, sizes):
    """Todas las formas de repartir `items` en grupos con los tamaños de `sizes`
    (sin repetir repartos que sólo cambian el orden de grupos del mismo tamaño)."""
    if not sizes:
        yield ()
        return
    size, rest = sizes[0], sizes[1:]
    first = items[0]
    for others in combinations(items[1:], size - 1):
        group = (first,) + others
        remaining = [i for i in items if i not in group]
        for tail in partitions(remaining, rest):
            yield (group,) + tail


def all_partitions(count, sizes):
    """Repartos de `count` QR en grupos de tamaños `sizes` (de mayor a menor)."""
    seen, result = set(), []
    for order in combinations(range(count), sizes[0]):
        remaining = [i for i in range(count) if i not in order]
        for tail in partitions(remaining, sizes[1:]):
            key = frozenset([order, *tail])
            if key not in seen:
                seen.add(key)
                result.append((order, *tail))
    return result


def worst_overlap(patterns, groups):
    """Mayor |solapamiento| entre dos QR de un mismo grupo (0 si todos los grupos son de 1)."""
    overlaps = np.abs(patterns @ patterns.T / patterns.shape[1])
    pairs = [overlaps[a, b] for group in groups for a, b in combinations(group, 2)]
    return max(pairs, default=0.)


def _label(groups):
    return ' | '.join(' · '.join(TITLES[list(QRS)[i]] for i in group) for group in groups)


def recall_rows(patterns, split, queries, extra):
    rows = []
    for kind, level, origin, seed, query in queries:
        result = split.recall(query, max_iterations=MAX_ITERATIONS)
        rows.append({**extra, 'kind': kind, 'level': level, 'qr': list(QRS)[origin], 'origin': origin,
                     'seed': seed, 'correct': result.match == origin, 'match': result.match,
                     'stored_outputs': result.stored_outputs, 'group': result.group})
    return rows


def all_queries(patterns):
    queries = [('ruido', q.noise, q.origin, q.seed, q.pattern) for q in make_queries(patterns, NOISE_LEVELS, SEEDS)]
    for origin, pattern in enumerate(patterns):
        for size in BLOCK_SIZES:
            for corner, (bottom, right) in CORNERS.items():
                queries.append(('esquina', size, origin, list(CORNERS).index(corner),
                                block(pattern, bottom * (37 - size), right * (37 - size), size, WHITE)))
    return queries


def outcome(row):
    if row.stored_outputs == 0:
        return 'ninguna'
    if row.stored_outputs == 1:
        return 'una_correcta' if row.correct else 'una_otro'
    return 'varias_bien' if row.correct else 'varias_mal'


# Experimento 5: reglas para elegir la salida, de la que más sabe a la que menos.
CHOICES = {
    'hamming': ('Sin red: el original más parecido a la consulta', MUTED, '--'),
    'oraculo': ('Dos redes, si supiéramos qué red tiene el QR', CATEGORICAL[2], ':'),
    'desempate_hamming': ('Dos redes, desempate comparando con los originales', CATEGORICAL[1], '-'),
    'energia': ('Dos redes, la red donde la consulta tiene menor energía', CATEGORICAL[0], '-'),
    'sin_desempate': ('Dos redes, sólo si una sola devuelve un QR guardado', CATEGORICAL[4], '-'),
}


def choice_rows(patterns, shared, groups, queries):
    """Una fila por consulta de ruido: si cada regla de elección termina en el QR correcto."""
    split = SplitHopfield(patterns, groups, shared)
    by_energy = SplitHopfield(patterns, groups, shared, selection='energy')
    rows = []
    for query in queries:
        outputs = [split._restore(network.recall(split._strip(query.pattern), max_iterations=MAX_ITERATIONS).state)
                   for network in split.networks_]
        matches = [next((i for i in group if np.array_equal(patterns[i], state)), None)
                   for group, state in zip(split.groups_, outputs)]
        stored = [m for m in matches if m is not None]
        own = next(index for index, group in enumerate(split.groups_) if query.origin in group)
        rows.append({'qr': list(QRS)[query.origin], 'level': query.noise, 'seed': query.seed,
                     'hamming': int(np.argmin(np.sum(patterns != query.pattern, axis=1))) == query.origin,
                     'oraculo': matches[own] == query.origin,
                     'desempate_hamming': split.recall(query.pattern, max_iterations=MAX_ITERATIONS).match == query.origin,
                     'energia': by_energy.recall(query.pattern, max_iterations=MAX_ITERATIONS).match == query.origin,
                     'sin_desempate': stored == [query.origin],
                     'otra_red_devuelve_qr': matches[1 - own] is not None})
    return pd.DataFrame(rows)


def plot_choice(summary, path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for key, (label, color, linestyle) in CHOICES.items():
        ax.plot(summary.index, summary[key], color=color, linestyle=linestyle, linewidth=2, marker='o', markersize=5,
                label=label)
    ax.set_xticks(NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS], fontsize=8)
    ax.set(xlabel='Nivel de ruido p', ylabel='Fracción que devuelve el QR exacto', ylim=(-.03, 1.03),
           title='Experimento 5: ¿quién hace el trabajo, la red o la regla que elige?')
    _style(ax)
    ax.legend(loc='lower left', frameon=False, fontsize=8)
    _save(fig, path)


def choice_experiment(patterns, shared, groups):
    rows = choice_rows(patterns, shared, groups, make_queries(patterns, NOISE_LEVELS, SEEDS))
    rows.to_csv(OUT / 'eleccion.csv', index=False)
    summary = rows.groupby('level')[list(CHOICES) + ['otra_red_devuelve_qr']].mean()
    summary.to_csv(OUT / 'eleccion_resumen.csv')
    plot_choice(summary, OUT / '5_eleccion.png')
    print(summary.round(3).to_string())
    low = rows[rows.level <= .3]
    print(low.groupby('qr')[list(CHOICES) + ['otra_red_devuelve_qr']].mean().round(3).to_string())
    for key in ('desempate_hamming', 'energia'):
        print(f'{key}: acierta y Hamming no {int((rows[key] & ~rows.hamming).sum())}, '
              f'Hamming acierta y {key} no {int((~rows[key] & rows.hamming).sum())}, de {len(rows)}')


# --- Gráficos --------------------------------------------------------------------

def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.8)
    ax.set_axisbelow(True)


def plot_schemes(results, schemes, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for key, scheme in schemes.items():
        for ax, kind in ((left, 'ruido'), (right, 'esquina')):
            rate = results[(results.scheme == key) & (results.kind == kind)].groupby('level').correct.mean()
            ax.plot(rate.index, rate, color=scheme.color, linestyle=scheme.linestyle, linewidth=2, marker='o',
                    markersize=5, label=scheme.label)
    left.set_xticks(NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS], fontsize=8)
    left.set(xlabel='Nivel de ruido p', ylabel='Fracción que devuelve el QR exacto', ylim=(-.03, 1.03),
             title='Ruido aleatorio')
    labelled = (3, 9, 15, 21, 25, 29)
    right.set_xticks(BLOCK_SIZES, [f'{s}×{s}\n{s * s / 37 ** 2:.0%}' if s in labelled else '' for s in BLOCK_SIZES])
    right.set(xlabel='Esquina rota: lado del bloque y fracción del QR', title='Una esquina rota')
    for ax in (left, right):
        _style(ax)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.suptitle('Experimento 1: cuántas redes y cómo repartir los QR')
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_partitions(table, path):
    families = [('3+2', 'Terna + par, sin módulos compartidos'), ('2+2+1', 'Dos pares + uno, QR completos')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, (family, title) in zip(axes, families):
        rows = table[table.family == family].sort_values('worst_overlap', ascending=False)
        y = np.arange(len(rows))
        ax.barh(y, rows.high_noise, color=CATEGORICAL[0], height=.7)
        for position, row in zip(y, rows.itertuples()):
            ax.text(row.high_noise + .01, position, f'{row.high_noise:.0%} · máx. |m| {row.worst_overlap:.2f}'
                    .replace('.', ','), va='center', fontsize=7, color=INK)
        ax.set_yticks(y, rows.partition, fontsize=6.5)
        ax.set(xlim=(0, 1.3), xlabel=f'Fracción correcta con ruido p = {", ".join(_decimal(p) for p in HIGH_NOISES)}',
               title=title)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Experimento 2: todos los repartos posibles (abajo, los que juntan los QR más parecidos en una red)',
                 fontsize=10)
    _save(fig, path)


def plot_diagnosis(diagnosis, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True,
                             gridspec_kw={'width_ratios': [len(NOISE_LEVELS), len(BLOCK_SIZES)]})
    for ax, kind, xlabel in ((axes[0], 'ruido', 'Nivel de ruido p'), (axes[1], 'esquina', 'Esquina rota: lado del bloque')):
        counts = diagnosis.loc[kind]
        fractions = counts.div(counts.sum(axis=1), axis=0)
        x = np.arange(len(fractions))
        bottom = np.zeros(len(fractions))
        for key, (label, color) in OUTCOMES.items():
            values = fractions.get(key, pd.Series(0, index=fractions.index)).to_numpy()
            ax.bar(x, values, bottom=bottom, width=.75, color=color, edgecolor='white', linewidth=1.5, label=label)
            bottom += values
        names = [_decimal(v) for v in fractions.index] if kind == 'ruido' else [f'{int(v)}' for v in fractions.index]
        ax.set_xticks(x, names, fontsize=8)
        ax.set(xlabel=xlabel, ylim=(0, 1))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Fracción de las consultas')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle('Experimento 3: qué devuelven las dos redes y cómo se elige (reparto elegido)')
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_per_qr(rows, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    noise = rows[rows.kind == 'ruido']
    for origin, name in enumerate(QRS):
        rate = noise[noise.origin == origin].groupby('level').correct.mean()
        ax.plot(rate.index, rate, color=CATEGORICAL[origin], linewidth=2, marker='o', markersize=5, label=TITLES[name])
    ax.set_xticks(NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS], fontsize=8)
    ax.set(xlabel='Nivel de ruido p', ylabel='Tasa de recuperación', ylim=(-.03, 1.03),
           title='Experimento 4: tasa por QR con el reparto elegido')
    _style(ax)
    ax.legend(loc='lower left', frameon=False, fontsize=8)
    _save(fig, path)


# --- Comparación con una red: los gráficos del informe principal --------------------

ONE, TWO, QUERY = ('Una red (Hebb, 5 QR completos)', CATEGORICAL[0]), ('Dos redes', CATEGORICAL[1]), \
    ('Sin red: la consulta', CATEGORICAL[2])
EXAMPLE_NOISES = (.05, .2, .4, .5)
ENERGY_NOISES = (0, .2, .45, .5)


def compare_sweep(patterns, split, queries, reader):
    """Las 1100 consultas de ruido por una red y por las dos redes."""
    one = Hopfield(patterns)
    originals = [reader.read(p).text for p in patterns]
    rows, finals = [], {}
    for query in queries:
        single = one.recall(query.pattern, max_iterations=MAX_ITERATIONS)
        double = split.recall(query.pattern, max_iterations=MAX_ITERATIONS)
        for key, state, iterations, stop in (('una_red', single.state, single.iterations, single.stop_reason),
                                             ('dos_redes', double.state, double.iterations, double.stop_reason)):
            outcome_ = classify(state, stop, patterns, query.origin)
            rows.append({'network': key, 'qr': list(QRS)[query.origin], 'noise': query.noise, 'seed': query.seed,
                         'category': outcome_.category, 'hamming': outcome_.hamming, 'iterations': iterations,
                         'read': reader.read(state).text == originals[query.origin],
                         'query_read': reader.read(query.pattern).text == originals[query.origin]})
            finals[key, query.origin, round(query.noise, 2), query.seed] = state
    return pd.DataFrame(rows), finals


def _overlap_panel(ax, matrix, names, title):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('divergente', ['#2a78d6', '#f0efec', '#e34948'])
    masked = np.ma.masked_where(np.eye(len(matrix), dtype=bool), matrix)
    ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, '—' if i == j else f'{value:.2f}'.replace('.', ','), ha='center', va='center', fontsize=9,
                color='white' if abs(value) > .7 and i != j else INK)
    ax.set_xticks(range(len(names)), names, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title(title, fontsize=9)


def plot_compare_overlaps(patterns, split, path):
    names = [TITLES[n] for n in QRS]
    panels = [(Hopfield(patterns).overlaps(), names, 'Una red: los 5 QR completos')]
    for index, (group, network) in enumerate(zip(split.groups_, split.networks_)):
        panels.append((network.overlaps(), [names[i] for i in group],
                       f'Dos redes: red {"AB"[index]} (sin módulos compartidos)'))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6),
                             gridspec_kw={'width_ratios': [len(p[1]) for p in panels]})
    for ax, (matrix, labels, title) in zip(axes, panels):
        _overlap_panel(ax, matrix, labels, title)
    fig.suptitle('c1: solapamiento entre los QR que guarda cada red (0 = ortogonales)')
    _save(fig, path)


def unstable_counts(patterns, split):
    """Neuronas inestables de cada QR sin ruido: en la red única y en su red del reparto."""
    one = Hopfield(patterns)
    rows = []
    for origin, name in enumerate(QRS):
        pattern = patterns[origin]
        index = next(i for i, g in enumerate(split.groups_) if origin in g)
        network = split.networks_[index]
        own = split._strip(pattern)
        rows.append({'qr': name, 'una_red': int(np.sum(pattern * (one.weights_ @ pattern) < 0)),
                     'dos_redes': int(np.sum(own * (network.weights_ @ own) < 0))})
    return pd.DataFrame(rows)


def plot_compare_unstable(table, path):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    y = np.arange(len(table))
    for offset, (key, (label, color)) in zip((-.2, .2), (('una_red', ONE), ('dos_redes', TWO))):
        bars = ax.barh(y + offset, table[key], height=.36, color=color, label=label)
        ax.bar_label(bars, padding=3, fontsize=8, color=INK)
    ax.set_yticks(y, [TITLES[n] for n in table.qr], fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel='Neuronas inestables con el QR sin ruido', title='c2: neuronas que la red da vuelta sobre un QR perfecto')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    _save(fig, path)


def plot_compare_categories(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, (key, (label, _)) in zip(axes, (('una_red', ONE), ('dos_redes', TWO))):
        counts = pd.crosstab(results[results.network == key].noise, results[results.network == key].category)
        counts = counts.reindex(columns=CATEGORIES, fill_value=0)
        fractions = counts.div(counts.sum(axis=1), axis=0)
        x = np.arange(len(fractions))
        bottom = np.zeros(len(fractions))
        for category in CATEGORIES:
            values = fractions[category].to_numpy()
            ax.bar(x, values, bottom=bottom, width=.7, color=COLORS[category], edgecolor='white', linewidth=2,
                   label=LABELS[category])
            bottom += values
        ax.set_xticks(x, [_decimal(p) for p in fractions.index], fontsize=8)
        ax.set(xlabel='Nivel de ruido p', title=label, ylim=(0, 1))
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Fracción de las 100 recuperaciones')
    handles, labels = axes[1].get_legend_handles_labels()
    present = [i for i, c in enumerate(CATEGORIES) if (results.category == c).any()]
    axes[1].legend([handles[i] for i in present], [labels[i] for i in present], loc='upper left',
                   bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle('c3: categoría de resultado según el ruido')
    _save(fig, path)


def plot_compare_examples(patterns, queries, results, finals, path, origin=0, seed=0):
    fig, axes = plt.subplots(len(EXAMPLE_NOISES), 4, figsize=(10.4, 3.3 * len(EXAMPLE_NOISES)))
    name = list(QRS)[origin]
    for row, noise in zip(axes, EXAMPLE_NOISES):
        query = find_query(queries, origin, noise, seed).pattern
        panels = [('Original', patterns[origin]),
                  (f'Consulta, p = {_decimal(noise)}\n{int(np.sum(query != patterns[origin]))} módulos mal', query)]
        for key, (label, _) in (('una_red', ONE), ('dos_redes', TWO)):
            record = results[(results.network == key) & (results.qr == name) & np.isclose(results.noise, noise)
                             & (results.seed == seed)].iloc[0]
            panels.append((f'{label.split(" (")[0]}: {LABELS[record.category].lower()}\n'
                           f'{record.hamming} módulos mal', finals[key, origin, round(noise, 2), seed]))
        for ax, (title, pattern) in zip(row, panels):
            ax.imshow(pattern_to_image(pattern), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_title(title, fontsize=8)
            ax.axis('off')
    fig.suptitle(f'c4: QR {TITLES[name]}, una red contra dos redes (semilla {seed})')
    _save(fig, path)


def plot_compare_iterations(results, path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for key, (label, color) in (('una_red', ONE), ('dos_redes', TWO)):
        stats = results[results.network == key].groupby('noise').iterations.agg(['mean', 'min', 'max'])
        ax.fill_between(stats.index, stats['min'], stats['max'], color=color, alpha=.15, linewidth=0)
        ax.plot(stats.index, stats['mean'], color=color, linewidth=2, marker='o', markersize=5,
                label=f'{label}: media (banda: mín.–máx.)')
    ax.set_xticks(NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS], fontsize=8)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set(xlabel='Nivel de ruido p', ylabel='Iteraciones hasta detenerse', ylim=(0, None),
           title='c5: iteraciones según el ruido (en dos redes, las de la red elegida)')
    _style(ax)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    _save(fig, path)


def plot_compare_energy(patterns, split, queries, path, origin=0, seed=0):
    """Fila de arriba: la red única, con la energía del QR como referencia. Fila de
    abajo: cada una de las dos redes, con la energía de cada QR que guarda."""
    one = Hopfield(patterns)
    fig, axes = plt.subplots(2, len(ENERGY_NOISES), figsize=(3.2 * len(ENERGY_NOISES), 6.6))
    for column, noise in enumerate(ENERGY_NOISES):
        query = find_query(queries, origin, noise, seed).pattern
        top, bottom = axes[0, column], axes[1, column]
        energies = one.recall(query, max_iterations=MAX_ITERATIONS).energies
        top.plot(range(len(energies)), energies, color=ONE[1], linewidth=2, marker='o', markersize=6)
        top.axhline(one.energy(patterns[origin]), color=MUTED, linestyle='--', linewidth=1)
        top.set_title(f'p = {_decimal(noise)}', fontsize=10)
        for index, (group, network) in enumerate(zip(split.groups_, split.networks_)):
            curve = network.recall(split._strip(query), max_iterations=MAX_ITERATIONS).energies
            bottom.plot(range(len(curve)), curve, color=CATEGORICAL[1 + index], linewidth=2, marker='o',
                        markersize=6, label=f'red {"AB"[index]}: ' + ' · '.join(TITLES[list(QRS)[i]] for i in group))
            for i in group:
                if i == origin:
                    bottom.axhline(network.energy(split._strip(patterns[i])), color=CATEGORICAL[1 + index],
                                   linestyle='--', linewidth=1)
        for ax in (top, bottom):
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set_xlabel('Iteración', fontsize=8)
            _style(ax)
    axes[0, 0].set_ylabel('Una red: energía H')
    axes[1, 0].set_ylabel('Dos redes: energía H')
    axes[0, 0].plot([], [], color=MUTED, linestyle='--', label=f'H del QR {TITLES[list(QRS)[origin]]}')
    axes[0, 0].legend(loc='upper right', frameon=False, fontsize=7)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    axes[1, 0].legend(handles + [matplotlib.lines.Line2D([], [], color=MUTED, linestyle='--')],
                      labels + [f'H del QR {TITLES[list(QRS)[origin]]} en su red'], loc='upper right',
                      frameon=False, fontsize=7)
    fig.suptitle(f'c6: energía por iteración, QR {TITLES[list(QRS)[origin]]} (semilla {seed}): '
                 'arriba una red, abajo cada una de las dos redes')
    _save(fig, path)


def plot_compare_reader(results, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key, (label, color) in (('una_red', ONE), ('dos_redes', TWO)):
        rate = results[results.network == key].groupby('noise').read.mean()
        ax.plot(rate.index, rate, color=color, linewidth=2, marker='o', markersize=5, label=f'{label}: salida')
    rate = results[results.network == 'una_red'].groupby('noise').query_read.mean()
    ax.plot(rate.index, rate, color=QUERY[1], linewidth=2, linestyle='--', marker='^', markersize=5, label=QUERY[0])
    ax.set_xticks(NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS], fontsize=8)
    ax.set(xlabel='Nivel de ruido p', ylabel='Fracción que el lector lee bien', ylim=(-.03, 1.03),
           title='c7: ¿el lector de QR (OpenCV) lee la URL correcta?')
    _style(ax)
    ax.legend(loc='center right', frameon=False, fontsize=8)
    _save(fig, path)


def group_energies(patterns, split):
    """En cada una de las dos redes: energía de cada QR guardado y del estado mezcla del grupo."""
    rows = []
    for index, (group, network) in enumerate(zip(split.groups_, split.networks_)):
        mixture = np.sign(network.patterns_.sum(axis=0))
        mixture = np.where(mixture == 0, 1., mixture)  # un par puede empatar; se desempata a +1
        for i, pattern in zip(group, network.patterns_):
            rows.append({'network': 'AB'[index], 'state': list(QRS)[i], 'energy': network.energy(pattern),
                         'stable': network.recall(pattern, max_iterations=MAX_ITERATIONS).iterations == 0})
        rows.append({'network': 'AB'[index], 'state': 'estado_mezcla', 'energy': network.energy(mixture),
                     'stable': network.recall(mixture, max_iterations=MAX_ITERATIONS).iterations == 0})
    return pd.DataFrame(rows)


def compare_with_one_network(patterns, shared, groups):
    split = SplitHopfield(patterns, groups, shared)
    queries = make_queries(patterns, NOISE_LEVELS, SEEDS)
    results, finals = compare_sweep(patterns, split, queries, Reader())
    results.to_csv(OUT / 'comparacion_resultados.csv', index=False)
    unstable = unstable_counts(patterns, split)
    unstable.to_csv(OUT / 'comparacion_inestables.csv', index=False)
    energies = group_energies(patterns, split)
    energies.to_csv(OUT / 'comparacion_energia.csv', index=False)

    plot_compare_overlaps(patterns, split, OUT / 'c1_solapamientos.png')
    plot_compare_unstable(unstable, OUT / 'c2_inestables.png')
    plot_compare_categories(results, OUT / 'c3_categorias.png')
    plot_compare_examples(patterns, queries, results, finals, OUT / 'c4_ejemplos.png')
    plot_compare_iterations(results, OUT / 'c5_iteraciones.png')
    plot_compare_energy(patterns, split, queries, OUT / 'c6_energia.png')
    plot_compare_reader(results, OUT / 'c7_lector.png')

    print(unstable.to_string(index=False))
    print(energies.round(2).to_string(index=False))
    print(pd.crosstab([results.network, results.noise], results.category).to_string())
    print(results.groupby(['network', 'noise'])[['iterations', 'read', 'query_read']].mean().round(2).unstack('network')
          .to_string())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns().astype(float)
    shared = SharedModules(patterns)
    stripped = shared.strip(patterns)
    queries = all_queries(patterns)
    noise_queries = [q for q in queries if q[0] == 'ruido']

    # Experimento 2: todos los repartos.
    families = {'3+2': ((3, 2), True), '2+2+1': ((2, 2, 1), False)}
    rows, table = [], []
    for family, (sizes, use_stripped) in families.items():
        space = stripped if use_stripped else patterns
        for groups in all_partitions(len(patterns), sizes):
            split = SplitHopfield(patterns, groups, shared if use_stripped else None)
            found = pd.DataFrame(recall_rows(patterns, split, noise_queries, {}))
            table.append({'family': family, 'partition': _label(groups), 'groups': groups,
                          'worst_overlap': worst_overlap(space, groups),
                          'low_noise': found[found.level <= .3].correct.mean(),
                          'high_noise': found[found.level.isin(HIGH_NOISES)].correct.mean(),
                          'all_noise': found.correct.mean()})
    table = pd.DataFrame(table)
    table.drop(columns='groups').to_csv(OUT / 'repartos.csv', index=False)
    plot_partitions(table, OUT / '2_repartos.png')
    print(table.drop(columns='groups').round(3).sort_values(['family', 'high_noise']).to_string(index=False))

    # Experimento 1: esquemas.
    best = {f: table[table.family == f].sort_values('worst_overlap').iloc[0].groups for f in families}
    # El peor se elige por resultado: un par siempre es estable, así que su solapamiento
    # no indica cuán malo es un reparto.
    worst = table[table.family == '3+2'].sort_values('all_noise').iloc[0].groups
    schemes = {
        'una_red': Scheme('1 red con los 5 QR completos (el informe principal)', ((0, 1, 2, 3, 4),), False,
                          CATEGORICAL[0], '-'),
        '3+2_mejor': Scheme(f'3 + 2 sin compartidos, mejor reparto: {_label(best["3+2"])}', best['3+2'], True,
                            CATEGORICAL[1], '-'),
        '3+2_peor': Scheme(f'3 + 2 sin compartidos, peor reparto: {_label(worst)}', worst, True, CATEGORICAL[2], '-'),
        '2+2+1': Scheme(f'2 + 2 + 1 con QR completos: {_label(best["2+2+1"])}', best['2+2+1'], False,
                        CATEGORICAL[3], '-'),
        'cinco': Scheme('5 redes de 1 QR (equivale a elegir el original más cercano)', tuple((i,) for i in range(5)),
                        False, MUTED, '--'),
    }
    results = pd.concat([pd.DataFrame(recall_rows(patterns, SplitHopfield(patterns, s.groups, shared if s.stripped else None),
                                                  queries, {'scheme': key}))
                         for key, s in schemes.items()], ignore_index=True)
    results.to_csv(OUT / 'resultados.csv', index=False)
    summary = results.groupby(['scheme', 'kind', 'level']).correct.mean().unstack('scheme')
    summary.to_csv(OUT / 'resumen_esquemas.csv')
    plot_schemes(results, schemes, OUT / '1_esquemas.png')
    print(summary.round(3).to_string())

    # Experimento 3 y 4: el reparto elegido.
    chosen = results[results.scheme == '3+2_mejor'].copy()
    chosen['outcome'] = chosen.apply(outcome, axis=1)
    diagnosis = chosen.groupby(['kind', 'level']).outcome.value_counts().unstack(fill_value=0)
    diagnosis = diagnosis.reindex(columns=list(OUTCOMES), fill_value=0)
    diagnosis.to_csv(OUT / 'diagnostico.csv')
    plot_diagnosis(diagnosis, OUT / '3_diagnostico.png')
    print(diagnosis.to_string())
    plot_per_qr(chosen, OUT / '4_por_qr.png')

    choice_experiment(patterns, shared, best['3+2'])
    compare_with_one_network(patterns, shared, best['3+2'])


if __name__ == '__main__':
    main()
