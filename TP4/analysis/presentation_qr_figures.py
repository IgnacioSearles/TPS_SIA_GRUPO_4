"""Gráficos del ejercicio 2 para la presentación: QR grandes y comparaciones con pocas líneas."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np

from analysis.hopfield_qr import ENERGY_EXAMPLE_NOISES, MAX_ITERATIONS, QRS, _save, find_query, recall_query
from analysis.hopfield_qr_mejoras import WHITE, best_partition, block
from src.tp4.hopfield import Hopfield
from src.tp4.qr import SIDE
from src.tp4.queries import make_queries
from src.tp4.shared_modules import SharedModules
from src.tp4.split_hopfield import SplitHopfield

MENU_NAMES = {'menu_principal': 'menú principal', 'vinos': 'vinos', 'cocteles_y_bebidas': 'cócteles',
              'postres': 'postres', 'especiales': 'especiales'}
EXAMPLE_NOISE, EXAMPLE_SEED = .1, 0
CORNER_BLOCK = 11  # el mismo bloque que el ejemplo de mejoras/
WRONG = np.array([214, 39, 40]) / 255  # módulos mal: rojo


def _qr_image(pattern, wrong=None):
    """QR en RGB (módulo negro = +1); con `wrong`, esos módulos en rojo."""
    image = np.where(pattern.reshape(SIDE, SIDE, 1) > 0, 0., 1.).repeat(3, axis=2)
    if wrong is not None:
        image[wrong.reshape(SIDE, SIDE)] = WRONG
    return image


def _show_qr(ax, pattern, title, wrong=None):
    ax.imshow(_qr_image(pattern, wrong), interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')


def plot_queries_and_outputs(patterns, path):
    """Arriba, una consulta con ruido de cada QR; abajo, lo que devuelve la red de Hebb: siempre lo mismo."""
    network = Hopfield(patterns)
    queries = make_queries(patterns, [EXAMPLE_NOISE], [EXAMPLE_SEED])
    fig, axes = plt.subplots(2, len(patterns), figsize=(3.2 * len(patterns), 7))
    for origin, name in enumerate(QRS):
        query = find_query(queries, origin, EXAMPLE_NOISE, EXAMPLE_SEED).pattern
        output = network.recall(query, max_iterations=MAX_ITERATIONS).state
        _show_qr(axes[0, origin], query, MENU_NAMES[name])
        _show_qr(axes[1, origin], output, '')
    axes[0, 0].text(-6, SIDE / 2, 'consulta', rotation=90, ha='center', va='center', fontsize=18)
    axes[1, 0].text(-6, SIDE / 2, 'salida', rotation=90, ha='center', va='center', fontsize=18)
    _save(fig, path)


def _decimal(value):
    return f'{value:g}'.replace('.', ',')


def _recall_axis(ax, xlabel):
    ax.set(xlabel=xlabel, ylabel='Recuperaciones correctas', ylim=(-.03, 1.05))


def plot_energy(patterns, path, origin=0):
    """Energía por iteración de una consulta del menú principal en cuatro niveles de ruido, con la del QR
    original como referencia: la red siempre baja, pero por debajo del QR."""
    network = Hopfield(patterns)
    queries = make_queries(patterns, ENERGY_EXAMPLE_NOISES, [EXAMPLE_SEED])
    reference = network.energy(network.patterns_[origin])
    fig, axes = plt.subplots(1, len(ENERGY_EXAMPLE_NOISES), figsize=(14, 4.8), sharey=True)
    for ax, noise in zip(axes, ENERGY_EXAMPLE_NOISES):
        energies = recall_query(network, find_query(queries, origin, noise, EXAMPLE_SEED)).energies
        ax.plot(range(len(energies)), energies, marker='o', lw=2.5, ms=9, label='la red')
        ax.axhline(reference, color='gray', ls='--', lw=1.5, label='el QR original')
        ax.set(title=f'ruido {_decimal(noise)}', xlabel='Iteración', xticks=range(len(energies)))
    axes[0].set_ylabel('Energía')
    axes[0].legend(loc='center right')
    _save(fig, path)


def plot_random_vs_qr(summary, path):
    """Misma red y mismo tamaño: cinco patrones al azar contra los cinco QR."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for column, label in {'al_azar': '5 patrones al azar', 'qr': '5 QR del restaurante'}.items():
        ax.plot(summary.index, summary[column], marker='o', lw=3, label=label)
    _recall_axis(ax, 'Fracción de módulos invertidos')
    ax.legend(loc='center left', fontsize=18)
    _save(fig, path)


def plot_shared_fraction(summary, qr_fraction, stored, path):
    """Patrones al azar que comparten una fracción de sus módulos: la red se corta en 1/(k − 1)."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for noise in summary.columns:
        ax.plot(summary.index, summary[noise], marker='o', lw=2.5, label=f'ruido {_decimal(float(noise))}')
    threshold = 1 / (stored - 1)
    ax.axvline(threshold, color='gray', ls=':', lw=2)
    ax.text(threshold + .015, .5, f'límite\n{threshold:.0%}'.replace('%', ' %'), fontsize=16, color='gray')
    ax.axvline(qr_fraction, color='C3', ls='--', lw=2)
    ax.text(qr_fraction + .015, .5, f'los QR\n{qr_fraction:.0%}'.replace('%', ' %'), fontsize=16, color='C3')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    _recall_axis(ax, 'Módulos iguales en los 5 patrones')
    ax.legend(loc='upper right')
    _save(fig, path)


def plot_stable_subsets(table, path):
    """Cuántos subconjuntos de k QR quedan estables, con los QR completos y sin los módulos compartidos."""
    variants = {'completo': 'QR completos', 'sin_compartidos': 'sin módulos compartidos'}
    sizes = sorted(table.stored.unique())
    positions, width = np.arange(len(sizes)), .38
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for offset, (variant, label) in zip((-width / 2, width / 2), variants.items()):
        rows = table[table.variant == variant].set_index('stored').loc[sizes]
        bars = ax.bar(positions + offset, rows.all_stable / rows.subsets, width, label=label)
        ax.bar_label(bars, [f'{stable}/{total}' for stable, total in zip(rows.all_stable, rows.subsets)],
                     padding=4, fontsize=16)
    ax.set_xticks(positions, [f'{size} QR' for size in sizes])
    ax.set(xlabel='QR guardados a la vez', ylabel='Subconjuntos estables', ylim=(0, 1.2))
    ax.legend(loc='upper right')
    _save(fig, path)


def plot_shared(patterns, path):
    """Qué módulos valen lo mismo en los cinco QR."""
    differing = SharedModules(patterns).differing
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(differing.reshape(SIDE, SIDE), cmap=ListedColormap(['C0', 'lightgray']), interpolation='nearest')
    ax.axis('off')
    ax.legend(handles=[Patch(color='C0', label=f'iguales en los cinco ({(~differing).sum()})'),
                       Patch(color='lightgray', label=f'difieren ({differing.sum()})')],
              loc='upper center', bbox_to_anchor=(.5, 0), ncol=2)
    _save(fig, path)


def plot_broken_corner(patterns, path, origin=0):
    """Una esquina rota: una red con los cinco QR contra dos redes con los QR repartidos. En rojo, lo que
    cada salida tiene distinto del original."""
    shared = SharedModules(patterns)
    split = SplitHopfield(patterns, best_partition(patterns, shared), shared)
    original = patterns[origin]
    query = block(original, 0, 0, CORNER_BLOCK, WHITE)
    outputs = {'una red': Hopfield(patterns).recall(query, max_iterations=MAX_ITERATIONS).state,
               'dos redes': split.recall(query, max_iterations=MAX_ITERATIONS).state}
    fig, axes = plt.subplots(1, 2 + len(outputs), figsize=(16, 4.9))
    _show_qr(axes[0], original, 'original')
    _show_qr(axes[1], query, 'esquina rota')
    axes[1].add_patch(Rectangle((-.5, -.5), CORNER_BLOCK, CORNER_BLOCK, fill=False, ec=WRONG, lw=3, ls='--'))
    for ax, (name, output) in zip(axes[2:], outputs.items()):
        wrong = output != original
        _show_qr(ax, output, f'{name}: {int(wrong.sum())} módulos mal', wrong=wrong)
    _save(fig, path)


def plot_methods(summary, path):
    """Ruido aleatorio: una red de Hebb con los cinco QR contra dos redes de Hebb."""
    noise = summary[summary.kind == 'ruido'].pivot(index='level', columns='method', values='correct')
    lines = {'dos_redes': ('dos redes', 'C1'), 'hebb': ('una red', 'C0')}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for method, (label, color) in lines.items():
        ax.plot(noise.index, noise[method], marker='o', lw=3, color=color, label=label)
    ax.set(xlabel='Fracción de módulos invertidos', ylabel='QR recuperado exacto', ylim=(-.03, 1.05))
    ax.legend(loc='center left', fontsize=20)
    _save(fig, path)
