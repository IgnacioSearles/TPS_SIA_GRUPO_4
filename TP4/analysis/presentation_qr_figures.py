"""Gráficos del ejercicio 2 para la presentación: QR grandes y comparaciones con pocas líneas."""
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import numpy as np
import pandas as pd

from analysis.hopfield_qr import ENERGY_EXAMPLE_NOISES, MAX_ITERATIONS, QRS, _save, find_query, recall_query
from analysis.hopfield_qr_mejoras import WHITE, best_partition, block
from src.tp4.hopfield import Hopfield
from src.tp4.qr import SIDE
from src.tp4.queries import make_queries
from src.tp4.shared_modules import SharedModules
from src.tp4.split_hopfield import SplitHopfield
from src.tp4.unlearning import UnlearningTrainer

MENU_NAMES = {'menu_principal': 'food', 'vinos': 'wine', 'cocteles_y_bebidas': 'beverages',
              'postres': 'dessert', 'especiales': 'specials'}
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
    """Arriba, una consulta con ruido de cada QR; abajo, la única imagen que devuelve la red de Hebb para las
    cinco: todas las flechas llegan al mismo lugar."""
    network = Hopfield(patterns)
    queries = make_queries(patterns, [EXAMPLE_NOISE], [EXAMPLE_SEED])
    fig = plt.figure(figsize=(3.2 * len(patterns), 8))
    grid = fig.add_gridspec(2, len(patterns), height_ratios=(1, 1.25), hspace=.45)
    tops, outputs = [], []
    for origin, name in enumerate(QRS):
        query = find_query(queries, origin, EXAMPLE_NOISE, EXAMPLE_SEED).pattern
        outputs.append(network.recall(query, max_iterations=MAX_ITERATIONS).state)
        tops.append(fig.add_subplot(grid[0, origin]))
        _show_qr(tops[-1], query, MENU_NAMES[name])
    differences = max(int(np.sum(output != outputs[0])) for output in outputs)
    bottom = fig.add_subplot(grid[1, len(patterns) // 2])
    _show_qr(bottom, outputs[0], f'las 5 salidas: {differences} módulos de diferencia entre sí')
    end = bottom.get_position()
    for top in tops:
        start = top.get_position()
        fig.add_artist(FancyArrowPatch(((start.x0 + start.x1) / 2, start.y0 - .01), ((end.x0 + end.x1) / 2, end.y1 + .06),
                                       transform=fig.transFigure, arrowstyle='-|>', mutation_scale=22, lw=2,
                                       color='gray'))
    tops[0].text(-6, SIDE / 2, 'consulta', rotation=90, ha='center', va='center', fontsize=18)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


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


def plot_two_networks(summary, path):
    """Ruido aleatorio con dos redes de Hebb, según cómo se elige la salida, contra comparar con los
    originales sin red."""
    lines = {'sin_desempate': ('dos redes, sin desempate', 'C3', '-'),
             'desempate_hamming': ('dos redes, desempate con los originales', 'C1', '-'),
             'energia': ('dos redes, elige la energía', 'C0', '-'),
             'hamming': ('sin red: comparar con los originales', 'gray', '--')}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for key, (label, color, linestyle) in lines.items():
        ax.plot(summary.index, summary[key], marker='o', lw=3, color=color, linestyle=linestyle, label=label)
    _recall_axis(ax, 'Fracción de módulos invertidos')
    ax.legend(loc='lower left', fontsize=16)
    _save(fig, path)


def plot_similar_pair(results, path, pair=('menu_principal', 'vinos')):
    """Sin módulos compartidos y guardando tres QR: las ternas que incluyen a menú y vinos contra las demás."""
    triads = results[(results.variant == 'sin_compartidos') & (results.stored == 3)]
    with_pair = triads.subset.str.contains(pair[0]) & triads.subset.str.contains(pair[1])
    correct = (triads.category == 'correct').groupby([with_pair, triads.noise]).mean().unstack(0)
    counts = triads.groupby(with_pair).subset.nunique()
    lines = {False: (f'las {counts[False]} ternas sin ese par', 'C0'),
             True: (f'las {counts[True]} ternas con food y wine', 'C3')}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for has_pair, (label, color) in lines.items():
        ax.plot(correct.index, correct[has_pair], marker='o', lw=3, color=color, label=label)
    _recall_axis(ax, 'Fracción de módulos invertidos')
    ax.legend(loc='lower left', fontsize=18)
    _save(fig, path)


def plot_rules(summary, path):
    """Recuperación media con ruido (sin contar el nivel 0) de cada regla, sobre las mismas consultas nuevas.
    Comparar con los originales no es una red: es la referencia."""
    mean = summary[summary.noise > 0].groupby('method').accuracy.mean()
    bars = {'hebb': ('Hebb', 'C0'),
            'pseudoinverse_stripped': ('pseudoinversa, sin lo común', 'C1'),
            'unlearning': ('Hebb + desaprendizaje, sin lo común', 'C1'),
            'pseudoinverse_full': ('pseudoinversa, QR completo', 'C1'),
            'nearest': ('comparar con los originales', 'gray')}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    rows = ax.barh([label for label, _ in bars.values()], mean[list(bars)], color=[color for _, color in bars.values()])
    ax.bar_label(rows, [f'{value:.1%}'.replace('.', ',').replace('%', ' %') for value in mean[list(bars)]],
                 padding=6, fontsize=18)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.15), xlabel='Recuperación media con ruido')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    _save(fig, path)


def plot_overlap_before_after(overlaps, path, pair=('menu_principal', 'vinos')):
    """Solapamiento de cada par de QR con el QR completo y sin los módulos compartidos: todos bajan, el de menú
    y vinos queda arriba."""
    overlaps = overlaps.rename(columns={overlaps.columns[1]: 'qr'}).set_index(['variant', 'qr'])
    stages = {'completo': 'QR completo', 'sin_compartidos': 'sin lo común'}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for a, b in combinations(QRS, 2):
        values = [overlaps.loc[(variant, a), b] for variant in stages]
        highlighted = {a, b} == set(pair)
        ax.plot(range(len(stages)), values, marker='o', lw=4 if highlighted else 2, ms=10,
                color='C3' if highlighted else 'lightgray', zorder=3 if highlighted else 2)
        if highlighted:
            for x, value in enumerate(values):
                ax.annotate(f'food y wine: {_decimal(round(value, 2))}', (x, value), xytext=(0, 14),
                            textcoords='offset points', ha='center', va='bottom', color='C3', fontsize=17)
    ax.axhline(0, color='gray', ls=':', lw=1.5)
    ax.text(.5, .03, 'ortogonales', color='gray', ha='center', fontsize=15)
    ax.set_xticks(range(len(stages)), list(stages.values()))
    ax.set(xlim=(-.9, 1.9), ylabel='Solapamiento entre dos QR')
    _save(fig, path)


def plot_unstable(patterns, path):
    """Crosstalk del apunte: en rojo, las neuronas donde el campo de Hebb tiene el signo contrario al del QR
    guardado, o sea donde la interferencia de los otros QR le gana a la señal propia."""
    network = Hopfield(patterns)
    fig, axes = plt.subplots(1, len(patterns), figsize=(3.2 * len(patterns), 4))
    for ax, name, pattern in zip(axes, QRS, network.patterns_):
        unstable = np.sign(network.weights_ @ pattern) != pattern
        _show_qr(ax, pattern, f'{MENU_NAMES[name]}: {int(unstable.sum())}', wrong=unstable)
    _save(fig, path)


PERCENT = matplotlib.ticker.PercentFormatter(1, decimals=0, symbol=' %')


def _percent_axis(ax, xlabel):
    ax.set(xlabel=xlabel, ylabel='Recuperación exacta', ylim=(-.03, 1.05))
    ax.yaxis.set_major_formatter(PERCENT)


def plot_energy_levels(patterns, path, origin=0):
    """Izquierda, energía por iteración de una consulta del menú con tres niveles de ruido; derecha, la energía
    de cada QR guardado y de la mezcla: la mezcla está más abajo que cualquier QR, por eso la red cae ahí."""
    network = Hopfield(patterns)
    noises = (0, .2, .45)
    queries = make_queries(patterns, noises, [EXAMPLE_SEED])
    mixture = np.sign(network.patterns_.sum(axis=0))
    levels = {MENU_NAMES[name]: network.energy(pattern) for name, pattern in zip(QRS, network.patterns_)}
    levels['mezcla'] = network.energy(mixture)
    fig = plt.figure(figsize=(19, 5.2))
    grid = fig.add_gridspec(1, 5, width_ratios=(1, 1, 1, .75, 1.5), wspace=.55)
    axes = [fig.add_subplot(grid[0, 0])]
    axes += [fig.add_subplot(grid[0, i], sharey=axes[0]) for i in (1, 2)]
    for ax, noise in zip(axes, noises):
        energies = recall_query(network, find_query(queries, origin, noise, EXAMPLE_SEED)).energies
        ax.plot(range(len(energies)), energies, marker='o', lw=2.5, ms=8, color='C0', label='consulta de food')
        ax.axhline(levels[MENU_NAMES['menu_principal']], color='gray', ls='--', lw=1.5, label='food guardado')
        ax.axhline(levels['mezcla'], color='C3', ls=':', lw=2, label='mezcla')
        ax.set(title=f'ruido {_decimal(noise * 100)} %', xlabel='Iteración', xticks=range(len(energies)))
    axes[0].set_ylabel('Energía')
    axes[0].legend(loc='center right', fontsize=12)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    bars = fig.add_subplot(grid[0, 4])
    names = list(levels)
    rows = bars.barh(names, [levels[name] for name in names],
                     color=['lightgray'] * (len(names) - 1) + ['C3'])
    bars.bar_label(rows, [f'{levels[name]:,.0f}'.replace(',', '.') for name in names], label_type='center',
                   fontsize=13)
    bars.set(xlim=(min(levels.values()) * 1.05, 0), title='Energía de cada estado', xticks=[])
    bars.invert_yaxis()
    bars.spines[['bottom']].set_visible(False)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_shared_annotated(patterns, path):
    """Mapa de módulos iguales en los cinco QR, con dónde empieza a escribirse el mensaje y dónde difieren."""
    differing = SharedModules(patterns).differing.reshape(SIDE, SIDE)
    left = int(np.sum(differing[:, :19]))
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    ax.imshow(differing, cmap=ListedColormap(['C0', 'lightgray']), interpolation='nearest')
    ax.annotate('el mensaje empieza\nen esta esquina:\nlas 5 URL comparten\n72 caracteres',
                xy=(SIDE - 1.5, SIDE - 1.5), xytext=(21, 20), color='white', fontsize=14,
                arrowprops={'arrowstyle': '->', 'lw': 2, 'color': 'white'})
    ax.annotate('', xy=(-.5, -1.6), xytext=(18.5, -1.6), arrowprops={'arrowstyle': '<->', 'lw': 2},
                annotation_clip=False)
    ax.text(9, -2.6, f'{left} de los {differing.sum()} que difieren', ha='center', fontsize=14)
    ax.axis('off')
    ax.legend(handles=[Patch(color='C0', label=f'iguales en los cinco ({(~differing).sum()})'),
                       Patch(color='lightgray', label=f'difieren ({differing.sum()})')],
              loc='upper center', bbox_to_anchor=(.5, 0), ncol=2)
    _save(fig, path)


def hebb_sink_rejections(patterns, seeds=(0, 1, 2), attempts=50, learning_rate=.1):
    """Neuronas que rechazan el pozo falso de Hebb sin lo común, antes y después de desaprender.

    El pozo es el estado estable al que llega Hebb desde la mezcla mayoritaria (la mezcla misma no es
    estable sin lo común). Las redes se reentrenan con la configuración congelada de la confirmación
    (sueños, η = 0,1, 50 intentos, semillas 0/1/2); devuelve el valor con Hebb y la media de las tres."""
    active = SharedModules(patterns).strip(patterns.astype(float))
    hebb = Hopfield(active)
    sink = hebb.recall(UnlearningTrainer(active).mixture, max_iterations=MAX_ITERATIONS)
    if sink.stop_reason != 'stable':
        raise ValueError('Hebb no llega a un estado estable desde la mezcla')
    rejecting = lambda network: int(np.count_nonzero(sink.state * (network.weights_ @ sink.state) < 0))
    after = []
    for seed in seeds:
        trainer = UnlearningTrainer(active, mode='dream', learning_rate=learning_rate, seed=seed,
                                    max_iterations=MAX_ITERATIONS)
        for _ in range(attempts):
            trainer.step()
        after.append(rejecting(trainer.network))
    return rejecting(hebb), float(np.mean(after))


def plot_unlearning_training(stability, training, sink, path):
    """Qué cambia el desaprendizaje: neuronas que rechazan cada QR y el pozo falso de Hebb (campo con el
    signo contrario) antes y después, y qué pasó en los 50 intentos de cada semilla de entrenamiento.
    `sink` es el par (Hebb, después) de `hebb_sink_rejections`."""
    states = list(MENU_NAMES)
    labels = [MENU_NAMES[s] for s in MENU_NAMES] + ['pozo falso\nde Hebb']
    before = stability[stability.method == 'hebb'].set_index('state').wrong_fields.reindex(states).tolist()
    after = stability[stability.method == 'unlearning'].groupby('state').wrong_fields.mean().reindex(states).tolist()
    before.append(sink[0])
    after.append(sink[1])
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={'width_ratios': (1.7, 1)})
    positions, width = np.arange(len(labels)), .38
    for offset, values, label, color in ((-width / 2, before, 'Hebb', 'C0'),
                                         (width / 2, after, 'después de desaprender', 'C1')):
        bars = left.bar(positions + offset, values, width, label=label, color=color)
        left.bar_label(bars, [f'{v:.0f}' for v in values], padding=3, fontsize=13)
    left.set_xticks(positions, labels)
    left.set(ylabel='Neuronas que lo rechazan', ylim=(0, max(before + after) * 1.25))
    left.legend(loc='upper left')
    counts = training.groupby(['training_seed', 'action']).size().unstack(fill_value=0)
    seeds = [f'semilla {seed}' for seed in counts.index]
    kinds = {'updated': ('cayó en un espurio: se resta', 'C1'),
             'protected': ('cayó en un QR o su inverso: no se toca', 'lightgray')}
    start = np.zeros(len(counts))
    for action, (label, color) in kinds.items():
        values = counts.get(action, pd.Series(0, index=counts.index)).to_numpy()
        bars = right.barh(seeds, values, left=start, color=color, label=label)
        right.bar_label(bars, [str(v) for v in values], label_type='center', fontsize=13)
        start += values
    right.set(xlabel='Intentos (50 por semilla)', xlim=(0, 50))
    right.invert_yaxis()
    right.legend(loc='upper center', bbox_to_anchor=(.5, -.2), fontsize=12)
    _save(fig, path)


def plot_unlearning_result(by_noise, results, path):
    """Izquierda, recuperación según el ruido con desaprendizaje, Hebb y comparar con los originales;
    derecha, cuántas iteraciones tarda el desaprendizaje en quedar estable."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 5.2))
    lines = {'hebb': ('Hebb, sin lo común', 'C0', '-'),
             'unlearning': ('Hebb + desaprendizaje, sin lo común', 'C1', '-'),
             'nearest': ('sin red: distancia de Hamming a los originales', 'gray', '--')}
    for method, (label, color, style) in lines.items():
        rows = by_noise[by_noise.method == method].groupby('noise').accuracy.mean()
        left.plot(rows.index, rows, marker='o', lw=3, color=color, linestyle=style, label=label)
    left.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    _percent_axis(left, 'Módulos invertidos')
    left.legend(loc='lower left', fontsize=13)
    unlearning = results[results.method == 'unlearning']
    shares = unlearning.iterations.clip(upper=3).groupby(unlearning.noise).value_counts(normalize=True)
    shares = shares.unstack(fill_value=0)
    names = {0: 'ya estable', 1: '1 iteración', 2: '2 iteraciones', 3: '3 a 6'}
    bottom = np.zeros(len(shares))
    for steps, color in zip(shares.columns, ('lightgray', 'C0', 'C1', 'C3')):
        right.bar(np.arange(len(shares)), shares[steps], bottom=bottom, color=color, label=names[steps], width=.7)
        bottom += shares[steps].to_numpy()
    right.set_xticks(np.arange(len(shares)), [f'{_decimal(n * 100)} %' for n in shares.index])
    right.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    right.set(xlabel='Módulos invertidos', ylabel='Consultas', ylim=(0, 1), title='Iteraciones hasta quedar estable')
    right.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    _save(fig, path)


def plot_rules_by_noise(by_noise, pseudoinverse, path):
    """Las reglas sobre las mismas 700 consultas nuevas, según el ruido. Comparar con los originales no es una
    red: es el techo con este ruido."""
    rows = pd.concat([by_noise, pseudoinverse[pseudoinverse.method != 'unlearning']])
    lines = {'hebb': ('Hebb, sin lo común', 'C0', '-'),
             'unlearning': ('Hebb + desaprendizaje, sin lo común', 'C1', '-'),
             'pseudoinverse_stripped': ('pseudoinversa, sin lo común', 'C4', '-'),
             'pseudoinverse_full': ('pseudoinversa, QR completo', '#1a9e6e', '-'),
             'nearest': ('sin red: distancia de Hamming a los originales', 'gray', '--')}
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for method, (label, color, style) in lines.items():
        values = rows[rows.method == method].groupby('noise').accuracy.mean()
        ax.plot(values.index, values, marker='o', lw=3, color=color, linestyle=style, label=label)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    _percent_axis(ax, 'Módulos invertidos')
    ax.legend(loc='lower left', fontsize=14)
    _save(fig, path)


def plot_realistic_damage(summary, path, kinds=('desgaste', 'mancha_negra', 'rayones')):
    """Daño realista: recuperación exacta de las dos reglas que guardan los cinco QR y de comparar con los
    originales. `summary` puede juntar varias corridas: cada nivel se ubica por la fracción media de módulos
    alterados. Todos los paneles comparten el eje x hasta 50 %, y lo que no se probó queda sombreado, para
    que se vea hasta dónde llegó cada tipo."""
    titles = {'desgaste': 'Desgaste', 'mancha_negra': 'Manchas negras', 'rayones': 'Rayones'}
    lines = {'desaprendizaje': ('Hebb + desaprendizaje', 'C1', '-', 'exacta'),
             'pseudoinversa': ('pseudoinversa, QR completo', '#1a9e6e', '-', 'exacta'),
             'vecino': ('sin red: distancia de Hamming a los originales', 'gray', '--', 'exacta'),
             'vecino_asimetrico': ('sin red: sabiendo que solo se borra tinta', 'black', ':', 'exacta')}
    limit = 50
    fig, axes = plt.subplots(1, len(kinds), figsize=(15, 5.4), sharex=True, sharey=True)
    for ax, kind in zip(axes, kinds):
        rows = summary[summary.kind == kind]
        damage = (rows[rows.method == 'consulta'].set_index('level').dano / (SIDE * SIDE)).sort_values()
        tested = 100 * damage.max()
        if tested < limit - 10:
            ax.axvspan(tested + 1, limit + 1.5, color='0.92', zorder=0, lw=0)
            ax.text((tested + 2.5 + limit) / 2, .5, 'no probado', rotation=90, ha='center', va='center',
                    color='0.4', fontsize=17)
        for method, (label, color, style, column) in lines.items():
            if method == 'vecino_asimetrico' and kind != 'desgaste':
                continue
            values = rows[rows.method == method].set_index('level')[column].reindex(damage.index)
            ax.plot(100 * damage, values, marker='o', markersize=5, lw=3.4, color=color, linestyle=style, label=label)
        ax.set_title(titles[kind], fontsize=22)
        ax.set_xlim(0, limit + 1.5)
        ax.set_xticks(range(0, limit + 1, 20))
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0, symbol=' %'))
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.tick_params(labelsize=16)
        _percent_axis(ax, '')
    for ax in axes[1:]:
        ax.set_ylabel('')
    axes[0].set_ylabel('Recuperación exacta', fontsize=18)
    fig.supxlabel('Módulos alterados', fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=16, bbox_to_anchor=(.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .86))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_iterations(results, path):
    """Con desaprendizaje: qué fracción de las consultas queda estable tras 0, 1, 2 o 3 o más iteraciones,
    por nivel de ruido."""
    unlearning = results[results.method == 'unlearning']
    shares = unlearning.iterations.clip(upper=3).groupby(unlearning.noise).value_counts(normalize=True)
    shares = shares.unstack(fill_value=0)
    labels = {0: 'ya estable', 1: '1 iteración', 2: '2 iteraciones', 3: '3 o más'}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bottom = np.zeros(len(shares))
    positions = np.arange(len(shares))
    for steps, color in zip(shares.columns, ('lightgray', 'C0', 'C1', 'C3')):
        ax.bar(positions, shares[steps], bottom=bottom, color=color, label=labels[steps], width=.7)
        bottom += shares[steps].to_numpy()
    ax.set_xticks(positions, [_decimal(noise) for noise in shares.index])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set(xlabel='Fracción de módulos invertidos', ylabel='Consultas', ylim=(0, 1))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    _save(fig, path)
