"""Gráficos de las decisiones de campaña: reciben el catálogo con regiones y las selecciones ya calculadas.

Cada gráfico muestra un resultado de los métodos (PCA, Oja, Kohonen) sobre el que se apoya una decisión.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np

from .eda import TOPICS
from .plots import TOPIC_COLORS, TOPIC_LABELS, _save
from .som_plots import REGION_COLORS, _cells

COST_LABEL = 'USD por 1.000 cuentas argentinas'
FOLLOWERS_COLOR, CHOSEN_COLOR, OTHER_COLOR = 'C1', 'C0', 'lightgray'
# Variables del caso de uso que se dibujan como flechas en el plano PC1–PC2.
BIPLOT_VARIABLES = {
    'followers': 'Seguidores', 'non_follower_reach_pct': 'Alcance a no seguidores',
    'content_tutorial_pct': 'Tutoriales', 'saves_per_1000': 'Guardados',
    'argentina_audience_pct': 'Audiencia argentina', 'topic_art_pct': 'Arte', 'topic_dogs_pct': 'Perros',
}
# Tops de un solo indicador sobre el eje de Oja: marcador, color, etiqueta y altura sobre la fila del tema.
OJA_REFERENCES = {'followers': ('s', FOLLOWERS_COLOR, 'Top 3 por seguidores', .2),
                  'descubrimiento': ('D', 'C2', 'Top 3 por alcance a no seguidores', .32),
                  'profundidad': ('^', 'C4', 'Top 3 por tutoriales', .2)}


def plot_pca_axes(frame, pca, columns, chosen, output):
    """Plano PC1–PC2 con las flechas del caso de uso y los creadores elegidos."""
    scale = np.sqrt(pca.eigenvalues[:2])
    points = pca.scores[:, :2] / scale
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(*points.T, c=frame.main_topic.map(TOPIC_COLORS), s=30)
    is_chosen = frame.name.isin(chosen.name).to_numpy()
    ax.scatter(*points[is_chosen].T, s=200, facecolors='none', edgecolors='black', lw=1.5)
    loadings = pca.correlation_loadings[:, :2] * 2
    for column, label in BIPLOT_VARIABLES.items():
        tip = loadings[columns.index(column)]
        ax.annotate('', tip, (0, 0), arrowprops={'arrowstyle': '->', 'color': 'dimgray', 'lw': 1.5})
        ax.text(*(tip * 1.08), label, ha='left' if tip[0] >= 0 else 'right', va='center')
    ax.axhline(0, color='black', lw=.4)
    ax.axvline(0, color='black', lw=.4)
    ratio = pca.explained_variance_ratio
    ax.set(xlabel=f'PC1 ({ratio[0]:.0%}): ← visibilidad · profundidad →',
           ylabel=f'PC2 ({ratio[1]:.0%}): ← audiencia argentina · arte y exhibición →')
    ax.margins(x=.3)  # lugar para las etiquetas de las flechas
    handles = [plt.Line2D([], [], marker='o', ls='', color=TOPIC_COLORS[topic], label=TOPIC_LABELS[topic])
               for topic in TOPICS]
    handles.append(plt.Line2D([], [], marker='o', ls='', mfc='none', mec='black', ms=12, label='Elegidos'))
    ax.legend(handles=handles, loc='lower left')
    _save(fig, output / '1_ejes_pca.png')


def plot_oja_axis(frame, criteria, chosen, output):
    """Índice de Oja por tema: dónde caen los tops de cada criterio y los elegidos sobre el eje alcance–profundidad."""
    fig, ax = plt.subplots(figsize=(11, 5))
    rows = {topic: len(TOPICS) - position for position, topic in enumerate(TOPICS)}
    for topic, row in rows.items():
        members = frame[frame.main_topic == topic]
        ax.scatter(members.pc1, np.full(len(members), row), color=TOPIC_COLORS[topic], s=35, alpha=.6)
    index = frame.set_index('name').pc1
    for criterion, (marker, color, label, offset) in OJA_REFERENCES.items():
        top = criteria[criteria.criterion == criterion]
        ax.scatter(index[top.name], top.topic.map(rows) + offset, marker=marker, color=color, s=45, label=label)
    ax.scatter(index[chosen.name], chosen.topic.map(rows), s=200, facecolors='none', edgecolors='black', lw=1.5,
               label='Elegidos')
    ax.axvline(0, color='black', lw=.4)
    ax.set_yticks(list(rows.values()), [TOPIC_LABELS[topic] for topic in rows])
    ax.set(xlabel='Índice de Oja (= PC1): ← visibilidad · profundidad →', ylim=(.4, len(TOPICS) + .8))
    ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(.5, 1.13), frameon=False)
    _save(fig, output / '2_indice_oja.png')


def plot_map_targets(catalog, ar_plane, chosen, target_regions, output):
    """Izquierda: regiones del mapa con la región de cada cliente y sus elegidos. Derecha: plano de audiencia argentina."""
    grid = catalog.som.grid
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    regions = _cells(axes[0], grid, catalog.neuron_regions, ListedColormap(REGION_COLORS[:len(catalog.region_names)]),
                     edgecolors='white', linewidths=2)
    regions.set_alpha(.35)
    neurons = chosen.neuron.to_numpy()
    for position, ((column, row), label) in enumerate(zip(grid.positions, catalog.neuron_regions)):
        cell = '{},{}'.format(*grid.indices[position])
        names = '\n'.join(chosen.name[neurons == cell])
        axes[0].text(column, row - .32, catalog.region_names[label], ha='center', va='center', weight='bold')
        axes[0].text(column, row + .08, names, ha='center', va='center', fontsize=7)
    axes[0].set_title('Regiones y elegidos\n' + _client_targets(target_regions))

    plane = _cells(axes[1], grid, ar_plane, 'viridis')
    fig.colorbar(plane, ax=axes[1], shrink=.8, label='Audiencia argentina del prototipo (%)')
    for position, ((column, row), label) in enumerate(zip(grid.positions, catalog.neuron_regions)):
        axes[1].text(column, row, f'{catalog.region_names[label]}\n{ar_plane[position]:.0f} %', ha='center',
                     va='center', fontsize=9, bbox={'facecolor': 'white', 'alpha': .7, 'edgecolor': 'none', 'pad': 1})
    axes[1].set_title('Plano de audiencia argentina')
    _save(fig, output / '3_mapa_clientes.png')


def _client_targets(target_regions):
    """'Perros → R1 · Fútbol y Maquillaje → R4 + R5': agrupa clientes con las mismas regiones."""
    groups = {}
    for topic, regions in target_regions.items():
        groups.setdefault(' + '.join(regions), []).append(TOPIC_LABELS[topic])
    return ' · '.join(f'{" y ".join(topics)} → {regions}' for regions, topics in groups.items())


def plot_region_separation(separation, random_baseline, output):
    """η² por variable: qué separa el mapa y qué no."""
    ordered = separation.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [CHOSEN_COLOR if value > 2 * random_baseline else OTHER_COLOR for value in ordered]
    ax.barh(ordered.index, ordered, color=colors)
    ax.axvline(random_baseline, color='black', ls='--', lw=1)
    ax.text(random_baseline + .01, len(ordered) - .5, f'azar ({_decimal(random_baseline, ".2f")})', va='center')
    for position, value in enumerate(ordered):
        ax.text(value + .01, position, _decimal(value, '.2f'), va='center')
    ax.set(xlabel='η² por región', xlim=(0, 1))
    _save(fig, output / '4_separacion.png')


def plot_candidates(frame, selections, target_regions, output):
    """Por cliente: todos los creadores del tema, la región objetivo, el top por seguidores y los elegidos."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, topic in zip(axes.flat, TOPICS):
        candidates = frame[frame.main_topic == topic]
        in_target = candidates.region.isin(target_regions[topic])
        ax.scatter(candidates.argentina_audience_pct[~in_target], candidates.cpm_arg[~in_target],
                   color=OTHER_COLOR, s=40, label='Otras regiones')
        ax.scatter(candidates.argentina_audience_pct[in_target], candidates.cpm_arg[in_target],
                   color=TOPIC_COLORS[topic], s=40, label='Región objetivo')
        picks = selections[selections.topic == topic]
        followers_top, chosen = picks[picks.strategy == 'seguidores'], picks[picks.strategy == 'som']
        _ring(ax, followers_top, FOLLOWERS_COLOR, 's', 'Top 3 por seguidores')
        _ring(ax, chosen, 'black', 'o', 'Elegidos')
        _label(ax, followers_top[~followers_top.name.isin(chosen.name)], below=True)
        _label(ax, chosen)
        _plain_log_y(ax)
        ax.set_ylim(candidates.cpm_arg.min() / 1.3, candidates.cpm_arg.max() * 1.15)  # lugar para las etiquetas
        ax.set(title=f'{TOPIC_LABELS[topic]}: {" + ".join(target_regions[topic])}',
               xlabel='Audiencia argentina (%)', ylabel=COST_LABEL)
    handles = [plt.Line2D([], [], marker='o', ls='', color=OTHER_COLOR, label='Otras regiones'),
               plt.Line2D([], [], marker='o', ls='', color='dimgray', label='Región objetivo (color del tema)'),
               plt.Line2D([], [], marker='s', ls='', mfc='none', mec=FOLLOWERS_COLOR, ms=12, label='Top 3 por seguidores'),
               plt.Line2D([], [], marker='o', ls='', mfc='none', mec='black', ms=12, label='Elegidos')]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)
    _save(fig, output / '5_candidatos.png', rect=(0, 0, 1, .96))


def _ring(ax, creators, color, marker, label):
    ax.scatter(creators.argentina_audience_pct, creators.cpm_arg, s=220, marker=marker,
               facecolors='none', edgecolors=color, lw=1.8, label=label)


def _label(ax, creators, below=False):
    # Los del top por seguidores van debajo del punto: así no chocan con un elegido cercano.
    offset, alignment = ((0, -16), 'top') if below else ((10, 0), 'center')
    for _, creator in creators.iterrows():
        ax.annotate(creator['name'], (creator.argentina_audience_pct, creator.cpm_arg), xytext=offset,
                    textcoords='offset points', fontsize=8, va=alignment, ha='center' if below else 'left')


def _plain_log_y(ax):
    """Escala logarítmica con marcas 10, 20, 50… en lugar de potencias de diez."""
    ax.set_yscale('log')
    axis = ax.yaxis
    axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(plt.NullFormatter())


def _decimal(value, spec):
    return format(value, spec).replace('-', '−').replace('.', ',')


def plot_against_followers(selections, output):
    """Por tema: top 3 por seguidores frente a la selección propuesta, en costo, audiencia y gasto."""
    picks = selections[selections.strategy.isin(['seguidores', 'som'])]
    grouped = picks.groupby(['topic', 'strategy'])
    table = grouped[['cpm_arg', 'argentina_audience_pct']].mean().join(grouped.price_usd.sum()).unstack('strategy') \
        .reindex(TOPICS)
    panels = {'cpm_arg': COST_LABEL, 'argentina_audience_pct': 'Audiencia argentina (%)',
              'price_usd': 'USD por tres publicaciones'}
    positions, width = np.arange(len(TOPICS)), .38
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (column, title) in zip(axes, panels.items()):
        followers, proposed = table[column, 'seguidores'], table[column, 'som']
        ax.bar(positions - width / 2, followers, width, color=FOLLOWERS_COLOR, label='Top 3 por seguidores')
        ax.bar(positions + width / 2, proposed, width, color=CHOSEN_COLOR, label='Elegidos')
        ax.set_xticks(positions, [TOPIC_LABELS[topic] for topic in TOPICS])
        ax.set(title=title)
    for position, (before, after) in enumerate(zip(table['cpm_arg', 'seguidores'], table['cpm_arg', 'som'])):
        axes[0].text(position + width / 2, after, _decimal(after / before - 1, '+.0%'), ha='center', va='bottom')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False)
    _save(fig, output / '6_frente_a_seguidores.png', rect=(0, 0, 1, .92))
