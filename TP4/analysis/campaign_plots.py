"""Gráficos de las decisiones de campaña: reciben el catálogo con regiones y las selecciones ya calculadas."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .eda import TOPICS
from .plots import TOPIC_COLORS, TOPIC_LABELS, _save, _topic_legend

COST_LABEL = 'USD por 1.000 cuentas argentinas'
FOLLOWERS_COLOR, CHOSEN_COLOR, OTHER_COLOR = 'C1', 'C0', 'lightgray'
CRITERION_LABELS = {'followers': 'Seguidores', 'descubrimiento': 'Descubrimiento',
                    'profundidad': 'Profundidad', 'som': 'Elegido'}
FOOTBALL_METRICS = {'cpm_arg': COST_LABEL, 'argentina_audience_pct': 'Audiencia argentina (%)',
                    'non_follower_reach_pct': 'Alcance a no seguidores (%)', 'content_tutorial_pct': 'Tutoriales (%)'}


def plot_problem(frame, output):
    """Seguidores frente a costo por cuenta argentina y frente a tutoriales."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = frame.main_topic.map(TOPIC_COLORS)
    for ax, column, label in ((axes[0], 'cpm_arg', COST_LABEL), (axes[1], 'content_tutorial_pct', 'Tutoriales (%)')):
        ax.scatter(frame.followers, frame[column], c=colors, s=30)
        ax.set_xscale('log')
        rho = spearmanr(frame.followers, frame[column]).statistic
        ax.set(xlabel='Seguidores', ylabel=label, title=f'ρ de Spearman = {_decimal(rho, "+.2f")}')
    _plain_log_y(axes[0])
    _topic_legend(axes[1])
    _save(fig, output / '1_problema.png')


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
    _save(fig, output / '2_separacion.png')


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
    _save(fig, output / '3_candidatos.png', rect=(0, 0, 1, .96))


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


def plot_football_criteria(criteria, selections, output):
    """Fútbol: promedio del top 3 con cada criterio de un solo indicador frente a la selección propuesta."""
    football = criteria[criteria.topic == 'football']
    chosen = selections[(selections.topic == 'football') & (selections.strategy == 'som')].assign(criterion='som')
    means = pd.concat([football, chosen]).groupby('criterion')[list(FOOTBALL_METRICS)].mean() \
        .reindex(list(CRITERION_LABELS)[::-1])
    labels = [CRITERION_LABELS[criterion] for criterion in means.index]
    colors = [CHOSEN_COLOR if criterion == 'som' else OTHER_COLOR for criterion in means.index]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, (column, title) in zip(axes, FOOTBALL_METRICS.items()):
        ax.barh(labels, means[column], color=colors)
        for position, value in enumerate(means[column]):
            ax.text(value, position, f' {value:.0f}', va='center')
        ax.set(title=title)
        ax.set_xlim(0, means[column].max() * 1.25)
    _save(fig, output / '4_futbol_criterios.png')


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
    _save(fig, output / '5_frente_a_seguidores.png', rect=(0, 0, 1, .92))


def plot_campaigns(frame, criteria, selections, separation, random_baseline, target_regions, output):
    plot_problem(frame, output)
    plot_region_separation(separation, random_baseline, output)
    plot_candidates(frame, selections, target_regions, output)
    plot_football_criteria(criteria, selections, output)
    plot_against_followers(selections, output)
