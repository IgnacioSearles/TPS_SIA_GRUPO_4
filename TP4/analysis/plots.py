"""Gráficos del EDA: recibe resultados calculados, sin entrenar modelos."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from .eda import TOPICS, TopicTables, VariableBlocks, cost_per_thousand_argentine_reach, main_topic
from .representation import PCAAnalysis

TOPIC_LABELS = {'football': 'Fútbol', 'makeup': 'Maquillaje', 'dogs': 'Perros', 'art': 'Arte'}
TOPIC_COLORS = {topic: f'C{index}' for index, topic in enumerate(TOPICS)}
# Flechas del biplot más cercanas que esto (en unidades del dibujo) comparten etiqueta.
ARROW_LABEL_MERGE_DISTANCE = .2
LOG_SCALE_COLUMNS = {'followers', 'price_usd'}


def _save(fig, path, dpi=160):
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def _topic_legend(ax, **options):
    handles = [plt.Line2D([], [], marker='o', ls='', color=TOPIC_COLORS[topic]) for topic in TOPICS]
    ax.legend(handles, [TOPIC_LABELS[topic] for topic in TOPICS], title='Tema principal', **options)


def plot_variance(analysis: PCAAnalysis, output):
    result = analysis.pca
    indices = np.arange(1, len(result.eigenvalues) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(indices, result.explained_variance_ratio, label='Individual')
    ax.plot(indices, result.cumulative_variance, 'o-', label='Acumulada')
    ax.axhline(.9, color='gray', ls='--')
    ax.set(xlabel='Componente', ylabel='Fracción de varianza',
           title='PCA: variables seleccionadas y estandarizadas')
    ax.legend()
    _save(fig, output / 'varianza.png')


def plot_biplot(frame, analysis: PCAAnalysis, output):
    result = analysis.pca
    points = result.scores[:, :2] / np.sqrt(result.eigenvalues[:2])
    colors = main_topic(frame).map(TOPIC_COLORS)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(*points.T, c=colors)
    _annotate_creators(ax, frame, points)
    _draw_loadings(ax, analysis)
    _topic_legend(ax, loc='upper right')
    ratio = result.explained_variance_ratio
    ax.set(xlabel=f'PC1 ({ratio[0]:.1%}) / desvío PC1',
           ylabel=f'PC2 ({ratio[1]:.1%}) / desvío PC2',
           title='Biplot: flechas = correlaciones × 2 (escala visual)')
    ax.axhline(0, color='black', lw=.4)
    ax.axvline(0, color='black', lw=.4)
    ax.margins(x=.18)  # Deja lugar a las etiquetas de las flechas que salen por los costados.
    _save(fig, output / 'biplot.png')


def _annotate_creators(ax, frame, points):
    for name, point in zip(frame.name, points):
        ax.annotate(name, point, xytext=(3, 2), textcoords='offset points', fontsize=6, alpha=.75)


def _draw_loadings(ax, analysis):
    tips = analysis.pca.correlation_loadings[:, :2] * 2
    for horizontal, vertical in tips:
        ax.arrow(0, 0, horizontal, vertical, color='gray', alpha=.5,
                 head_width=.025, length_includes_head=True)
    for members in _coincident_groups(tips):
        anchor = tips[members].mean(axis=0) * 1.04
        label = '\n'.join(analysis.columns[index] for index in members)
        ax.text(*anchor, label, fontsize=7, ha='left' if anchor[0] >= 0 else 'right', va='center')


def _coincident_groups(tips):
    """Agrupa flechas cuyas puntas casi coinciden, para no superponer sus etiquetas."""
    groups = []
    for index, tip in enumerate(tips):
        group = next((g for g in groups if np.linalg.norm(tips[g[0]] - tip) < ARROW_LABEL_MERGE_DISTANCE), None)
        if group is None:
            groups.append([index])
        else:
            group.append(index)
    return groups


def plot_correlations(correlations, output):
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
    indices = range(len(correlations))
    ax.set_xticks(indices, correlations.columns, rotation=90, fontsize=6)
    ax.set_yticks(indices, correlations.columns, fontsize=6)
    fig.colorbar(image, ax=ax)
    _save(fig, output / 'correlaciones.png')


def plot_variable_blocks(blocks: VariableBlocks, output):
    correlations = blocks.ordered_correlations
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
    for (row, column), value in np.ndenumerate(correlations.to_numpy()):
        ax.text(column, row, f'{value:.1f}'.replace('0.', '.'), ha='center', va='center', fontsize=5)
    start = 0
    for size in blocks.membership.value_counts(sort=False).loc[blocks.membership.unique()]:
        if size > 1:
            ax.add_patch(Rectangle((start - .5, start - .5), size, size, fill=False, lw=2))
        start += size
    indices = range(len(correlations))
    labels = [f'{name} [{block}]' for name, block in blocks.membership.items()]
    ax.set_xticks(indices, labels, rotation=90, fontsize=7)
    ax.set_yticks(indices, labels, fontsize=7)
    ax.set_title('Correlaciones de las 27 variables seleccionadas, agrupadas en bloques [número]')
    fig.colorbar(image, ax=ax)
    _save(fig, output / 'correlaciones_bloques.png')


def plot_scale_comparison(raw, standardized, output):
    """Mismo orden de variables: en crudo dominan las de mayor unidad; en z quedan comparables."""
    order = raw.std().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    axes[0].boxplot(raw[order].to_numpy(), vert=False, tick_labels=order)
    axes[0].set(title='Valores originales', xlabel='Valor (unidades de cada variable)')
    axes[1].boxplot(standardized[order].to_numpy(), vert=False, tick_labels=order)
    axes[1].set(title='Estandarizadas (z-score)', xlabel='Desvíos respecto de la media')
    axes[1].axvline(0, color='gray', lw=.6)
    for ax in axes:
        ax.tick_params(axis='y', labelsize=7)
    _save(fig, output / 'escalas.png')


def plot_topic_composition(frame, topics: TopicTables, output):
    shares = frame[[f'topic_{topic}_pct' for topic in TOPICS]].to_numpy()
    rank = {topic: index for index, topic in enumerate(TOPICS)}
    order = topics.specialization.assign(position=topics.specialization.main_topic.map(rank)) \
        .sort_values(['position', 'main_share']).index.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 12))
    left = np.zeros(len(order))
    for column, topic in enumerate(TOPICS):
        ax.barh(np.arange(len(order)), shares[order, column], left=left,
                color=TOPIC_COLORS[topic], label=TOPIC_LABELS[topic])
        left += shares[order, column]
    ax.set_yticks(np.arange(len(order)), frame.name.to_numpy()[order], fontsize=6)
    ax.set(xlabel='% de publicaciones por tema', xlim=(0, 100),
           title='Composición temática de cada creador (ordenados por tema principal)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.04), ncol=4)
    _save(fig, output / 'temas_composicion.png')


def plot_topic_profiles(frame, columns, output):
    """Una caja por tema principal con los creadores superpuestos: cada grupo tiene 11–15."""
    profiled = frame.assign(cpm_arg=cost_per_thousand_argentine_reach(frame), main_topic=main_topic(frame))
    jitter = np.random.default_rng(0)
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, column in zip(axes.flat, columns):
        groups = [profiled.loc[profiled.main_topic == topic, column].to_numpy() for topic in TOPICS]
        ax.boxplot(groups, tick_labels=[TOPIC_LABELS[topic] for topic in TOPICS], showfliers=False)
        for position, (topic, values) in enumerate(zip(TOPICS, groups), start=1):
            offsets = position + jitter.uniform(-.15, .15, len(values))
            ax.scatter(offsets, values, s=14, color=TOPIC_COLORS[topic], alpha=.8, zorder=3)
        if column in LOG_SCALE_COLUMNS:
            ax.set_yscale('log')
        ax.set_title(column, fontsize=10)
    fig.suptitle('Perfil por tema principal (escala log en seguidores y precio)')
    _save(fig, output / 'temas_perfiles.png')


def plot_follower_relations(frame, relations, output):
    profiled = frame.assign(cpm_arg=cost_per_thousand_argentine_reach(frame))
    colors = main_topic(frame).map(TOPIC_COLORS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (column, row) in zip(axes.flat, relations.iterrows()):
        ax.scatter(profiled.followers, profiled[column], c=colors, s=20)
        ax.set_xscale('log')
        if column in LOG_SCALE_COLUMNS:
            ax.set_yscale('log')
        ax.set(xlabel='followers', ylabel=column,
               title=f'r = {row.pearson:+.2f}   ρ Spearman = {row.spearman:+.2f}')
    _topic_legend(axes.flat[0], fontsize=8)
    fig.suptitle('Seguidores frente a variables de costo, audiencia y profundidad')
    _save(fig, output / 'seguidores_relaciones.png')


def plot_distributions(numeric, output):
    axes = numeric.hist(figsize=(20, 18), bins=10)
    _save(axes.flatten()[0].figure, output / 'distribuciones.png', dpi=130)
