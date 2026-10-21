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
# En 3D las etiquetas se superponen más: se agrupan puntas más lejanas y no se rotulan las flechas cortas.
ARROW_LABEL_MERGE_DISTANCE_3D = .6
# Correlación mínima en el plano o espacio dibujado para rotular una flecha: las cortas están mal representadas ahí.
MIN_LABELED_LOADING = .45
MIN_LABELED_LOADING_3D = .6
LOG_SCALE_COLUMNS = {'followers', 'price_usd'}


def _save(fig, path, dpi=160, rect=(0, 0, 1, 1)):
    """rect reserva lugar fuera de los ejes, por ejemplo para una leyenda de toda la figura arriba."""
    try:
        fig.tight_layout(rect=rect)
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
    fig, ax = plt.subplots(figsize=(14, 10))
    _biplot_panel(ax, frame, analysis, (0, 1))
    ax.set_title('Biplot: flechas = correlaciones × 2 (escala visual)')
    _save(fig, output / 'biplot.png')


def plot_biplot_pc3(frame, analysis: PCAAnalysis, output):
    """Los dos planos que agregan PC3 al biplot: PC1–PC3 y PC2–PC3."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, components in zip(axes, [(0, 2), (1, 2)]):
        _biplot_panel(ax, frame, analysis, components, legend=ax is axes[0])
    fig.suptitle('Biplots con PC3: flechas = correlaciones × 2; se rotulan las de correlación ≥ '
                 f'{MIN_LABELED_LOADING:.2f} en cada plano'.replace('.', ','))
    _save(fig, output / 'biplot_pc3.png')


def _biplot_panel(ax, frame, analysis, components, legend=True):
    result = analysis.pca
    components = list(components)
    points = result.scores[:, components] / np.sqrt(result.eigenvalues[components])
    ax.scatter(*points.T, c=main_topic(frame).map(TOPIC_COLORS))
    _annotate_creators(ax, frame, points)
    _draw_loadings(ax, analysis, components, min_labeled=0 if list(components) == [0, 1] else MIN_LABELED_LOADING)
    if legend:
        _topic_legend(ax, loc='upper right')
    first, second = pca_axis_labels(analysis, max(components) + 1, components)
    ax.set(xlabel=f'{first} / desvío', ylabel=f'{second} / desvío')
    ax.axhline(0, color='black', lw=.4)
    ax.axvline(0, color='black', lw=.4)
    ax.margins(x=.18)  # Deja lugar a las etiquetas de las flechas que salen por los costados.


def pca_axis_labels(analysis: PCAAnalysis, count, components=None):
    ratio = analysis.pca.explained_variance_ratio
    return [f'PC{index + 1} ({ratio[index]:.1%})' for index in (components or range(count))]


def scatter_3d(ax, scores, colors, labels):
    """Scores sin reescalar, con caja proporcional a su rango: PC1 se ve tan dominante como es.

    Una línea vertical une cada punto con el piso para leer su posición en PC1–PC2.
    """
    floor = scores[:, 2].min()
    for x, y, z in scores:
        ax.plot([x, x], [y, y], [floor, z], color='gray', lw=.4, alpha=.5)
    ax.scatter(*scores.T, c=colors, s=30, depthshade=False, edgecolors='black', linewidths=.3)
    ax.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])
    ax.set_box_aspect(np.ptp(scores, axis=0))


def plot_pca_3d(frame, analysis: PCAAnalysis, output):
    """Biplot 3D: dos vistas fijas en PNG y una rotación completa en GIF, coloreado por tema principal."""
    fig = plt.figure(figsize=(17, 8))
    for position, (elevation, azimuth) in enumerate([(20, -60), (10, 15)], start=1):
        ax = fig.add_subplot(1, 2, position, projection='3d')
        _biplot_3d(ax, frame, analysis)
        ax.view_init(elevation, azimuth)
    _topic_legend(fig.axes[0], loc='upper left')
    fig.suptitle(f'Biplot 3D, PC1–PC3 ({analysis.pca.cumulative_variance[2]:.1%} de la varianza): '
                 f'flechas = correlaciones × 2, rotuladas si ≥ {MIN_LABELED_LOADING_3D:.1f}'.replace('.', ','))
    _save(fig, output / 'pca_3d.png')

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection='3d')
    _biplot_3d(ax, frame, analysis)
    ax.grid(False)  # La grilla cambia en cada cuadro y es lo que más pesa en el GIF.
    _topic_legend(ax, loc='upper left')
    ax.set_title('Biplot 3D por tema principal')
    try:
        _save_rotation(fig, ax, output / 'pca_3d.gif')
    finally:
        plt.close(fig)


def _save_rotation(fig, ax, path, frames=36, dpi=60, seconds_per_frame=.17):
    """Una vuelta completa con una paleta fija de grises y colores de tema: el GIF queda en unos cientos de KB.

    Una paleta calculada desde los cuadros pierde los colores de los puntos, que ocupan pocos píxeles.
    """
    from PIL import Image
    from matplotlib.colors import to_rgb

    images = []
    for step in range(frames):
        ax.view_init(20, -60 + 360 * step / frames)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert('RGB'))
    palette = _fixed_palette([to_rgb(TOPIC_COLORS[topic]) for topic in TOPICS])
    quantized = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    quantized[0].save(path, save_all=True, append_images=quantized[1:], loop=0, optimize=True,
                      duration=int(seconds_per_frame * 1000))


def _biplot_3d(ax, frame, analysis):
    result = analysis.pca
    points = result.scores[:, :3] / np.sqrt(result.eigenvalues[:3])
    ax.scatter(*points.T, c=main_topic(frame).map(TOPIC_COLORS).to_numpy(), s=28, depthshade=False,
               edgecolors='black', linewidths=.3)
    tips = result.correlation_loadings[:, :3] * 2
    ax.quiver(0, 0, 0, *tips.T, color='gray', alpha=.6, arrow_length_ratio=.06, lw=.8)
    labeled = np.flatnonzero(np.linalg.norm(tips, axis=1) >= MIN_LABELED_LOADING_3D * 2)
    for members in _coincident_groups(tips[labeled], ARROW_LABEL_MERGE_DISTANCE_3D):
        anchor = tips[labeled[members]].mean(axis=0) * 1.1
        ax.text(*anchor, _group_label(analysis.columns[labeled[members]]), fontsize=7,
                bbox={'facecolor': 'white', 'alpha': .7, 'edgecolor': 'none', 'pad': 1})
    ax.set(xlabel='PC1 / desvío', ylabel='PC2 / desvío', zlabel='PC3 / desvío')
    ax.set_box_aspect((1, 1, 1))


def _group_label(names, shown=3):
    """Etiqueta de flechas casi coincidentes: las primeras y cuántas más."""
    names = list(names)
    extra = f'\n(+{len(names) - shown} más)' if len(names) > shown else ''
    return '\n'.join(names[:shown]) + extra


def _fixed_palette(colors, grays=24):
    from PIL import Image

    levels = [(value,) * 3 for value in np.linspace(0, 255, grays).astype(int)]
    entries = levels + [tuple(int(channel * 255) for channel in color) for color in colors]
    palette = Image.new('P', (1, 1))
    palette.putpalette([channel for entry in entries for channel in entry] + [0] * (768 - 3 * len(entries)))
    return palette


def _annotate_creators(ax, frame, points):
    for name, point in zip(frame.name, points):
        ax.annotate(name, point, xytext=(3, 2), textcoords='offset points', fontsize=6, alpha=.75)


def _draw_loadings(ax, analysis, components=(0, 1), min_labeled=0.0):
    tips = analysis.pca.correlation_loadings[:, list(components)] * 2
    for horizontal, vertical in tips:
        ax.arrow(0, 0, horizontal, vertical, color='gray', alpha=.5,
                 head_width=.025, length_includes_head=True)
    labeled = np.flatnonzero(np.linalg.norm(tips, axis=1) >= min_labeled * 2)
    for members in _coincident_groups(tips[labeled]):
        members = labeled[members]
        anchor = tips[members].mean(axis=0) * 1.04
        label = '\n'.join(analysis.columns[index] for index in members)
        ax.text(*anchor, label, fontsize=7, ha='left' if anchor[0] >= 0 else 'right', va='center')


def _coincident_groups(tips, distance=ARROW_LABEL_MERGE_DISTANCE):
    """Agrupa flechas cuyas puntas casi coinciden, para no superponer sus etiquetas."""
    groups = []
    for index, tip in enumerate(tips):
        group = next((g for g in groups if np.linalg.norm(tips[g[0]] - tip) < distance), None)
        if group is None:
            groups.append([index])
        else:
            group.append(index)
    return groups


def _show_correlations(ax, correlations):
    """Mapa de calor sin la diagonal: la correlación de una variable consigo misma es 1 y no informa nada."""
    off_diagonal = np.ma.masked_where(np.eye(len(correlations), dtype=bool), correlations.to_numpy())
    cmap = plt.get_cmap('coolwarm').copy()
    cmap.set_bad('white')
    return ax.imshow(off_diagonal, vmin=-1, vmax=1, cmap=cmap)


def plot_correlations(correlations, output):
    fig, ax = plt.subplots(figsize=(13, 11))
    image = _show_correlations(ax, correlations)
    indices = range(len(correlations))
    ax.set_xticks(indices, correlations.columns, rotation=90, fontsize=6)
    ax.set_yticks(indices, correlations.columns, fontsize=6)
    fig.colorbar(image, ax=ax)
    _save(fig, output / 'correlaciones.png')


def plot_variable_blocks(blocks: VariableBlocks, output):
    correlations = blocks.ordered_correlations
    fig, ax = plt.subplots(figsize=(13, 11))
    image = _show_correlations(ax, correlations)
    for (row, column), value in np.ndenumerate(correlations.to_numpy()):
        if row == column:
            continue
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
