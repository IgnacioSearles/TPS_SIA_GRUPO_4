"""Gráficos del ejercicio 1 para la presentación: los mismos resultados que los informes, con menos
rótulos, nombres en castellano y letra grande."""
from dataclasses import replace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from matplotlib.patches import Rectangle
import numpy as np

from analysis.campaign_plots import CHOSEN_COLOR, FOLLOWERS_COLOR
from analysis.eda import TOPICS, variable_blocks
from analysis.plots import TOPIC_COLORS, TOPIC_LABELS, _show_correlations
from analysis.som_campanas import MODEL, REGION_COUNT, TARGET_REGIONS
from analysis.som_regions import contiguous_ward
from analysis.som_plots import _cells, draw_region_borders, draw_region_map
from src.tp4.kohonen import Kohonen

LABELS = {
    'followers': 'seguidores', 'average_reach': 'alcance', 'average_impressions': 'impresiones',
    'average_video_views': 'vistas de video', 'monthly_posts': 'publicaciones por mes',
    'argentina_audience_pct': 'audiencia argentina', 'non_follower_reach_pct': 'alcance a no seguidores',
    'likes_per_1000': 'likes', 'comments_per_1000': 'comentarios', 'saves_per_1000': 'guardados',
    'shares_per_1000': 'compartidos', 'profile_visits_per_1000': 'visitas al perfil', 'price_usd': 'precio',
    'sponsored_posts_pct': 'publicidad', 'topic_football_pct': 'fútbol', 'topic_makeup_pct': 'maquillaje',
    'topic_dogs_pct': 'perros', 'topic_art_pct': 'arte', 'content_tutorial_pct': 'tutoriales',
    'content_review_pct': 'reseñas', 'content_entertainment_pct': 'entretenimiento',
    'content_showcase_pct': 'exhibición', 'average_caption_words': 'texto largo',
    'average_watch_time_seconds': 'tiempo de visualización', 'video_posts_pct': 'videos',
    'short_form_posts_pct': 'formato corto', 'trend_posts_pct': 'tendencias',
}
# Flechas del biplot: las que apuntan casi igual comparten etiqueta.
BIPLOT_GROUPS = {
    'seguidores, alcance, precio': ['followers', 'average_reach', 'price_usd'],
    'alcance a no seguidores,\ntendencias': ['non_follower_reach_pct', 'trend_posts_pct'],
    'tutoriales, guardados,\ntexto largo': ['content_tutorial_pct', 'saves_per_1000', 'average_caption_words'],
    'audiencia argentina': ['argentina_audience_pct'],
    'perros': ['topic_dogs_pct'],
    'arte': ['topic_art_pct'],
    'exhibición': ['content_showcase_pct'],
}
BLOCK_NAMES = {'followers': 'visibilidad', 'saves_per_1000': 'profundidad'}
SIZE_COLUMNS = ['followers', 'average_impressions', 'average_reach', 'average_video_views']
VISIBILITY_PLANES = ['followers', 'non_follower_reach_pct', 'content_tutorial_pct', 'saves_per_1000']
TOPIC_PLANES = [f'topic_{topic}_pct' for topic in TOPICS]
MUTED = 'gray'
KEPT, DROPPED = 'C0', (*matplotlib.colors.to_rgb('C0'), .35)  # barras que cuentan para la conclusión y las demás
UNITS = {'followers': 'seguidores', 'saves_per_1000': 'cada 1.000 vistas'}  # el resto es un porcentaje
MERGE_STEPS = [16, 11, 8, REGION_COUNT]
FOLLOWER_TICKS = [2e4, 5e4, 1e5, 2e5, 5e5]
COLORBAR_FOLLOWER_TICKS = [2e4, 1e5, 5e5]  # la barra es angosta: menos marcas


def _save(fig, path):
    """Recorta al contenido: los nombres de bloque y las etiquetas de flechas quedan fuera de los ejes."""
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches='tight', pad_inches=.15)
    finally:
        plt.close(fig)


def _save_with_colorbar(fig, path):
    """Como `_save`, sin tight_layout: no es compatible con las barras de color compartidas."""
    try:
        fig.savefig(path, dpi=160, bbox_inches='tight', pad_inches=.15)
    finally:
        plt.close(fig)


def _topic_legend(ax, **kwargs):
    handles = [plt.Line2D([], [], marker='o', ls='', ms=10, color=TOPIC_COLORS[topic], label=TOPIC_LABELS[topic])
               for topic in TOPICS]
    ax.legend(handles=handles, **kwargs)


# --- Exploración --------------------------------------------------------------------

def plot_variance_share(frame, columns, path):
    """Parte de la varianza total que aporta cada variable, sin escalar y estandarizada."""
    raw = frame[columns].var()
    shares = {'Sin escalar': raw / raw.sum(), 'Estandarizadas (z-score)': raw * 0 + 1 / len(columns)}
    shown = SIZE_COLUMNS
    labels = [LABELS[column] for column in shown] + [f'las otras {len(columns) - len(shown)}']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (title, share) in zip(axes, shares.items()):
        values = [*share[shown], share.drop(shown).sum()]
        bars = ax.barh(labels, values, color=['C0'] * len(shown) + [MUTED])
        ax.bar_label(bars, [f'{value:.1%}'.replace('.', ',') for value in values], padding=6)
        ax.set(title=title, xlim=(0, 1), xticks=[])
        ax.spines['bottom'].set_visible(False)
    axes[0].invert_yaxis()
    _save(fig, path)


def plot_topic_composition(frame, topics, path):
    """Cada creador como una columna: qué parte de sus publicaciones es de cada tema."""
    shares = frame[[f'topic_{topic}_pct' for topic in TOPICS]].to_numpy()
    rank = {topic: index for index, topic in enumerate(TOPICS)}
    order = sorted(range(len(frame)), key=lambda row: (rank[topics[row]], -shares[row].max()))
    fig, ax = plt.subplots(figsize=(13, 5))
    bottom = np.zeros(len(order))
    for index, topic in enumerate(TOPICS):
        ax.bar(range(len(order)), shares[order, index], .85, bottom=bottom, color=TOPIC_COLORS[topic])
        bottom += shares[order, index]
    ax.set(xticks=[], ylabel='% de publicaciones', xlabel='Los 50 creadores, agrupados por tema principal',
           xlim=(-.6, len(order) - .4), ylim=(0, 100))
    _topic_legend(ax, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=4)
    _save(fig, path)


def plot_topic_profiles(frame, topics, path):
    """Perros frente al resto: tamaño, audiencia argentina y tutoriales por tema principal."""
    panels = {'followers': 'Seguidores', 'argentina_audience_pct': 'Audiencia argentina (%)',
              'content_tutorial_pct': 'Tutoriales (%)'}
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 5))
    for ax, (column, title) in zip(axes, panels.items()):
        groups = [frame.loc[topics == topic, column] for topic in TOPICS]
        ax.boxplot(groups, widths=.55, showfliers=False, medianprops={'color': 'black', 'lw': 2})
        for position, (topic, values) in enumerate(zip(TOPICS, groups), start=1):
            ax.scatter(position + rng.uniform(-.15, .15, len(values)), values, s=30, color=TOPIC_COLORS[topic], zorder=3)
        ax.set_xticks(range(1, len(TOPICS) + 1), [TOPIC_LABELS[topic] for topic in TOPICS], fontsize=13)
        ax.set_title(title)
        if column == 'followers':
            ax.set_yscale('log')
    _save(fig, path)


def plot_blocks(frame, columns, path):
    """Matriz de correlaciones ordenada por bloques, con nombres y el nombre de cada bloque al costado."""
    blocks = variable_blocks(frame[columns])
    correlations = blocks.ordered_correlations
    fig, ax = plt.subplots(figsize=(11, 9.5))
    image = _show_correlations(ax, correlations)
    start = 0
    for block_id in blocks.membership.unique():
        members = blocks.membership[blocks.membership == block_id].index
        if len(members) > 2:
            ax.add_patch(Rectangle((start - .5, start - .5), len(members), len(members), fill=False, lw=3))
            name = next(label for column, label in BLOCK_NAMES.items() if column in members)
            ax.annotate(name, (len(correlations) - .2, start + len(members) / 2 - .5), xytext=(10, 0),
                        textcoords='offset points', va='center', fontsize=22, weight='bold', annotation_clip=False)
        start += len(members)
    ax.set_yticks(range(len(correlations)), [LABELS[column] for column in correlations.index], fontsize=13)
    ax.set_xticks([])
    fig.colorbar(image, ax=ax, label='Correlación', location='bottom', pad=.03, shrink=.6, aspect=30)
    _save(fig, path)


def plot_biplot(frame, topics, pca, columns, path):
    """Plano PC1–PC2 con las variables que dan sentido a cada eje."""
    points = pca.scores[:, :2] / np.sqrt(pca.eigenvalues[:2])
    loadings = pca.correlation_loadings[:, :2] * 2
    fig, ax = plt.subplots(figsize=(12, 8.5))
    ax.scatter(*points.T, c=[TOPIC_COLORS[topic] for topic in topics], s=45, alpha=.55)
    for label, group in BIPLOT_GROUPS.items():
        tips = np.array([loadings[columns.index(column)] for column in group])
        for tip in tips:
            ax.annotate('', tip, (0, 0), arrowprops={'arrowstyle': '->', 'color': 'k', 'lw': 2})
        anchor = tips.mean(axis=0) * 1.1
        ax.text(*anchor, label, ha='left' if anchor[0] >= 0 else 'right', va='center', fontsize=16,
                bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none', 'pad': 1})
    ax.axhline(0, color='black', lw=.5)
    ax.axvline(0, color='black', lw=.5)
    ratio = pca.explained_variance_ratio
    ax.set(xlabel=f'PC1 ({ratio[0]:.0%}):  ← visibilidad · profundidad →',
           ylabel=f'PC2 ({ratio[1]:.0%}):  ← audiencia argentina · arte →', xticks=[], yticks=[])
    ax.margins(x=.75, y=.1)
    _topic_legend(ax, loc='lower left')
    _save(fig, path)


def plot_followers(frame, path):
    """Seguidores frente a precio y frente a audiencia argentina."""
    panels = {'price_usd': 'Precio por publicación (USD)', 'argentina_audience_pct': 'Audiencia argentina (%)'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (column, label) in zip(axes, panels.items()):
        ax.scatter(frame.followers, frame[column], s=45, color='C0')
        ax.set(xscale='log', xlabel='Seguidores', ylabel=label)
        if column == 'price_usd':
            ax.set_yscale('log')
        correlation = np.corrcoef(frame.followers, frame[column])[0, 1]
        ax.set_title(f'r = {correlation:+.2f}'.replace('.', ',').replace('+0,00', '0,00').replace('-0,00', '0,00'),
                     fontsize=22, weight='bold')
    _save(fig, path)


def plot_variance(pca, path):
    """Varianza por componente: en oscuro, las que hacen falta para llegar al 90 %."""
    cumulative = pca.cumulative_variance
    count = pca.components_for_variance(.9)
    positions = np.arange(1, len(cumulative) + 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(positions, pca.explained_variance_ratio, color=[KEPT if position <= count else DROPPED for position in positions])
    ax.plot(positions, cumulative, marker='o', color='k', lw=2, label='acumulada')
    ax.axhline(.9, color='gray', ls='--', lw=1)
    ax.text(count + .6, cumulative[count - 1] - .05, f'{count} componentes: {cumulative[count - 1]:.1%}'.replace('.', ','),
            fontsize=18, weight='bold', va='top')
    ax.set(xlabel='Componente', ylabel='Fracción de varianza', ylim=(0, 1.04), xlim=(0, len(cumulative) + 1))
    ax.legend(loc='center right')
    _save(fig, path)


# --- Kohonen ------------------------------------------------------------------------

def plot_occupancy(x, path):
    """Creadores por neurona con el mapa elegido (4×4) y con uno más grande (5×5), en la misma escala."""
    sizes = {4: MODEL, 5: replace(MODEL, rows=5, cols=5, radius_start=2.5)}
    soms = {side: Kohonen(config).fit(x) for side, config in sizes.items()}
    hits = {side: som.occupancy(x).ravel() for side, som in soms.items()}
    norm = Normalize(0, max(counts.max() for counts in hits.values()))
    fig, axes = plt.subplots(1, len(sizes), figsize=(13, 6.8))
    for ax, side in zip(axes, sizes):
        cells = _cells(ax, soms[side].grid, hits[side], 'viridis', edgecolors='white', linewidths=2, norm=norm)
        for (column, row), count in zip(soms[side].grid.positions, hits[side]):
            ax.text(column, row, str(count) if count else 'vacía', ha='center', va='center',
                    fontsize=22 if count else 15, weight='bold' if count else 'normal',
                    color='black' if norm(count) > .55 else 'white')
        empty, single = int(np.sum(hits[side] == 0)), int(np.sum(hits[side] == 1))
        ax.set_title(f'{side}×{side}: {empty} vacías, {single} con un solo creador', fontsize=18)
        ax.axis('off')
    fig.colorbar(cells, ax=axes, label='Creadores por neurona', location='bottom', shrink=.4, aspect=25, pad=.04)
    _save_with_colorbar(fig, path)


def _compact(value, _position=None):
    """1500 → 1,5 mil; 2000000 → 2 M: los seguidores van de miles a millones."""
    for size, suffix in ((1e6, ' M'), (1e3, ' mil')):
        if value >= size:
            return f'{value / size:.3g}{suffix}'.replace('.', ',')
    return f'{value:.3g}'.replace('.', ',')


def _followers_axis(axis, ticks=FOLLOWER_TICKS):
    """Eje logarítmico de seguidores con pocas marcas legibles: 20 mil, 50 mil, 100 mil..."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(_compact))
    axis.set_minor_formatter(NullFormatter())


def plot_planes(catalog, planes, columns, path):
    """Un mapa por variable, con los bordes de las regiones y su escala en unidades originales."""
    fig, axes = plt.subplots(1, len(columns), figsize=(4.2 * len(columns), 6.2))
    for ax, column in zip(axes, columns):
        norm = LogNorm() if column == 'followers' else None
        cells = _cells(ax, catalog.som.grid, planes[column], 'viridis', norm=norm)
        draw_region_borders(ax, catalog)
        ax.set_title(LABELS[column], fontsize=20)
        ax.axis('off')
        bar = fig.colorbar(cells, ax=ax, location='bottom', pad=.04, shrink=.9, aspect=14,
                           label=UNITS.get(column, '%'), format=FuncFormatter(_compact))
        bar.ax.tick_params(labelsize=13)
        if column == 'followers':
            _followers_axis(bar.ax.xaxis, COLORBAR_FOLLOWER_TICKS)
    _save_with_colorbar(fig, path)


def plot_region_merges(catalog, path):
    """Cómo se arman las regiones: se unen de a dos las zonas vecinas más parecidas, hasta que quedan cinco."""
    grid, weights = catalog.som.grid, catalog.som.weights_
    fig, axes = plt.subplots(1, len(MERGE_STEPS), figsize=(3 * len(MERGE_STEPS), 3.5))
    for ax, count in zip(axes, MERGE_STEPS):
        if count == REGION_COUNT:
            draw_region_map(ax, catalog, facecolors='white')
        else:
            _cells(ax, grid, np.zeros(len(weights)), ListedColormap(['white']), edgecolors='lightgray', linewidths=1)
            draw_region_borders(ax, replace(catalog, neuron_regions=contiguous_ward(weights, grid.adjacency, count)))
        ax.set_title(f'{count} neuronas' if count == len(weights) else f'{count} regiones', fontsize=20)
        ax.axis('off')
    _save(fig, path)


def plot_region_count(table, path):
    """Por qué cinco regiones: es la primera cantidad que separa los temas sin dejar regiones minúsculas."""
    panels = {'topic_purity': ('Creadores del tema más común', '{:.0%}'),
              'smallest_region': ('Creadores en la región más chica', '{:.0f}')}
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5))
    for ax, (column, (title, pattern)) in zip(axes, panels.items()):
        bars = ax.bar(table.regions, table[column],
                      color=[KEPT if count == REGION_COUNT else DROPPED for count in table.regions])
        ax.bar_label(bars, [pattern.format(value).replace('%', ' %') for value in table[column]], padding=4,
                     fontsize=14)
        ax.set(title=title, xlabel='Cantidad de regiones', xticks=table.regions, yticks=[])
        ax.spines['left'].set_visible(False)
    _save(fig, path)


# --- Campañas -----------------------------------------------------------------------

def plot_football(frame, selections, criteria, path):
    """Creadores de fútbol sobre PC1 y su audiencia argentina, con los elegidos y los tops de profundidad."""
    football = frame[frame.main_topic == 'football']
    chosen = football.name.isin(selections.query("topic == 'football' and strategy == 'som'").name)
    depth = football.name.isin(criteria.query("topic == 'football' and criterion == 'profundidad'").name)
    followers = football.name.isin(selections.query("topic == 'football' and strategy == 'seguidores'").name)
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.axvspan(football.pc1.min() - 1, 0, color='gray', alpha=.12, zorder=0)
    ax.scatter(football.pc1, football.argentina_audience_pct, s=70, color=MUTED, label='otros de fútbol')
    ax.scatter(football.pc1[followers], football.argentina_audience_pct[followers], s=170, marker='s',
               color=FOLLOWERS_COLOR, label='top 3 por seguidores')
    ax.scatter(football.pc1[depth], football.argentina_audience_pct[depth], s=190, marker='^', color='C4',
               label='top 3 por tutoriales')
    ax.scatter(football.pc1[chosen], football.argentina_audience_pct[chosen], s=380, facecolors='none',
               edgecolors=CHOSEN_COLOR, lw=3.5, label='elegidos')
    ax.text(.02, .96, 'visibilidad', transform=ax.transAxes, fontsize=22, weight='bold', va='top')
    ax.text(.98, .96, 'profundidad', transform=ax.transAxes, fontsize=22, weight='bold', va='top', ha='right')
    ax.set(xlabel='PC1 (índice de Oja)', ylabel='Audiencia argentina (%)',
           xlim=(football.pc1.min() - 1, football.pc1.max() + 1), ylim=(15, 100))
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=4, fontsize=14)
    _save(fig, path)


def plot_reach_on_pc1(frame, path):
    """Seguidores y alcance a no seguidores de los creadores de fútbol a lo largo de PC1."""
    football = frame[frame.main_topic == 'football']
    panels = {'followers': ('Masivo: seguidores', np.log(football.followers)),
              'non_follower_reach_pct': ('Descubrimiento: alcance a no seguidores (%)', football.non_follower_reach_pct)}
    limits = (football.pc1.min() - 1, football.pc1.max() + 1)
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5.4))
    for ax, (column, (title, compared)) in zip(axes, panels.items()):
        ax.axvspan(limits[0], 0, color='gray', alpha=.12, zorder=0)
        ax.scatter(football.pc1, football[column], s=70, color=TOPIC_COLORS['football'])
        correlation = np.corrcoef(football.pc1, compared)[0, 1]
        ax.set_title(f'{title}\nr = {correlation:+.2f}'.replace('.', ','), fontsize=18)
        ax.set(xlabel='PC1:  ← visibilidad · profundidad →', xlim=limits)
        if column == 'followers':
            ax.set_yscale('log')
            _followers_axis(ax.yaxis)
    _save(fig, path)


def plot_client_map(catalog, topic, path):
    """El mapa con las regiones elegidas para un cliente pintadas con el color de su tema."""
    targets = [catalog.region_names.index(region) for region in TARGET_REGIONS[topic]]
    colors = [TOPIC_COLORS[topic] if label in targets else 'whitesmoke' for label in catalog.neuron_regions]
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    draw_region_map(ax, catalog, facecolors=[matplotlib.colors.to_rgba(color, .55) for color in colors])
    ax.axis('off')
    _save(fig, path)


def plot_cost(selections, path):
    """Costo por cuenta argentina: top 3 por seguidores frente a los elegidos, por cliente."""
    picks = selections[selections.strategy.isin(['seguidores', 'som'])]
    cost = picks.groupby(['topic', 'strategy']).cpm_arg.mean().unstack('strategy').reindex(TOPICS)
    positions, width = np.arange(len(TOPICS)), .38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(positions - width / 2, cost.seguidores, width, color=FOLLOWERS_COLOR, label='Top 3 por seguidores')
    ax.bar(positions + width / 2, cost.som, width, color=CHOSEN_COLOR, label='Elegidos con el mapa')
    for position, (before, after) in enumerate(zip(cost.seguidores, cost.som)):
        ax.text(position + width / 2, after + .8, f'−{1 - after / before:.0%}'.replace('%', ' %'),
                ha='center', va='bottom', fontsize=18, weight='bold')
    ax.set_xticks(positions, [TOPIC_LABELS[topic] for topic in TOPICS])
    ax.set(ylabel='USD por 1.000 cuentas argentinas', ylim=(0, cost.to_numpy().max() * 1.15))
    ax.legend(loc='upper right')
    _save(fig, path)
