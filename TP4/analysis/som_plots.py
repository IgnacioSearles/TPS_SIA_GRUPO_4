"""Gráficos del mapa de Kohonen: reciben el modelo y las regiones ya calculadas."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from .eda import TOPICS
from .plots import TOPIC_LABELS, _save

REGION_COLORS = [f'C{index}' for index in range(10)]
TOPIC_MARKERS = dict(zip(TOPICS, 'os^D'))
LOG_PLANES = {'followers', 'price_usd'}


def _cells(ax, grid, values, cmap, edgecolors='gray', linewidths=.5, norm=None):
    polygons = [grid.layout.cell_vertices(center) for center in grid.positions]
    collection = PolyCollection(polygons, cmap=cmap, norm=norm, edgecolors=edgecolors, linewidths=linewidths)
    collection.set_array(np.ravel(values).astype(float))
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return collection


def plot_map(catalog, topics, output):
    """Izquierda: matriz U con creadores por neurona. Derecha: regiones con su composición temática."""
    som, grid = catalog.som, catalog.som.grid
    hits = np.bincount(catalog.winners, minlength=len(som.weights_))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    u_matrix = _cells(axes[0], grid, som.u_matrix(), 'viridis')
    fig.colorbar(u_matrix, ax=axes[0], shrink=.8, label='Distancia media a las neuronas vecinas')
    for (column, row), value, count in zip(grid.positions, som.u_matrix().ravel(), hits):
        axes[0].text(column, row, f'{value:.2f}\n{count} creador{"es" if count != 1 else ""}', ha='center', va='center', fontsize=8,
                     bbox={'facecolor': 'white', 'alpha': .7, 'edgecolor': 'none', 'pad': 1})
    axes[0].set_title('Matriz U (valor y ocupación por neurona)')

    colors = [REGION_COLORS[label] for label in catalog.neuron_regions]
    polygons = [grid.layout.cell_vertices(center) for center in grid.positions]
    axes[1].add_collection(PolyCollection(polygons, facecolors=colors, alpha=.35, edgecolors='white', linewidths=2))
    axes[1].autoscale_view()
    axes[1].invert_yaxis()
    axes[1].set_aspect('equal')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for neuron, ((column, row), label) in enumerate(zip(grid.positions, catalog.neuron_regions)):
        members = catalog.winners == neuron
        composition = ' '.join(f'{TOPIC_LABELS[t][:3]} {count}' for t, count in
                               zip(*np.unique(topics[members], return_counts=True))) if members.any() else 'vacía'
        axes[1].text(column, row, f'{catalog.region_names[label]}\n{composition}', ha='center', va='center',
                     fontsize=7)
    axes[1].set_title('Regiones (Ward sobre prototipos vecinos) y temas por neurona')
    _save(fig, output / 'mapa.png')


def plot_component_planes(grid, planes, output):
    """Un mapa por variable: el valor del prototipo de cada neurona, en unidades originales."""
    columns = 4
    rows = int(np.ceil(len(planes) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows))
    for ax, (name, values) in zip(axes.flat, planes.items()):
        norm = matplotlib.colors.LogNorm() if name in LOG_PLANES else None
        collection = _cells(ax, grid, np.maximum(values, 1) if norm else values, 'viridis', norm=norm)
        fig.colorbar(collection, ax=ax, shrink=.8)
        ax.set_title(name, fontsize=9)
    for ax in axes.flat[len(planes):]:
        ax.set_visible(False)
    fig.suptitle('Planos de componentes: prototipos en unidades originales (escala log en seguidores y precio)')
    _save(fig, output / 'planos.png')


def plot_regions_on_pca(frame, output):
    """Plano PC1–PC2 del biplot con cada creador coloreado por su región del SOM."""
    fig, ax = plt.subplots(figsize=(11, 8))
    regions = sorted(frame.region.unique())
    for index, region in enumerate(regions):
        members = frame[frame.region == region]
        for topic, marker in TOPIC_MARKERS.items():
            subset = members[members.main_topic == topic]
            ax.scatter(subset.pc1, subset.pc2, color=REGION_COLORS[index], marker=marker, s=45)
        ax.annotate(region, (members.pc1.mean(), members.pc2.mean()), fontsize=14, weight='bold',
                    color=REGION_COLORS[index], ha='center', va='center')
    region_handles = [plt.Line2D([], [], marker='o', ls='', color=REGION_COLORS[index]) for index in range(len(regions))]
    topic_handles = [plt.Line2D([], [], marker=marker, ls='', color='gray') for marker in TOPIC_MARKERS.values()]
    first = ax.legend(region_handles, regions, title='Región SOM', loc='upper left')
    ax.add_artist(first)
    ax.legend(topic_handles, [TOPIC_LABELS[topic] for topic in TOPICS], title='Tema principal', loc='upper right')
    ax.axhline(0, color='black', lw=.4)
    ax.axvline(0, color='black', lw=.4)
    ax.set(xlabel='PC1 (− visibilidad · + profundidad)', ylabel='PC2 (+ arte y exhibición · − audiencia argentina y perros)',
           title='Regiones del SOM sobre el plano de PCA')
    _save(fig, output / 'regiones_pca.png')


def plot_region_agreement(reference, partitions, output):
    """Creadores compartidos entre las regiones de 27 variables (filas) y las de cada representación (columnas)."""
    import pandas as pd
    columns = 2
    rows = int(np.ceil(len(partitions) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9, 4.2 * rows), squeeze=False)
    for ax in axes.flat[len(partitions):]:
        ax.set_visible(False)
    for ax, (label, regions) in zip(axes.flat, partitions.items()):
        table = pd.crosstab(pd.Series(reference, name='27 variables'), pd.Series(regions, name=label))
        ax.imshow(table.to_numpy(), cmap='Blues')
        for (row, column), value in np.ndenumerate(table.to_numpy()):
            if value:
                ax.text(column, row, value, ha='center', va='center', fontsize=8)
        ax.set_xticks(range(table.shape[1]), table.columns, fontsize=8)
        ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
        ax.set(title=label, xlabel=f'Región ({label})', ylabel='Región (27 variables)')
    fig.suptitle('Creadores en común con las regiones de 27 variables\n'
                 '(cada modelo numera R1–R5 por mediana de seguidores)')
    _save(fig, output / 'coincidencias.png')


def plot_regions_3d(frame, output):
    """PC1–PC3 coloreado por región del SOM: muestra qué separa el mapa más allá del plano del biplot."""
    from .plots import scatter_3d
    regions = sorted(frame.region.unique())
    colors = frame.region.map({region: REGION_COLORS[index] for index, region in enumerate(regions)}).to_numpy()
    fig = plt.figure(figsize=(15, 6.5))
    for position, (elevation, azimuth) in enumerate([(20, -60), (15, 30)], start=1):
        ax = fig.add_subplot(1, 2, position, projection='3d')
        scatter_3d(ax, frame[['pc1', 'pc2', 'pc3']].to_numpy(), colors, ['PC1', 'PC2', 'PC3'])
        ax.view_init(elevation, azimuth)
    handles = [plt.Line2D([], [], marker='o', ls='', color=REGION_COLORS[index]) for index in range(len(regions))]
    fig.axes[0].legend(handles, regions, title='Región SOM', loc='upper left')
    fig.suptitle('Regiones del SOM en PC1–PC3: dos vistas')
    _save(fig, output / 'regiones_pca_3d.png')
