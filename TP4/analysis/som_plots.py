"""Gráficos del mapa de Kohonen: reciben el modelo y las regiones ya calculadas."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from .eda import TOPICS
from .plots import TOPIC_COLORS, TOPIC_LABELS, _save, _topic_legend

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

    draw_region_map(axes[1], catalog)
    for neuron, (column, row) in enumerate(grid.positions):
        members = catalog.winners == neuron
        composition = ' '.join(f'{TOPIC_LABELS[t][:3]} {count}' for t, count in
                               zip(*np.unique(topics[members], return_counts=True))) if members.any() else 'vacía'
        axes[1].text(column, row + .1, composition, ha='center', va='center', fontsize=7)
    axes[1].set_title('Regiones (Ward sobre prototipos vecinos) y temas por neurona')
    _save(fig, output / 'mapa.png')


def draw_region_map(ax, catalog, facecolors='whitesmoke'):
    """Grilla con los bordes de las regiones y su nombre. Las regiones no usan color de relleno, porque los
    colores del informe identifican temas; facecolors permite pintar cada neurona con otra variable."""
    grid = catalog.som.grid
    polygons = [grid.layout.cell_vertices(center) for center in grid.positions]
    ax.add_collection(PolyCollection(polygons, facecolors=facecolors, edgecolors='white', linewidths=1))
    draw_region_borders(ax, catalog)
    for label, region in enumerate(catalog.region_names):
        # La etiqueta va en la neurona de la región más cercana a su centro: el centro puede caer en un borde.
        cells = grid.positions[catalog.neuron_regions == label]
        column, row = cells[np.linalg.norm(cells - cells.mean(axis=0), axis=1).argmin()]
        ax.text(column, row - .36, region, ha='center', va='center', weight='bold', fontsize=12,
                bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 2})
    ax.autoscale_view()
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def draw_region_borders(ax, catalog):
    """Línea gruesa en cada lado compartido por dos neuronas vecinas de regiones distintas."""
    grid = catalog.som.grid
    first, second = np.where(np.triu(grid.adjacency, 1))
    for a, b in zip(first, second):
        if catalog.neuron_regions[a] == catalog.neuron_regions[b]:
            continue
        vertices_b = grid.layout.cell_vertices(grid.positions[b])
        shared = [vertex for vertex in grid.layout.cell_vertices(grid.positions[a])
                  if np.min(np.linalg.norm(vertices_b - vertex, axis=1)) < 1e-6]
        if len(shared) == 2:
            ax.plot(*np.array(shared).T, color='black', lw=3)


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
    """Plano PC1–PC2 del biplot: cada creador con el color de su tema y cada región como el contorno de sus creadores."""
    from scipy.spatial import ConvexHull
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(frame.pc1, frame.pc2, c=frame.main_topic.map(TOPIC_COLORS), s=45)
    for region in sorted(frame.region.unique()):
        points = frame.loc[frame.region == region, ['pc1', 'pc2']].to_numpy()
        hull = points[ConvexHull(points).vertices]
        ax.fill(*hull.T, facecolor='none', edgecolor='dimgray', lw=1.5, ls='--')
        ax.text(*points.mean(axis=0), region, fontsize=13, weight='bold', ha='center', va='center',
                bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 2})
    _topic_legend(ax, loc='upper right')
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
    """PC1–PC3 con el color de cada tema y el nombre de cada región en su centro: muestra qué separa el mapa
    más allá del plano del biplot."""
    from .plots import scatter_3d
    scores = frame[['pc1', 'pc2', 'pc3']].to_numpy()
    colors = frame.main_topic.map(TOPIC_COLORS).to_numpy()
    centers = frame.groupby('region')[['pc1', 'pc2', 'pc3']].mean()
    fig = plt.figure(figsize=(15, 6.5))
    for position, (elevation, azimuth) in enumerate([(20, -60), (15, 30)], start=1):
        ax = fig.add_subplot(1, 2, position, projection='3d')
        scatter_3d(ax, scores, colors, ['PC1', 'PC2', 'PC3'])
        for region, center in centers.iterrows():
            ax.text(*center, region, fontsize=12, weight='bold', ha='center',
                    bbox={'facecolor': 'white', 'edgecolor': 'black', 'pad': 2})
        ax.view_init(elevation, azimuth)
    _topic_legend(fig.axes[0], loc='upper left')
    fig.suptitle('Regiones del SOM en PC1–PC3: dos vistas')
    _save(fig, output / 'regiones_pca_3d.png')
