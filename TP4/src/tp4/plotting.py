"""Visualización independiente del entrenamiento y de la persistencia."""
import numpy as np


def plot_grid(path, grid, values, title):
    # Importación diferida: usar los modelos no requiere matplotlib.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    polygons = [grid.layout.cell_vertices(center) for center in grid.positions]
    collection = PolyCollection(polygons, cmap='viridis', edgecolor='gray')
    collection.set_array(values.ravel().astype(float))
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.add_collection(collection)
        _annotate_cells(ax, grid, values)
        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set(
            title=f'{title} — {grid.topology}',
            xlabel='Posición horizontal',
            ylabel='Posición vertical',
        )
        fig.colorbar(collection, ax=ax)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
    finally:
        plt.close(fig)


def _annotate_cells(ax, grid, values):
    for center, indices, value in zip(grid.positions, grid.indices, np.ravel(values)):
        column, row = center
        row_index, column_index = indices
        ax.text(
            column, row, f'{row_index},{column_index}\n{value:.2g}',
            ha='center', va='center', fontsize=7,
            bbox={'facecolor': 'white', 'alpha': .7, 'edgecolor': 'none', 'pad': 1},
        )


def plot_learning_curves(path, history):
    """Métricas de convergencia; se dibujan al finalizar o pausar, no por época."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .results import KohonenEpoch

    if not history:
        return
    fields = (
        [('quantization_error', 'Error de cuantización'),
         ('topographic_error', 'Error topográfico'), ('weight_change', 'Cambio de pesos')]
        if isinstance(history[0], KohonenEpoch) else
        [('pc1_alignment', 'Alineación absoluta con PC1'),
         ('explained_variance_ratio', 'Proporción de varianza capturada'),
         ('direction_change', 'Cambio de dirección')]
    )
    fig, axes = plt.subplots(1, len(fields), figsize=(14, 4))
    try:
        epochs = [epoch.epoch for epoch in history]
        for ax, (field, title) in zip(axes, fields):
            ax.plot(epochs, [getattr(epoch, field) for epoch in history])
            ax.set(title=title, xlabel='Época')
            ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
    finally:
        plt.close(fig)
