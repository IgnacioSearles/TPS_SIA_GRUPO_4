"""Gráficos del EDA: recibe resultados calculados, sin entrenar modelos."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .representation import PCAAnalysis


def _save(fig, path, dpi=160):
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


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
    topics = frame.filter(regex='^topic_').to_numpy().argmax(axis=1)
    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(*points.T, c=topics, cmap='tab10')
    _annotate_creators(ax, frame, points)
    _draw_loadings(ax, analysis)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, ['Fútbol', 'Maquillaje', 'Perros', 'Arte'], title='Tema dominante')
    ratio = result.explained_variance_ratio
    ax.set(xlabel=f'PC1 ({ratio[0]:.1%}) / desvío PC1',
           ylabel=f'PC2 ({ratio[1]:.1%}) / desvío PC2',
           title='Biplot: flechas = correlaciones × 2 (escala visual)')
    ax.axhline(0, color='black', lw=.4)
    ax.axvline(0, color='black', lw=.4)
    _save(fig, output / 'biplot.png')


def _annotate_creators(ax, frame, points):
    for identifier, point in zip(frame.creator_id, points):
        ax.annotate(identifier, point, fontsize=6, alpha=.65)


def _draw_loadings(ax, analysis):
    for column, (horizontal, vertical) in zip(analysis.columns, analysis.pca.correlation_loadings[:, :2]):
        ax.arrow(0, 0, horizontal * 2, vertical * 2, color='gray', alpha=.5,
                 head_width=.025, length_includes_head=True)
        ax.text(horizontal * 2.08, vertical * 2.08, column, fontsize=7)


def plot_correlations(correlations, output):
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
    indices = range(len(correlations))
    ax.set_xticks(indices, correlations.columns, rotation=90, fontsize=6)
    ax.set_yticks(indices, correlations.columns, fontsize=6)
    fig.colorbar(image, ax=ax)
    _save(fig, output / 'correlaciones.png')


def plot_distributions(numeric, output):
    axes = numeric.hist(figsize=(20, 18), bins=10)
    _save(axes.flatten()[0].figure, output / 'distribuciones.png', dpi=130)
