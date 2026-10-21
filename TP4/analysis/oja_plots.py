"""Gráficos del índice de Oja: reciben resultados calculados, sin entrenar modelos."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .plots import TOPIC_COLORS, _save, _topic_legend


def plot_convergence(runs, output):
    """1 − |cos(w, PC1)| por época: escala log para ver el error residual de cada tasa."""
    rates = sorted({run.learning_rate for run in runs}, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for color, rate in zip((f'C{index}' for index in range(len(rates))), rates):
        attempts = [run for run in runs if run.learning_rate == rate]
        trained = [run for run in attempts if run.history is not None]
        diverged = len(attempts) - len(trained)
        label = f'η₀ = {rate:g}' + (f' ({diverged} de {len(attempts)} divergieron)' if diverged else '')
        if not trained:
            ax.plot([], [], color=color, label=label)
            continue
        for position, run in enumerate(trained):
            misalignment = np.clip(1 - run.history.pc1_alignment.to_numpy(), 1e-16, None)
            ax.plot(run.history.epoch, misalignment, color=color, alpha=.7, lw=1,
                    label=label if position == 0 else None)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set(xlabel='Época', ylabel='1 − |cos(w, PC1)|',
           title='Convergencia de Oja hacia PC1 (5 semillas por tasa; tasa final = η₀ / 100)')
    ax.legend()
    _save(fig, output / 'convergencia.png')


def plot_coefficients(coefficients, output):
    positions = np.arange(len(coefficients))
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(positions - .2, coefficients.pc1, height=.4, label='PC1 (SVD)')
    ax.barh(positions + .2, coefficients.oja, height=.4, label='Oja')
    ax.set_yticks(positions, coefficients.index, fontsize=8)
    ax.axvline(0, color='black', lw=.5)
    ax.set(xlabel='Coeficiente del vector unitario', title='Pesos de Oja frente a los coeficientes de PC1')
    ax.legend()
    _save(fig, output / 'coeficientes.png')


def plot_index(index, output):
    ordered = index.sort_values('oja_index')
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.barh(np.arange(len(ordered)), ordered.oja_index, color=ordered.main_topic.map(TOPIC_COLORS))
    ax.set_yticks(np.arange(len(ordered)), ordered.name, fontsize=6)
    ax.axvline(0, color='black', lw=.5)
    ax.set(xlabel='Índice de Oja (− visibilidad / formato breve · + profundidad)',
           title='Índice de Oja por creador')
    _topic_legend(ax, loc='lower right')
    _save(fig, output / 'indice.png')


def plot_index_campaigns(index, output):
    panels = [('followers', 'Seguidores', True), ('cpm_arg', 'USD por 1.000 cuentas argentinas', True),
              ('argentina_audience_pct', '% audiencia argentina', False),
              ('content_tutorial_pct', '% tutoriales', False)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = index.main_topic.map(TOPIC_COLORS)
    for ax, (column, label, logarithmic) in zip(axes.flat, panels):
        ax.scatter(index.oja_index, index[column], c=colors, s=22)
        if logarithmic:
            ax.set_yscale('log')
        rho = index.oja_index.corr(index[column], method='spearman')
        ax.set(xlabel='Índice de Oja', ylabel=label, title=f'ρ Spearman = {rho:+.2f}')
    _topic_legend(axes.flat[0], fontsize=8)
    fig.suptitle('Qué ordena el índice: tamaño y profundidad sí; costo local y audiencia argentina no')
    _save(fig, output / 'indice_campanas.png')

