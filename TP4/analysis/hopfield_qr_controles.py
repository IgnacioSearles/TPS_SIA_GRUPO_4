"""Ej. 2, controles: ¿el problema es la red o son los QR?

Misma red, mismo tamaño (N = 1369) y mismas consultas que el barrido principal, pero
con patrones generados al azar en lugar de los QR:

1. Cinco patrones al azar (casi ortogonales) contra los cinco QR.
2. Cinco patrones al azar que comparten una fracción fija de sus módulos, para
   aislar el efecto del parecido.
3. Cuántos patrones entran: patrones al azar contra subconjuntos de los QR.

Uso desde TP4: python3 analysis/hopfield_qr_controles.py
Salidas (regenerables, fuera de git) en reports/ejercicio2/controles/.
"""
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.hopfield_qr import MAX_ITERATIONS, NOISE_LEVELS, QRS, SEEDS, _decimal, _save, load_patterns
from src.tp4.hopfield import Hopfield
from src.tp4.queries import make_queries, noisy_query
from src.tp4.recall_outcome import classify
from src.tp4.shared_modules import SharedModules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ejercicio2' / 'controles'

SHARED_FRACTIONS = (0, .1, .2, .22, .24, .26, .28, .3, .4, .5, .6, .7, .8, .9)
SHARED_NOISES = (0, .1, .25)
CAPACITY_COUNTS = (1, 2, 3, 5, 10, 25, 50, 100, 150, 175, 200, 250, 300)
CAPACITY_TICKS = (1, 2, 3, 5, 10, 25, 50, 100, 200, 300)
CAPACITY_NOISE = .1
CAPACITY_QUERIES = 100  # consultas por cantidad de patrones al azar
CAPACITY_FRACTION = .15

# Paleta categórica de referencia (dataviz): al azar en el slot 1, QR en el 2,
# QR sin módulos compartidos en el 3.
RANDOM, QR, QR_STRIPPED = '#2a78d6', '#eb6834', '#1baf7a'
SEQUENTIAL = ['#86b6ef', '#3987e5', '#0d366b']  # azul, claro → oscuro
INK, MUTED = '#1f1f1f', '#888888'


def random_patterns(rng, count, neurons, shared_fraction=0.):
    """`count` patrones ±1 al azar; una fracción `shared_fraction` de los módulos
    (elegidos al azar) vale lo mismo en todos."""
    patterns = rng.choice([-1., 1.], size=(count, neurons))
    shared = rng.permutation(neurons)[:round(shared_fraction * neurons)]
    patterns[:, shared] = patterns[0, shared]
    return patterns


def recall_correct(network, query, origin):
    result = network.recall(query, max_iterations=MAX_ITERATIONS)
    return classify(result.state, result.stop_reason, network.patterns_, origin).category == 'correct'


def max_overlap(patterns):
    overlaps = Hopfield(patterns).overlaps()
    return float(np.abs(overlaps[np.triu_indices(len(patterns), 1)]).max())


def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.8)
    ax.set_axisbelow(True)


# --- Control 1: cinco patrones al azar contra los cinco QR -------------------------

def random_vs_qr(qrs):
    """Barrido de ruido completo con los cinco QR y con cinco patrones al azar del
    mismo tamaño, con el mismo generador de consultas."""
    rows = []
    sets = {'al_azar': random_patterns(np.random.default_rng(0), len(qrs), qrs.shape[1]), 'qr': qrs}
    for key, patterns in sets.items():
        network = Hopfield(patterns)
        for query in make_queries(patterns, NOISE_LEVELS, SEEDS):
            rows.append({'patterns': key, 'noise': query.noise, 'seed': query.seed, 'origin': query.origin,
                         'correct': recall_correct(network, query.pattern, query.origin)})
    overlaps = {key: max_overlap(patterns) for key, patterns in sets.items()}
    return pd.DataFrame(rows), overlaps


def plot_random_vs_qr(summary, overlaps, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = {'al_azar': ('5 patrones al azar', RANDOM), 'qr': ('5 QR del restaurante', QR)}
    for key, (label, color) in labels.items():
        rows = summary[key]
        ax.plot(rows.index, rows, color=color, linewidth=2, marker='o', markersize=5,
                label=f'{label} (solapamiento máx. {_decimal(round(overlaps[key], 2))})')
    ax.set_xticks(NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS], fontsize=8)
    ax.set(xlabel='Nivel de ruido p (fracción de módulos invertidos)', ylabel='Fracción de recuperaciones correctas',
           ylim=(-.03, 1.03), title='Control 1: misma red y mismo N = 1369, distintos patrones')
    _style(ax)
    ax.legend(loc='center left', frameon=False, fontsize=9)
    _save(fig, path)


# --- Control 2: patrones al azar con módulos compartidos ---------------------------

def shared_fraction_sweep(neurons, count):
    """Para cada fracción compartida y cada semilla, un juego nuevo de patrones al
    azar; cada patrón se consulta con los ruidos de SHARED_NOISES."""
    rows = []
    for fraction in SHARED_FRACTIONS:
        for seed in SEEDS:
            rng = np.random.default_rng((1, round(fraction * 100), seed))
            patterns = random_patterns(rng, count, neurons, fraction)
            network = Hopfield(patterns)
            for noise in SHARED_NOISES:
                for origin, pattern in enumerate(patterns):
                    query = noisy_query(pattern, noise, rng)
                    rows.append({'shared_fraction': fraction, 'seed': seed, 'noise': noise, 'origin': origin,
                                 'correct': recall_correct(network, query, origin)})
    return pd.DataFrame(rows)


def plot_shared_fraction(summary, qr_fraction, stored, path):
    """La línea punteada gris es el umbral 1/(k − 1), con k patrones guardados: en un
    módulo no compartido en el que los otros k − 1 valen lo contrario, el crosstalk
    alineado vale −(k − 1)·f contra una señal de 1, así que con f > 1/(k − 1) ningún
    patrón es estable (y esos módulos son, en promedio, (1 − f)/2^(k−1) de N)."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for color, noise in zip(SEQUENTIAL, SHARED_NOISES):
        rows = summary[noise]
        ax.plot(rows.index, rows, color=color, linewidth=2, marker='o', markersize=5,
                label=f'patrones al azar, ruido p = {_decimal(noise)}')
    threshold = 1 / (stored - 1)
    ax.axvline(threshold, color=MUTED, linewidth=1, linestyle=':')
    ax.text(threshold + .01, .85, f'umbral teórico\n1/(patrones − 1) = {threshold:.0%}', color=MUTED, fontsize=8)
    ax.axvline(qr_fraction, color=QR, linewidth=1.5, linestyle='--')
    ax.text(qr_fraction + .01, .55, f'los QR del\nrestaurante:\n{qr_fraction:.1%} compartido'.replace('.', ','),
            color=INK, fontsize=8)
    ticks = [f for f in SHARED_FRACTIONS if round(f * 100) % 10 == 0]
    ax.set_xticks(ticks, [f'{f:.0%}' for f in ticks], fontsize=8)
    ax.set(xlabel='Fracción de módulos iguales en los 5 patrones', ylabel='Fracción de recuperaciones correctas',
           ylim=(-.03, 1.03), title='Control 2: 5 patrones al azar que comparten parte de sus módulos')
    _style(ax)
    ax.legend(loc='lower left', frameon=False, fontsize=8)
    _save(fig, path)


# --- Control 3: capacidad --------------------------------------------------------

def random_capacity(neurons):
    """Patrones al azar: para cada cantidad k, la fracción de consultas con ruido
    CAPACITY_NOISE que vuelven exactamente a su patrón."""
    rows = []
    for count in CAPACITY_COUNTS:
        rng = np.random.default_rng((2, count))
        network = Hopfield(random_patterns(rng, count, neurons))
        for trial in range(CAPACITY_QUERIES):
            origin = int(rng.integers(count))
            query = noisy_query(network.patterns_[origin], CAPACITY_NOISE, rng)
            rows.append({'patterns': 'al_azar', 'stored': count, 'correct': recall_correct(network, query, origin)})
    return rows


def qr_capacity(qrs, strip, key):
    """QR: para cada k, todos los subconjuntos de k QR, cada QR del subconjunto con
    las 20 semillas de ruido CAPACITY_NOISE del barrido principal."""
    level = list(NOISE_LEVELS).index(CAPACITY_NOISE)
    rows = []
    for count in range(1, len(qrs) + 1):
        for subset in combinations(range(len(qrs)), count):
            network = Hopfield(strip(qrs[list(subset)]))
            for position, origin in enumerate(subset):
                for seed in SEEDS:
                    rng = np.random.default_rng((origin, level, seed))  # misma consulta que make_queries
                    query = strip(noisy_query(qrs[origin], CAPACITY_NOISE, rng))
                    rows.append({'patterns': key, 'stored': count,
                                 'correct': recall_correct(network, query, position)})
    return rows


def plot_capacity(summary, neurons, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = {'al_azar': (f'patrones al azar (N = {neurons})', RANDOM, 'o'),
              'qr': (f'QR completos (N = {neurons})', QR, 's'),
              'qr_sin_compartidos': ('QR sin módulos compartidos (N = 414)', QR_STRIPPED, '^')}
    for key, (label, color, marker) in labels.items():
        rows = summary[key].dropna()
        ax.plot(rows.index, rows, color=color, linewidth=2, marker=marker, markersize=6, label=label)
    limit = CAPACITY_FRACTION * neurons
    ax.axvline(limit, color=MUTED, linestyle='--', linewidth=1)
    ax.text(limit * 1.03, .5, f'0,15·N ≈ {limit:.0f}\n(cota del apunte)', color=MUTED, fontsize=8)
    ax.set_xscale('log')
    ax.set_xticks(CAPACITY_TICKS, [str(k) for k in CAPACITY_TICKS], fontsize=8)
    ax.minorticks_off()
    ax.set(xlabel='Patrones guardados (escala logarítmica)',
           ylabel=f'Fracción correcta con ruido p = {_decimal(CAPACITY_NOISE)}', ylim=(-.03, 1.03),
           title='Control 3: cuántos patrones entran en la red')
    _style(ax)
    ax.legend(loc='center left', bbox_to_anchor=(0, .4), frameon=False, fontsize=8)
    _save(fig, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qrs = load_patterns().astype(float)
    neurons = qrs.shape[1]
    shared = SharedModules(qrs)
    qr_fraction = 1 - shared.count / neurons

    results, overlaps = random_vs_qr(qrs)
    results.to_csv(OUT / 'control1_resultados.csv', index=False)
    summary = results.pivot_table(index='noise', columns='patterns', values='correct')
    summary.to_csv(OUT / 'control1_resumen.csv')
    plot_random_vs_qr(summary, overlaps, OUT / 'control1_azar_vs_qr.png')
    print('Control 1: solapamiento máximo', {k: round(v, 3) for k, v in overlaps.items()})
    print(summary.round(3).to_string())

    shared_results = shared_fraction_sweep(neurons, len(qrs))
    shared_results.to_csv(OUT / 'control2_resultados.csv', index=False)
    shared_summary = shared_results.pivot_table(index='shared_fraction', columns='noise', values='correct')
    shared_summary.to_csv(OUT / 'control2_resumen.csv')
    plot_shared_fraction(shared_summary, qr_fraction, len(qrs), OUT / 'control2_compartidos.png')
    print(f'Control 2 (fracción compartida en los QR: {qr_fraction:.3f})')
    print(shared_summary.round(3).to_string())

    capacity = pd.DataFrame(random_capacity(neurons) + qr_capacity(qrs, np.asarray, 'qr')
                            + qr_capacity(qrs, shared.strip, 'qr_sin_compartidos'))
    capacity.to_csv(OUT / 'control3_resultados.csv', index=False)
    capacity_summary = capacity.pivot_table(index='stored', columns='patterns', values='correct')
    capacity_summary.to_csv(OUT / 'control3_resumen.csv')
    plot_capacity(capacity_summary, neurons, OUT / 'control3_capacidad.png')
    print('Control 3:')
    print(capacity_summary.round(3).to_string())


if __name__ == '__main__':
    main()
