"""Ej. 2, experimentos de ortogonalidad: los mismos que el análisis de letras de
otro grupo (sus notas no están en el repo), aplicados a los QR.

1. Ranking de todos los subconjuntos de QR por ortogonalidad, y cuánto recupera cada uno.
2. Mapa del crosstalk de cada neurona: QR contra patrones al azar.
3. Paso a paso con 15 % y 40 % de ruido, con la energía de cada paso.
4. Tasa de recuperación por QR: la terna más ortogonal contra la menos ortogonal.
5. Más neuronas: los QR al doble de resolución.

Las ternas usan la variante sin módulos compartidos (con los QR completos ninguna
terna es estable). Actualización sincrónica.

Uso desde TP4: python3 analysis/hopfield_qr_ortogonalidad.py
Salidas (regenerables, fuera de git) en reports/ejercicio2/ortogonalidad/.
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

from analysis.hopfield_qr import (MAX_ITERATIONS, NOISE_LEVELS, QRS, SEEDS, TITLES, _decimal, _save,
                                  find_query, load_patterns)
from analysis.hopfield_qr_controles import random_patterns
from analysis.hopfield_qr_variante import variants
from src.tp4.hopfield import Hopfield
from src.tp4.qr import SIDE, pattern_to_image
from src.tp4.queries import make_queries
from src.tp4.recall_outcome import classify
from src.tp4.shared_modules import SharedModules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ejercicio2' / 'ortogonalidad'

SUBSET_SIZES = (2, 3, 4)
STOP_LABELS = {'stable': 'estado estable', 'cycle': 'ciclo', 'max_iterations': 'tope'}
SCALE = 2  # cada módulo pasa a SCALE × SCALE neuronas
SCALE_NOISES = (0, .1, .2, .3)

# Paleta categórica de referencia (dataviz), en orden fijo.
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
DIVERGING = ['#2a78d6', '#f0efec', '#e34948']  # en contra (azul) · 0 · a favor (rojo)
INK, MUTED = '#1f1f1f', '#888888'


def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=.8)
    ax.set_axisbelow(True)


def _label(subset):
    return ' · '.join(TITLES[list(QRS)[i]] for i in subset)


def overlap_stats(patterns):
    """Máximo y promedio del solapamiento absoluto entre pares (sin la diagonal)."""
    overlaps = np.abs(patterns @ patterns.T / patterns.shape[1])[np.triu_indices(len(patterns), 1)]
    return overlaps.max(), overlaps.mean()


# --- Experimento 1: ranking de subconjuntos ----------------------------------------

def subset_ranking(patterns, queries, spaces):
    """Para cada variante y cada subconjunto de 2 a 4 QR: solapamiento máximo y medio,
    y fracción de recuperaciones correctas por nivel de ruido (consultas del barrido principal)."""
    rows = []
    for key, space in spaces.items():
        for size in SUBSET_SIZES:
            for subset in combinations(range(len(patterns)), size):
                network = Hopfield(space.strip(patterns[list(subset)]))
                worst, mean = overlap_stats(network.patterns_)
                for query in (q for q in queries if q.origin in subset):
                    result = network.recall(space.strip(query.pattern), max_iterations=MAX_ITERATIONS)
                    outcome = classify(result.state, result.stop_reason, network.patterns_,
                                       subset.index(query.origin))
                    rows.append({'variant': key, 'stored': size, 'subset': subset, 'max_overlap': worst,
                                 'mean_overlap': mean, 'qr': list(QRS)[query.origin], 'noise': query.noise,
                                 'correct': outcome.category == 'correct', 'category': outcome.category})
    return pd.DataFrame(rows)


def ranking_table(results):
    """Una fila por (variante, subconjunto), ordenadas como en las notas: por solapamiento
    máximo y, a igualdad, por el promedio. `correct_low_noise` promedia p de 0 a 0,3."""
    low = results[results.noise <= .3]
    table = low.groupby(['variant', 'stored', 'subset', 'max_overlap', 'mean_overlap']).correct.mean()
    table = table.rename('correct_low_noise').reset_index()
    table['subset'] = table.subset.map(_label)
    return table.sort_values(['variant', 'stored', 'max_overlap', 'mean_overlap'])


def plot_ranking(table, spaces, path):
    """Solapamiento máximo del subconjunto contra la fracción correcta: un punto por subconjunto."""
    markers = {2: 'o', 3: 's', 4: '^'}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for key, space in spaces.items():
        for size in SUBSET_SIZES:
            rows = table[(table.variant == key) & (table.stored == size)]
            ax.scatter(rows.max_overlap, rows.correct_low_noise, s=60, marker=markers[size], color=space.color,
                       edgecolor='white', linewidth=1.5, zorder=3,
                       label=f'{space.label.split(":")[1].strip().split(" (")[0]}, {size} QR')
    ax.set(xlabel='Solapamiento máximo entre dos QR del subconjunto (|m|)',
           ylabel='Fracción correcta (ruido p de 0 a 0,3)', ylim=(-.05, 1.05),
           title='Experimento 1: cuanto menos ortogonal el subconjunto, peor recupera')
    _style(ax)
    ax.legend(loc='center right', frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


# --- Experimento 2: mapa de crosstalk ---------------------------------------------

def plot_crosstalk_maps(qrs, randoms, path):
    """ξᵢ·Cᵢ por neurona, con los cinco guardados: negativo = el crosstalk empuja en
    contra del QR; por debajo de −1 la neurona es inestable."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('divergente', DIVERGING)
    rows = [('QR del restaurante', Hopfield(qrs), [TITLES[name] for name in QRS]),
            ('Patrones al azar', Hopfield(randoms), [f'patrón {i + 1}' for i in range(len(randoms))])]
    fig, axes = plt.subplots(2, len(qrs), figsize=(2.6 * len(qrs) + 1, 6.2))
    for row, (name, network, titles) in zip(axes, rows):
        for index, (ax, title) in enumerate(zip(row, titles)):
            aligned = (network.patterns_[index] * network.crosstalk(index)).reshape(SIDE, SIDE)
            image = ax.imshow(aligned, cmap=cmap, vmin=-3, vmax=3, interpolation='nearest')
            unstable = int(np.sum(aligned < -(1 - len(network.patterns_) / network.patterns_.shape[1])))
            ax.set_title(f'{title}\n|C| medio {_decimal(round(float(np.abs(aligned).mean()), 2))} · '
                         f'{unstable} inestables', fontsize=8)
            ax.axis('off')
        row[0].text(-8, SIDE / 2, name, rotation=90, va='center', ha='center', fontsize=10)
    fig.colorbar(image, ax=axes, fraction=.02, pad=.02,
                 label='Crosstalk alineado ξᵢ·Cᵢ (la señal propia vale +1)')
    fig.suptitle('Experimento 2: crosstalk en cada neurona, con los cinco guardados\n'
                 'rojo = empuja a favor del patrón · azul = en contra · menos de −1 = la neurona se da vuelta',
                 fontsize=10)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


# --- Experimento 3: paso a paso --------------------------------------------------

def plot_steps(stored, queries, space, origin, cases, title, path):
    """Una fila por (ruido, semilla): consulta, cada iteración hasta detenerse y la energía.
    Los pasos se muestran con los módulos compartidos repuestos; la consulta, tal como llega."""
    network = Hopfield(space.strip(stored))
    recalls = []
    for noise, seed in cases:
        query = find_query(queries, origin, noise, seed).pattern
        recalls.append((noise, seed, query, network.recall(space.strip(query), max_iterations=MAX_ITERATIONS,
                                                           return_states=True)))
    columns = max(len(r.states) for *_, r in recalls) + 2
    fig, axes = plt.subplots(len(cases), columns, figsize=(2.3 * columns, 2.8 * len(cases)), squeeze=False)
    original = find_query(queries, origin, 0, 0).pattern  # la consulta con p = 0 es el QR original
    for row, (noise, seed, query, result) in zip(axes, recalls):
        panels = [('Original', original)]
        for step, state in enumerate(result.states):
            restored = query if step == 0 else space.restore(state)
            hamming = int(np.sum(restored != original))
            label = f'Consulta, p = {_decimal(noise)}, semilla {seed}' if step == 0 else f'Paso {step}'
            panels.append((f'{label}\n{hamming} módulos mal', restored))
        for ax in row:
            ax.axis('off')
        for ax, (name, pattern) in zip(row, panels):
            ax.imshow(pattern_to_image(pattern), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
            ax.set_title(name, fontsize=8)
        energy = row[-1]
        energy.axis('on')
        energy.plot(range(len(result.energies)), result.energies, color=CATEGORICAL[0], linewidth=2,
                    marker='o', markersize=6)
        energy.set_title(f'Energía ({STOP_LABELS[result.stop_reason]})',
                         fontsize=8)
        energy.set_xlabel('Paso', fontsize=8)
        energy.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        energy.tick_params(labelsize=7)
        _style(energy)
    fig.suptitle(title, fontsize=10)
    _save(fig, path)


def failing_seed(stored, queries, space, origin, noise):
    """Primera semilla con la que la consulta de `origin` no se recupera."""
    network = Hopfield(space.strip(stored))
    target = space.strip(find_query(queries, origin, 0, 0).pattern)
    for seed in SEEDS:
        state = network.recall(space.strip(find_query(queries, origin, noise, seed).pattern),
                               max_iterations=MAX_ITERATIONS).state
        if not np.array_equal(state, target):
            return seed
    return None


# --- Experimento 4: tasa por QR, terna más y menos ortogonal ---------------------

def plot_per_qr(results, table, path):
    triples = table[(table.variant == 'sin_compartidos') & (table.stored == 3)]
    best, worst = triples.iloc[0], triples.iloc[-1]
    labels = {_label(s): s for s in results.subset.unique()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, row, kind in zip(axes, (best, worst), ('más ortogonal', 'menos ortogonal')):
        subset = labels[row.subset]
        rows = results[(results.variant == 'sin_compartidos') & (results.subset == subset)]
        for origin in subset:
            name = list(QRS)[origin]
            rate = rows[rows.qr == name].groupby('noise').correct.mean()
            ax.plot(rate.index, rate, color=CATEGORICAL[origin], linewidth=2, marker='o', markersize=5,
                    label=TITLES[name])
        ax.set_xticks(NOISE_LEVELS, [_decimal(p) for p in NOISE_LEVELS], fontsize=8)
        ax.set(xlabel='Nivel de ruido p', ylim=(-.03, 1.03),
               title=f'Terna {kind} (máx. |m| = {_decimal(round(row.max_overlap, 2))})')
        _style(ax)
        ax.legend(loc='lower left', frameon=False, fontsize=8)
    axes[0].set_ylabel('Tasa de recuperación')
    fig.suptitle('Experimento 4: tasa de recuperación por QR (sin módulos compartidos, 3 QR guardados)')
    _save(fig, path)


# --- Experimento 5: más resolución ------------------------------------------------

def upscale(patterns, scale):
    """Cada módulo pasa a ser un bloque de scale × scale neuronas iguales."""
    grids = patterns.reshape(len(patterns), SIDE, SIDE)
    return np.kron(grids, np.ones((scale, scale))).reshape(len(patterns), -1)


def resolution_table(qrs):
    rows = []
    for scale in (1, SCALE):
        patterns = upscale(qrs, scale)
        network = Hopfield(patterns)
        worst, mean = overlap_stats(patterns)
        mixture = np.sign(patterns.sum(axis=0))
        stable = sum(network.recall(p, max_iterations=MAX_ITERATIONS).iterations == 0 for p in patterns)
        for query in make_queries(patterns, SCALE_NOISES, SEEDS):
            result = network.recall(query.pattern, max_iterations=MAX_ITERATIONS)
            rows.append({'scale': scale, 'neurons': patterns.shape[1], 'capacity': .15 * patterns.shape[1],
                         'max_overlap': worst, 'mean_overlap': mean, 'stable_qrs': stable,
                         'noise': query.noise,
                         'correct': np.array_equal(result.state, patterns[query.origin]),
                         'mixture': np.array_equal(result.state, mixture)})
    results = pd.DataFrame(rows)
    return results.groupby(['scale', 'neurons', 'capacity', 'max_overlap', 'mean_overlap', 'stable_qrs']).agg(
        recalls=('correct', 'size'), correct=('correct', 'mean'), mixture=('mixture', 'mean')).reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qrs = load_patterns().astype(float)
    shared = SharedModules(qrs)
    spaces = variants(shared)
    queries = make_queries(qrs, NOISE_LEVELS, SEEDS)

    results = subset_ranking(qrs, queries, spaces)
    table = ranking_table(results)
    table.to_csv(OUT / 'ranking_subconjuntos.csv', index=False)
    results.drop(columns='subset').assign(subset=results.subset.map(_label)).to_csv(
        OUT / 'resultados_subconjuntos.csv', index=False)
    plot_ranking(table, spaces, OUT / '1_ranking.png')
    print(table.round(3).to_string(index=False))

    randoms = random_patterns(np.random.default_rng(0), len(qrs), qrs.shape[1])
    plot_crosstalk_maps(qrs, randoms, OUT / '2_crosstalk.png')

    triples = table[(table.variant == 'sin_compartidos') & (table.stored == 3)]
    best = next(s for s in results.subset.unique() if _label(s) == triples.iloc[0].subset)
    origin = best[0]
    seed40 = failing_seed(qrs[list(best)], queries, spaces['sin_compartidos'], origin, .4)
    print(f'Terna más ortogonal: {_label(best)}; semilla con p = 0,4 que falla: {seed40}')
    plot_steps(qrs, queries, spaces['completo'], origin, [(.15, 0), (.4, 0)],
               f'Experimento 3a: QR {TITLES[list(QRS)[origin]]}, los cinco QR completos guardados',
               OUT / '3a_paso_a_paso_completo.png')
    plot_steps(qrs[list(best)], queries, spaces['sin_compartidos'], origin,
               [(.15, 0), (.4, 0)] + ([(.4, seed40)] if seed40 is not None else []),
               f'Experimento 3b: QR {TITLES[list(QRS)[origin]]}, sin módulos compartidos, '
               f'guardado con {_label(best)}', OUT / '3b_paso_a_paso_variante.png')

    plot_per_qr(results, table, OUT / '4_tasa_por_qr.png')

    resolution = resolution_table(qrs)
    resolution.to_csv(OUT / 'resolucion.csv', index=False)
    print(resolution.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
