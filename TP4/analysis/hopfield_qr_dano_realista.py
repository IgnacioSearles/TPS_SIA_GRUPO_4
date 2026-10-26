"""Ej. 2: daño realista (desgaste, manchas irregulares, rayones y foto gastada).

Evalúa, sin reajustar parámetros, los métodos ya elegidos con ruido aleatorio:
- Hebb con los cinco QR completos.
- Desaprendizaje sin módulos compartidos: el candidato congelado (sueños, tasa 0,1,
  50 intentos, semillas de entrenamiento 0/1/2).
- Pseudoinversa, con QR completo y sin módulos compartidos.
- Sin red: el QR original a menor distancia de Hamming, y una variante asimétrica que
  sabe que el daño sólo borra tinta (un negro en la consulta no puede venir de un
  blanco del original). Ninguna de las dos es una memoria de Hopfield: son referencias.

Los daños están en `src/tp4/damage.py`. Semillas de consulta 200–219, nuevas: no se
usaron para elegir ningún método. No se guardan redes.

Uso desde TP4: python3 analysis/hopfield_qr_dano_realista.py --output reports/ejercicio2/dano_realista_20261026
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.hopfield_qr import MAX_ITERATIONS, QRS, load_patterns
from analysis.hopfield_qr_reader import Reader, Reading
from src.tp4.damage import BLACK, DAMAGES, WHITE
from src.tp4.hopfield import Hopfield
from src.tp4.qr import SIDE
from src.tp4.shared_modules import SharedModules
from src.tp4.unlearning import UnlearningTrainer

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(200, 220)
UNLEARNING = {'mode': 'dream', 'learning_rate': .1, 'attempts': 50, 'training_seeds': (0, 1, 2)}
ONE_SIDED = ('desgaste', 'desgaste_borde', 'mancha_blanca', 'rayones')
"""Daños que sólo borran tinta: para ellos la variante asimétrica es la decisión de
máxima verosimilitud entre los cinco QR."""
SOURCES = ('src/tp4/damage.py', 'src/tp4/hopfield.py', 'src/tp4/unlearning.py',
           'src/tp4/shared_modules.py', 'analysis/hopfield_qr_dano_realista.py')

NAMES = {'menu_principal': 'menú', 'vinos': 'vinos', 'cocteles_y_bebidas': 'cócteles',
         'postres': 'postres', 'especiales': 'especiales'}
TYPE_TITLES = {
    'desgaste': 'Desgaste (negro → blanco)', 'desgaste_borde': 'Desgaste mayor en un borde',
    'mancha_negra': 'Manchas negras irregulares', 'mancha_blanca': 'Manchas blancas (rasgado)',
    'rayones': 'Rayones', 'desgaste_y_mancha': 'Desgaste + mancha negra del 5 %',
    'foto': 'Foto gastada, binarizada',
}
LEVEL_LABELS = {
    'desgaste': 'probabilidad de perder tinta', 'desgaste_borde': 'probabilidad en el borde',
    'mancha_negra': 'fracción cubierta', 'mancha_blanca': 'fracción cubierta',
    'rayones': 'cantidad de rayones', 'desgaste_y_mancha': 'probabilidad de perder tinta',
    'foto': 'severidad',
}
STYLES = {
    'hebb': ('Hebb, QR completo', '#2a78d6', '-'),
    'desaprendizaje': ('Hebb + desaprendizaje, sin lo común', '#eb6834', '-'),
    'pseudoinversa': ('Pseudoinversa, QR completo', '#1a9e6e', '-'),
    'pseudoinversa_sin_compartidos': ('Pseudoinversa, sin lo común', '#7cc9a6', '-'),
    'vecino': ('Sin red: comparar con los originales', '#777777', '--'),
    'vecino_asimetrico': ('Sin red: comparar sabiendo que sólo se borra tinta', '#777777', ':'),
    'consulta': ('Sin red: el lector lee la consulta dañada', '#d6609b', ':'),
}


def digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def damaged_queries(patterns):
    queries = []
    for kind_index, (kind, (damage, levels)) in enumerate(DAMAGES.items()):
        for level_index, level in enumerate(levels):
            for origin, pattern in enumerate(patterns):
                for seed in SEEDS:
                    rng = np.random.default_rng((20261026, kind_index, level_index, origin, seed))
                    queries.append({'kind': kind, 'level': level, 'origin': origin, 'seed': seed,
                                    'pattern': damage(pattern, level, rng)})
    return queries


def nearest(patterns):
    return lambda query: (patterns[np.argmin((patterns != query).sum(axis=1))], 0, 'sin red')


def nearest_ink_loss(patterns):
    """Descarta los originales con blanco donde la consulta tiene negro (la tinta no
    aparece sola) y, entre los demás, elige el de menos tinta perdida."""
    def recover(query):
        impossible = ((query == BLACK) & (patterns == WHITE)).sum(axis=1)
        lost = ((query == WHITE) & (patterns == BLACK)).sum(axis=1)
        return patterns[np.lexsort((lost, impossible))[0]], 0, 'sin red'
    return recover


def network(model, shared=None):
    def recover(query):
        state = shared.strip(query) if shared is not None else query
        result = model.recall(state, max_iterations=MAX_ITERATIONS)
        output = shared.restore(result.state) if shared is not None else result.state
        return output, result.iterations, result.stop_reason
    return recover


def frozen_unlearning(active, seed):
    trainer = UnlearningTrainer(active, mode=UNLEARNING['mode'], learning_rate=UNLEARNING['learning_rate'],
                                seed=seed, max_iterations=MAX_ITERATIONS)
    for _ in range(UNLEARNING['attempts']):
        trainer.step()
    for pattern in active:
        result = trainer.network.recall(pattern, max_iterations=MAX_ITERATIONS)
        if result.stop_reason != 'stable' or not np.array_equal(result.state, pattern):
            raise ValueError('El candidato congelado no conserva estables los cinco QR')
    return trainer


def methods(patterns, shared):
    active = shared.strip(patterns)
    found = {'hebb': network(Hopfield(patterns)),
             'pseudoinversa': network(Hopfield(patterns, rule='pseudoinverse')),
             'pseudoinversa_sin_compartidos': network(Hopfield(active, rule='pseudoinverse'), shared),
             'vecino': nearest(patterns), 'vecino_asimetrico': nearest_ink_loss(patterns)}
    trainers = {}
    for seed in UNLEARNING['training_seeds']:
        trainers[seed] = frozen_unlearning(active, seed)
        found[f'desaprendizaje_{seed}'] = network(trainers[seed].network, shared)
    return found, {seed: t.updates for seed, t in trainers.items()}


class SafeReader(Reader):
    """Algunas versiones de OpenCV lanzan una excepción con contornos degenerados en
    vez de devolver «no leído»; se cuenta como una lectura fallida."""
    def read(self, pattern):
        try:
            return super().read(pattern)
        except cv2.error:
            return Reading(False, None)


def evaluate(patterns, queries, all_methods, shared):
    reader = SafeReader()
    originals = [reader.read(p).text for p in patterns]
    names = list(QRS)
    rows = []
    for number, query in enumerate(queries, 1):
        target = patterns[query['origin']]
        damaged = query['pattern'] != target
        base = {key: query[key] for key in ('kind', 'level', 'seed')}
        base.update(qr=names[query['origin']], query_hamming=int(damaged.sum()),
                    query_hamming_active=int(damaged[shared.differing].sum()))
        rows.append({**base, 'method': 'consulta', 'correct': not damaged.any(), 'hamming': base['query_hamming'],
                     'iterations': 0, 'stop_reason': 'sin red',
                     'read': reader.read(query['pattern']).text == originals[query['origin']]})
        for key, recover in all_methods.items():
            output, iterations, stop_reason = recover(query['pattern'])
            method, _, seed = key.rpartition('_') if key.startswith('desaprendizaje_') else (key, '', '')
            rows.append({**base, 'method': method, 'training_seed': int(seed) if seed else -1,
                         'correct': bool(np.array_equal(output, target)),
                         'hamming': int((output != target).sum()), 'iterations': iterations,
                         'stop_reason': stop_reason,
                         'read': reader.read(output).text == originals[query['origin']]})
        if number % 500 == 0:
            print(f'{number}/{len(queries)} consultas', flush=True)
    return pd.DataFrame(rows)


def summarize(results):
    return (results.groupby(['kind', 'level', 'method'], sort=False)
            .agg(consultas=('correct', 'size'), exacta=('correct', 'mean'), lector=('read', 'mean'),
                 modulos_mal=('hamming', 'mean'), iteraciones=('iterations', 'mean'),
                 dano=('query_hamming', 'mean'), dano_visible=('query_hamming_active', 'mean'))
            .reset_index())


def _decimal(value):
    return f'{value:g}'.replace('.', ',')


def plot_curves(summary, path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4), sharey=True)
    for ax, kind in zip(axes.flat, DAMAGES):
        rows = summary[summary.kind == kind]
        levels = list(DAMAGES[kind][1])
        positions = range(len(levels))
        for method, (label, color, style) in STYLES.items():
            if method == 'vecino_asimetrico' and kind not in ONE_SIDED:
                continue
            values = rows[rows.method == method].set_index('level').reindex(levels)
            column = 'lector' if method == 'consulta' else 'exacta'
            ax.plot(positions, 100 * values[column], color=color, linestyle=style, linewidth=2.2,
                    marker='o', markersize=4, label=label)
        damage = rows[rows.method == 'consulta'].set_index('level').reindex(levels)
        ax.set_xticks(list(positions), [f'{_decimal(l)}\n{d / (SIDE * SIDE):.0%}'.replace('.', ',')
                                        for l, d in zip(levels, damage.dano)], fontsize=8)
        ax.set_title(TYPE_TITLES[kind], fontsize=11)
        ax.set_xlabel(f'{LEVEL_LABELS[kind]}\n(abajo: módulos alterados)', fontsize=8)
        ax.set_ylim(-3, 103)
        ax.grid(axis='y', color='#dddddd', linewidth=.8)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Recuperación exacta (%)')
    legend = axes.flat[-1]
    legend.axis('off')
    legend.legend(*axes.flat[0].get_legend_handles_labels(), loc='center', frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_examples(patterns, queries, all_methods, path):
    shown = [('desaprendizaje_0', 'desaprendizaje'), ('pseudoinversa', 'pseudoinversa'), ('vecino', 'originales')]
    fig, axes = plt.subplots(len(DAMAGES), 2 + len(shown), figsize=(8.5, 2.05 * len(DAMAGES)))
    for row, kind in zip(axes, DAMAGES):
        level = DAMAGES[kind][1][3]
        query = next(q for q in queries if q['kind'] == kind and q['level'] == level and q['origin'] == 0)
        target = patterns[0]
        panels = [('original', target), (f'{TYPE_TITLES[kind]}', query['pattern'])]
        panels += [(title, all_methods[key](query['pattern'])[0]) for key, title in shown]
        for column, (ax, (title, image)) in enumerate(zip(row, panels)):
            ax.imshow(image.reshape(SIDE, SIDE), cmap='gray_r', vmin=-1, vmax=1, interpolation='nearest')
            wrong = int((image != target).sum())
            if column >= 2:
                title = f'{title}\n{"exacto" if wrong == 0 else f"{wrong} mal"}'
            elif column == 1:
                title = f'{title}\n{wrong} alterados'
            ax.set_title(title, fontsize=7.5)
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    patterns = load_patterns().astype(float)
    shared = SharedModules(patterns)
    queries = damaged_queries(patterns)
    all_methods, updates = methods(patterns, shared)
    (output / 'config.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'damages': {kind: list(levels) for kind, (_, levels) in DAMAGES.items()},
        'seeds': list(SEEDS), 'queries': len(queries), 'max_iterations': MAX_ITERATIONS,
        'unlearning': {**UNLEARNING, 'updates_by_seed': updates},
        'patterns_sha256': sha256(patterns.tobytes()).hexdigest(),
        'sources_sha256': {path: digest(path) for path in SOURCES},
    }, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(queries)} consultas; actualizaciones del desaprendizaje: {updates}', flush=True)
    results = evaluate(patterns, queries, all_methods, shared)
    results.to_csv(output / 'resultados.csv.gz', index=False)
    summary = summarize(results)
    summary.to_csv(output / 'resumen.csv', index=False)
    plot_curves(summary, output / 'curvas.png')
    plot_examples(patterns, queries, all_methods, output / 'ejemplos.png')
    (output / 'COMPLETED').write_text('Daño realista evaluado con los candidatos congelados.\n')
    print(summary.pivot_table(index=['kind', 'level'], columns='method', values='exacta').round(3).to_string())


if __name__ == '__main__':
    main()
