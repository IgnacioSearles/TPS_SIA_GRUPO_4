"""Confirmaciones nuevas del ejercicio 1; conserva métricas, nunca pesos.

Desde TP4: OPENBLAS_NUM_THREADS=1 python3 -m analysis.confirm_exercise1 --workers 3
El manifiesto se escribe antes de entrenar. La salida existente nunca se sobrescribe.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from itertools import permutations
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from src.tp4.config import KohonenConfig, OjaConfig
from src.tp4.kohonen import Kohonen
from src.tp4.oja import Oja
from src.tp4.pca import compute_pca
from src.tp4.preprocessing import Preprocessor
from .kohonen_experiments import adjusted_rand_index
from .som_regions import contiguous_ward
from .som_campanas import MODEL, TARGET_REGIONS, _order_by_median
from .eda import main_topic, cost_per_thousand_argentine_reach

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {'rect4x4': (4, 4, 'rectangular'),
              'hex4x4': (4, 4, 'hexagonal'), 'hex2x8': (2, 8, 'hexagonal')}


def align_regions(labels, reference):
    """Renombrado descriptivo; no modifica SOM ni interviene en ARI."""
    labels, reference = np.asarray(labels), np.asarray(reference)
    table = np.zeros((5, 5), dtype=int)
    np.add.at(table, (labels, reference), 1)
    mapping = max(permutations(range(5)), key=lambda p: sum(table[i, p[i]] for i in range(5)))
    return np.asarray(mapping)[labels]


def common_direction(direction, fitted_scale, reference_scale):
    """Coeficientes en unidades z de la referencia, para comparar escaladores distintos."""
    vector = direction * reference_scale / fitted_scale
    return vector / np.linalg.norm(vector)


def selections(frame, labels):
    topics = main_topic(frame).to_numpy()
    cost = cost_per_thousand_argentine_reach(frame).to_numpy()
    result = {}
    for topic, names in TARGET_REGIONS.items():
        allowed = [int(name[1:]) - 1 for name in names]
        candidates = np.flatnonzero((topics == topic) & np.isin(labels, allowed))
        result[topic] = candidates[np.argsort(cost[candidates], kind='stable')][:3].tolist()
    return result


def overlap(first, second):
    # Si el modelo no logra tres candidatos no se premia una lista corta idéntica.
    return len(set(first) & set(second)) / 3


def regions(model, x):
    labels = contiguous_ward(model.weights_, model.grid.adjacency, 5)
    return labels[model.predict(x)]


def jobs(frame):
    tasks = []
    for name in CANDIDATES:
        tasks.extend(dict(stage='finalists', candidate=name, seed=s) for s in range(100, 130))
    rng = np.random.default_rng(20261022)
    for sample in range(100):
        indices = sorted(rng.choice(len(frame), 40, replace=False).tolist())
        tasks.extend(dict(stage='subsamples', sample=sample, train=indices, seed=s)
                     for s in (200, 201, 202))
    for rate in (.0001, .001, .01):
        tasks.extend(dict(stage='oja_seeds', rate=rate, seed=s) for s in range(100, 110))
    followers = frame.followers.to_numpy()
    extreme_order = np.argsort(-followers, kind='stable')
    for removed in (1, 3):
        keep = np.setdiff1d(np.arange(len(frame)), extreme_order[:removed]).tolist()
        for scaling in ('frozen', 'refit'):
            tasks.extend(dict(stage='oja_extremes', removed=removed, scaling=scaling,
                              train=keep, rate=.001, seed=s) for s in range(100, 110))
    return tasks


def run_som(task, frame, preprocessor, x, reference):
    if task['stage'] == 'finalists':
        rows, cols, topology = CANDIDATES[task['candidate']]
        train, test = np.arange(len(frame)), np.array([], dtype=int)
    else:
        rows, cols, topology = CANDIDATES['rect4x4']
        train = np.asarray(task['train'])
        test = np.setdiff1d(np.arange(len(frame)), train)
        # Cada muestra ajusta su propio escalador, sin los diez excluidos.
        preprocessor = Preprocessor().fit(frame.iloc[train])
        x = preprocessor.transform(frame)
    cfg = KohonenConfig(rows=rows, cols=cols, topology=topology, seed=task['seed'],
                        epochs=1000, schedule_epochs=500, radius_start=2.)
    model = Kohonen(cfg).fit(x[train])
    labels = regions(model, x)
    aligned = align_regions(labels, reference)
    chosen, baseline = selections(frame, aligned), selections(frame, reference)
    result = dict(train_metrics=asdict(model.evaluate(x[train])),
                  ari_reference=adjusted_rand_index(labels, reference),
                  occupied_neurons=int((model.occupancy(x[train]) > 0).sum()),
                  labels=labels.tolist(), aligned_labels=aligned.tolist(), selections=chosen,
                  selection_overlap={topic: overlap(chosen[topic], baseline[topic]) for topic in chosen},
                  epochs=model.epochs_, updates=model.updates_)
    if len(test):
        result['excluded_metrics'] = asdict(model.evaluate(x[test]))
        result['ari_excluded_reference'] = adjusted_rand_index(labels[test], reference[test])
    return result


def run_oja(task, frame, reference_preprocessor, reference_x):
    train = np.asarray(task.get('train', list(range(len(frame)))))
    preprocessor = reference_preprocessor
    if task.get('scaling') == 'refit':
        preprocessor = Preprocessor().fit(frame.iloc[train])
    x = preprocessor.transform(frame)
    model = Oja(OjaConfig(seed=task['seed'], epochs=1000,
                         learning_rate_start=task['rate'], learning_rate_end=task['rate']/100)).fit(x[train])
    own_pc1 = compute_pca(x[train]).components[0]
    reference_pc1 = compute_pca(reference_x).components[0]
    own_common = common_direction(own_pc1, preprocessor.scale_, reference_preprocessor.scale_)
    learned_common = common_direction(model.weights_, preprocessor.scale_, reference_preprocessor.scale_)
    sign = 1 if learned_common @ reference_pc1 >= 0 else -1
    learned_scores = reference_x @ (sign * learned_common)
    reference_scores = reference_x @ reference_pc1
    aligned_own = own_common * (1 if own_common @ reference_pc1 >= 0 else -1)
    own_scores = reference_x @ aligned_own
    first = lambda t: next((h.epoch for h in model.history_ if h.pc1_alignment >= t), None)
    return dict(pc1_alignment=float(abs(model.weights_ @ own_pc1)),
                first_epoch_0999=first(.999), first_epoch_099999=first(.99999),
                pc1_drift_cosine=float(abs(own_common @ reference_pc1)),
                learned_reference_cosine=float(abs(learned_common @ reference_pc1)),
                score_rank_correlation=float(pd.Series(learned_scores).rank().corr(pd.Series(reference_scores).rank())),
                pca_score_rank_correlation=float(pd.Series(own_scores).rank().corr(pd.Series(reference_scores).rank())),
                pc1_explained_variance=float(compute_pca(x[train]).explained_variance_ratio[0]),
                epochs=model.epochs_, updates=model.updates_)


def execute(task, data, reference):
    start = time.perf_counter()
    frame = pd.read_csv(data)
    preprocessor = Preprocessor().fit(frame)
    x = preprocessor.transform(frame)
    try:
        if task['stage'] in ('finalists', 'subsamples'):
            result = run_som(task, frame, preprocessor, x, np.asarray(reference))
        else:
            result = run_oja(task, frame, preprocessor, x)
        return dict(task=task, status='ok', seconds=time.perf_counter()-start, **result)
    except (FloatingPointError, ValueError) as error:
        return dict(task=task, status='failed', seconds=time.perf_counter()-start,
                    error=f'{type(error).__name__}: {error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/ejercicio1/confirmacion_20261022')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--smoke', action='store_true', help='Un trabajo por etapa; directorio separado obligatorio')
    args = parser.parse_args()
    data = ROOT/'data/creadores.csv'
    frame = pd.read_csv(data)
    if len(frame) != 50:
        raise ValueError('El protocolo requiere el catálogo de 50 creadores')
    tasks = jobs(frame)
    if args.smoke:
        tasks = [next(t for t in tasks if t['stage'] == stage)
                 for stage in ('finalists', 'subsamples', 'oja_seeds', 'oja_extremes')]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = list((ROOT/'src/tp4').glob('*.py')) + [Path(__file__),
               ROOT/'analysis/som_regions.py', ROOT/'analysis/som_campanas.py',
               ROOT/'analysis/kohonen_experiments.py', ROOT/'analysis/eda.py']
    manifest = dict(data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                    sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    smoke=args.smoke, tasks=tasks, som_base=asdict(MODEL), region_count=5,
                    notes=['Métricas y etiquetas por posición de fila, sin nombres ni pesos.',
                           'ARI y selecciones contra seed 0 son estabilidad descriptiva.',
                           'Submuestras: escalador ajustado solo con las 40 filas de entrenamiento.',
                           'QE entre submuestras usa escalas distintas: no son comparables directamente.',
                           'Oja extremos: quitar top 1 o top 3 por followers; no remover según resultados.',
                           'Diez semillas Oja por condición no son diez catálogos independientes.'])
    (args.output/'protocol.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    x = Preprocessor().fit_transform(frame)
    reference = Kohonen(MODEL).fit(x)
    labels = contiguous_ward(reference.weights_, reference.grid.adjacency, 5)
    winners = reference.predict(x)
    labels = _order_by_median(labels, winners, frame.followers.to_numpy())[winners]
    (args.output/'reference.json').write_text(json.dumps(dict(labels=labels.tolist(),
        metrics=asdict(reference.evaluate(x)), selections=selections(frame, labels)), indent=2)+'\n')
    print(f'Protocolo fijado: {len(tasks)} corridas + 1 referencia; {args.output}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output/'runs.jsonl').open('w') as stream:
        futures = [pool.submit(execute, task, str(data), labels.tolist()) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            if completed % 10 == 0 or result['status'] != 'ok' or completed == len(tasks):
                print(f'{completed}/{len(tasks)}: {result["task"]} {result["status"]}', flush=True)


if __name__ == '__main__':
    main()
