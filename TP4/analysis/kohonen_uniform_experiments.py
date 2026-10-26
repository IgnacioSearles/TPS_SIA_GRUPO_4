"""Repite las pruebas recientes de Kohonen con dos vecindades uniformes.

Reutiliza controles gaussianos solo si coinciden hashes de datos y fuentes.
No escribe pesos ni toca presentation/. Desde TP4:
python3 -m analysis.kohonen_uniform_experiments --output reports/ejercicio1/uniforme_20261026
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from src.tp4.kohonen import Kohonen
from src.tp4.preprocessing import Preprocessor
from .confirm_exercise1 import CANDIDATES, align_regions, overlap, regions, selections
from .eda import main_topic, cost_per_thousand_argentine_reach
from .kohonen_augmentation import AugmentedKohonen, local_neighbors
from .kohonen_experiments import adjusted_rand_index
from .som_campanas import MODEL

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {'uniform_matched': .5, 'uniform_neighbors': 1.}
CONFIRMATION = Path('reports/ejercicio1/confirmacion_20261022')
AUGMENTATION = Path('reports/ejercicio1/augmentation_20261022/pilot')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(directory):
    return {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def task_key(task):
    return json.dumps(task, sort_keys=True)


def config_for(variant, seed=0, candidate='rect4x4'):
    rows, cols, topology = CANDIDATES[candidate]
    return replace(MODEL, seed=seed, rows=rows, cols=cols, topology=topology,
                   neighborhood='uniform', radius_start=2., radius_end=VARIANTS[variant])


def verify_legacy(directory, root=ROOT):
    protocol = json.loads((directory/'protocol.json').read_text())
    if protocol['smoke']:
        raise ValueError('Un control smoke no reemplaza un experimento completo')
    if digest(root/'data/creadores.csv') != protocol['data_sha256']:
        raise ValueError('Cambió el catálogo respecto del control gaussiano')
    for name, expected in protocol['sources'].items():
        if digest(root/name) != expected:
            raise ValueError(f'Cambió una fuente del control: {name}')
    rows = [json.loads(s) for s in (directory/'runs.jsonl').read_text().splitlines()]
    if sorted(task_key(r['task']) for r in rows) != sorted(task_key(t) for t in protocol['tasks']):
        raise ValueError('El control no coincide exactamente con sus tareas')
    if any(r['status'] != 'ok' for r in rows):
        raise ValueError('El control contiene fallos')
    return protocol, rows


def load_controls():
    first, normal = verify_legacy(ROOT/CONFIRMATION)
    second, augmentation = verify_legacy(ROOT/AUGMENTATION)
    reference = json.loads((ROOT/CONFIRMATION/'reference.json').read_text())
    other = json.loads((ROOT/AUGMENTATION/'reference.json').read_text())
    if reference['labels'] != other['labels']:
        raise ValueError('Las dos referencias gaussianas no coinciden')
    tasks, controls = [], []
    for row in normal:
        if row['task']['stage'] not in ('finalists', 'subsamples'):
            continue
        task = row['task']
        tasks.append(task)
        controls.append(dict(task=task, variant='gaussian', strength=0., **{k:v for k,v in row.items() if k != 'task'}))
    for row in augmentation:
        task = dict(stage='augmentation', **row['task'])
        tasks.append(task)
        if [c['strength'] for c in row['conditions']] != second['strengths']:
            raise ValueError('Condiciones de augmentation incompletas')
        for condition in row['conditions']:
            controls.append(dict(task=task, variant='gaussian', status='ok', **condition))
    return tasks, controls, reference, first, second


def measure(model, x, train, frame, reference, original_labels=None):
    labels = regions(model, x)
    aligned = align_regions(labels, np.asarray(reference['labels']))
    chosen = selections(frame, aligned)
    excluded = np.setdiff1d(np.arange(len(frame)), train)
    output = dict(train_metrics=asdict(model.evaluate(x[train])),
        ari_reference=adjusted_rand_index(labels, reference['labels']),
        labels=labels.tolist(), selections=chosen,
        selection_overlap={topic:overlap(chosen[topic], reference['selections'][topic]) for topic in chosen},
        selection_sizes={topic:len(chosen[topic]) for topic in chosen},
        occupied_neurons=int((model.occupancy(x[train]) > 0).sum()),
        occupied_regions=int(len(np.unique(labels[train]))),
        epochs=model.epochs_, updates=model.updates_)
    if len(excluded):
        output['excluded_metrics'] = asdict(model.evaluate(x[excluded]))
        output['ari_excluded_reference'] = adjusted_rand_index(labels[excluded], np.asarray(reference['labels'])[excluded])
    if original_labels is not None:
        output['ari_paired_baseline'] = adjusted_rand_index(labels, original_labels)
    return output


def run_task(task, data, reference):
    frame = pd.read_csv(data)
    train = np.asarray(task.get('train', list(range(len(frame)))))
    # Ni el escalado ni los vecinos usan los diez excluidos.
    prep = Preprocessor().fit(frame.iloc[train])
    x = prep.transform(frame)
    augmented = task['stage'] == 'augmentation'
    strengths = (0., .05, .15, .30) if augmented else (0.,)
    neighbors = local_neighbors(x[train], main_topic(frame.iloc[train]).to_numpy()) if augmented else None
    results = []
    for variant in VARIANTS:
        original_labels = None
        cfg = config_for(variant, task['seed'], task.get('candidate', 'rect4x4'))
        for strength in strengths:
            start = time.perf_counter()
            try:
                model = (AugmentedKohonen(cfg, neighbors, strength) if augmented else Kohonen(cfg)).fit(x[train])
                if augmented and strength == 0:
                    original_labels = regions(model, x)
                measured = measure(model, x, train, frame, reference, original_labels)
                if augmented:
                    measured['mean_displacement_z'] = model.displacement_sum/model.displacement_count
                results.append(dict(variant=variant, strength=strength, status='ok',
                                    seconds=time.perf_counter()-start, **measured))
            except (ValueError, FloatingPointError) as error:
                results.append(dict(variant=variant, strength=strength, status='failed',
                                    seconds=time.perf_counter()-start, error=f'{type(error).__name__}: {error}'))
    return dict(task=task, results=results)


def reference_models(frame, reference):
    x = Preprocessor().fit_transform(frame)
    outputs = {}
    for variant in VARIANTS:
        cfg = config_for(variant)
        model = Kohonen(cfg).fit(x)
        measured = measure(model, x, np.arange(len(frame)), frame, reference)
        aligned = align_regions(measured['labels'], reference['labels'])
        topics = main_topic(frame).to_numpy()
        cost = cost_per_thousand_argentine_reach(frame).to_numpy()
        profiles = []
        for label in range(5):
            indices = np.flatnonzero(aligned == label)
            profile = dict(aligned_region=f'R{label+1}', creators=len(indices),
                           topics={topic:int(np.count_nonzero(topics[indices] == topic)) for topic in reference['selections']})
            if len(indices):
                profile.update(followers_median=float(frame.followers.iloc[indices].median()),
                    argentina_audience_pct_median=float(frame.argentina_audience_pct.iloc[indices].median()),
                    tutorial_pct_median=float(frame.content_tutorial_pct.iloc[indices].median()),
                    cpm_arg_median=float(np.median(cost[indices])))
            profiles.append(profile)
        outputs[variant] = dict(config=asdict(cfg), profiles=profiles, u_matrix=model.u_matrix().tolist(), **measured)
    return outputs


def check_reused_controls(tasks, controls, data, reference):
    """Tres réplicas gaussianas: finalista, submuestra y augmentation, antes de reutilizar."""
    frame = pd.read_csv(data)
    checks = []
    for stage in ('finalists', 'subsamples', 'augmentation'):
        task = next(t for t in tasks if t['stage'] == stage)
        strength = .15 if stage == 'augmentation' else 0.
        saved = next(c for c in controls if c['task'] == task and c['strength'] == strength)
        train = np.asarray(task.get('train', list(range(len(frame)))))
        prep = Preprocessor().fit(frame.iloc[train])
        x = prep.transform(frame)
        rows, cols, topology = CANDIDATES[task.get('candidate', 'rect4x4')]
        cfg = replace(MODEL, seed=task['seed'], rows=rows, cols=cols, topology=topology, radius_start=2.)
        if stage == 'augmentation':
            neighbors = local_neighbors(x[train], main_topic(frame.iloc[train]).to_numpy())
            model = AugmentedKohonen(cfg, neighbors, strength).fit(x[train])
        else:
            model = Kohonen(cfg).fit(x[train])
        now = measure(model, x, train, frame, reference)
        np.testing.assert_array_equal(now['labels'], saved['labels'])
        for metric in ('quantization_error', 'topographic_error'):
            np.testing.assert_allclose(now['train_metrics'][metric], saved['train_metrics'][metric], rtol=0, atol=1e-12)
            if 'excluded_metrics' in saved:
                np.testing.assert_allclose(now['excluded_metrics'][metric], saved['excluded_metrics'][metric], rtol=0, atol=1e-12)
        checks.append(dict(stage=stage, task=task, strength=strength, reproduced=True))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    tasks, controls, reference, _, augmentation_protocol = load_controls()
    if args.smoke:
        tasks = [next(t for t in tasks if t['stage'] == stage) for stage in ('finalists','subsamples','augmentation')]
        controls = [c for c in controls if c['task'] in tasks]
    sources = list((ROOT/'src/tp4').glob('*.py')) + [Path(__file__), ROOT/'analysis/confirm_exercise1.py',
        ROOT/'analysis/kohonen_augmentation.py', ROOT/'analysis/kohonen_experiments.py',
        ROOT/'analysis/som_regions.py', ROOT/'analysis/som_campanas.py', ROOT/'analysis/eda.py']
    protocol = dict(smoke=args.smoke, tasks=tasks, variants={v:asdict(config_for(v)) for v in VARIANTS},
        fits=sum(8 if t['stage']=='augmentation' else 2 for t in tasks), reference_fits=2,
        data_sha256=digest(ROOT/'data/creadores.csv'), sources={str(p.relative_to(ROOT)):digest(p) for p in sources},
        legacy={str(p.relative_to(ROOT)):digest(p) for directory in (ROOT/CONFIRMATION, ROOT/AUGMENTATION)
                for p in (directory/'protocol.json',directory/'runs.jsonl',directory/'reference.json')},
        augmentation_gates=augmentation_protocol['gates'],
        presentation_fingerprint=fingerprint(ROOT/'presentation'),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        boundary='distancia de grilla <= radio; incluye vecinas inmediatas a radio 1',
        limitations=['Solo cambian neighborhood y, en uniform_neighbors, radius_end.',
            'Tasa exponencial 0.3 a 0.01, no se agregó la variante 1/i del apunte.',
            'Con radio 0.5 uniforme queda solo la ganadora; con radio 1 quedan adyacentes.',
            'Las campañas conservan reglas y referencia gaussianas; la alineación es descriptiva.',
            'No se repite Oja porque no depende de la vecindad del SOM.',
            'Se repiten finalistas, 100 submuestras y augmentation recientes; no los 1500 barridos históricos.',
            'No se promueve un modelo ni augmentation automáticamente con estas pruebas.'])
    if args.resume:
        saved = json.loads((args.output/'protocol.json').read_text())
        if saved != protocol:
            raise ValueError('El protocolo o las fuentes cambiaron; no se puede reanudar')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
        (args.output/'gaussian_reference.json').write_text(json.dumps(reference, indent=2)+'\n')
        with (args.output/'controls.jsonl').open('w') as stream:
            for row in controls:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        checks = check_reused_controls(tasks, controls, ROOT/'data/creadores.csv', reference)
        (args.output/'control_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
        refs = reference_models(pd.read_csv(ROOT/'data/creadores.csv'), reference)
        (args.output/'uniform_references.json').write_text(json.dumps(refs, indent=2)+'\n')
    completed = []
    runs_path = args.output/'runs.jsonl'
    if runs_path.exists():
        completed = [json.loads(s) for s in runs_path.read_text().splitlines()]
    done = {task_key(row['task']) for row in completed}
    pending = [t for t in tasks if task_key(t) not in done]
    print(f'Plan: {protocol["fits"]} modelos + 2 referencias; pendientes {len(pending)}/{len(tasks)} bloques', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, runs_path.open('a') as stream:
        futures = [pool.submit(run_task,t,str(ROOT/'data/creadores.csv'),reference) for t in pending]
        for count, future in enumerate(as_completed(futures), len(completed)+1):
            row = future.result()
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            failed = [r for r in row['results'] if r['status'] != 'ok']
            if count % 20 == 0 or failed or count == len(tasks):
                print(f'{count}/{len(tasks)} bloques; etapa {row["task"]["stage"]}; fallos bloque {len(failed)}', flush=True)
    if fingerprint(ROOT/'presentation') != protocol['presentation_fingerprint']:
        raise RuntimeError('Cambió presentation/ durante el experimento; revisar antes de concluir')
    (args.output/'COMPLETED').write_text('Finalizó el plan; el consolidado verifica tareas, métricas y fallos.\n')


if __name__ == '__main__':
    main()
