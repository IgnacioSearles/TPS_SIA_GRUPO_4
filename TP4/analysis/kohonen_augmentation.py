"""Augmentation local del SOM, con presupuesto de actualizaciones pareado.

python3 -m analysis.kohonen_augmentation --output reports/ejercicio1/augmentation_20261022/pilot
Las salidas contienen resultados, nunca pesos ni datos sintéticos.
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

from src.tp4.config import KohonenConfig
from src.tp4.kohonen import Kohonen
from src.tp4.preprocessing import Preprocessor
from .confirm_exercise1 import align_regions, overlap, regions, selections
from .eda import main_topic
from .kohonen_experiments import adjusted_rand_index
from .som_campanas import MODEL, _order_by_median
from .som_regions import contiguous_ward

ROOT = Path(__file__).resolve().parents[1]
STRENGTHS = (0., .05, .15, .30)


def local_neighbors(x, topics, count=3):
    """Solo filas de entrenamiento; vecino del mismo tema y distinto del propio."""
    neighbors = []
    for index, topic in enumerate(topics):
        candidates = np.flatnonzero((topics == topic) & (np.arange(len(x)) != index))
        if len(candidates):
            order = np.argsort(np.square(x[candidates]-x[index]).sum(axis=1), kind='stable')
            neighbors.append(candidates[order[:count]])
        else:
            # Un tema con un solo perfil no se inventa mezclándolo con otro tema.
            neighbors.append(np.array([index]))
    return neighbors


def perturb(x, neighbors, strength, rng):
    """Mitad de presentaciones intactas en esperanza; mezcla convexa en las restantes."""
    targets = np.array([rng.choice(options) for options in neighbors])
    alpha = rng.uniform(0., strength, len(x))
    alpha *= rng.random(len(x)) < .5
    return x + alpha[:, None] * (x[targets]-x)


class AugmentedKohonen(Kohonen):
    """Adaptador experimental: misma regla de Kohonen, sin checkpoints de augmentation."""
    def __init__(self, config, neighbors, strength):
        super().__init__(config)
        if not 0 <= strength <= 1:
            raise ValueError('strength debe estar entre cero y uno')
        if config.early_stopping.enabled:
            raise ValueError('El protocolo usa todas las actualizaciones, sin early stopping')
        self.neighbors, self.strength = neighbors, strength
        self.noise_rng = np.random.default_rng([config.seed, 20261022])
        self.displacement_sum = 0.
        self.displacement_count = 0

    def fit(self, x, *, checkpoint=None, on_epoch=None):
        if checkpoint is not None or on_epoch is not None:
            raise ValueError('Este adaptador experimental no guarda ni reanuda checkpoints')
        return super().fit(x)

    def _train_epoch(self, samples, rng, epoch):
        shown = samples if self.strength == 0 else perturb(samples, self.neighbors, self.strength, self.noise_rng)
        self.displacement_sum += float(np.linalg.norm(shown-samples, axis=1).sum())
        self.displacement_count += len(samples)
        record = super()._train_epoch(shown, rng, epoch)
        # Historial y selección de best_epoch describen los originales, no las copias.
        metrics = self.evaluate(samples)
        return replace(record, quantization_error=metrics.quantization_error,
                       topographic_error=metrics.topographic_error)


def fit_conditions(frame, task, strengths, reference):
    train = np.asarray(task['train'])
    excluded = np.setdiff1d(np.arange(len(frame)), train)
    prep = Preprocessor().fit(frame.iloc[train])
    x = prep.transform(frame)
    neighbors = local_neighbors(x[train], main_topic(frame.iloc[train]).to_numpy())
    baseline_choices = selections(frame, reference)
    output = []
    baseline_labels = None
    for strength in strengths:
        start = time.perf_counter()
        cfg = replace(MODEL, seed=task['seed'])
        model = AugmentedKohonen(cfg, neighbors, strength).fit(x[train])
        labels = regions(model, x)
        if strength == 0:
            baseline_labels = labels.copy()
        aligned = align_regions(labels, reference)
        chosen = selections(frame, aligned)
        output.append(dict(strength=strength, seconds=time.perf_counter()-start,
            train_metrics=asdict(model.evaluate(x[train])), excluded_metrics=asdict(model.evaluate(x[excluded])),
            ari_reference=adjusted_rand_index(labels, reference),
            ari_paired_baseline=adjusted_rand_index(labels, baseline_labels), labels=labels.tolist(),
            selection_overlap={topic: overlap(chosen[topic], baseline_choices[topic]) for topic in chosen},
            selection_sizes={topic: len(chosen[topic]) for topic in chosen},
            mean_displacement_z=model.displacement_sum/model.displacement_count,
            occupied_neurons=int((model.occupancy(x[train]) > 0).sum()),
            epochs=model.epochs_, updates=model.updates_))
    return dict(task=task, status='ok', conditions=output)


def worker(task, data, strengths, reference):
    return fit_conditions(pd.read_csv(data), task, strengths, np.asarray(reference))


def make_tasks(seed, model_seeds, samples=20):
    rng = np.random.default_rng(seed)
    tasks = []
    for sample in range(samples):
        train = sorted(rng.choice(50, 40, replace=False).tolist())
        tasks.extend(dict(sample=sample, seed=s, train=train) for s in model_seeds)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--selection', type=Path, help='Confirmar una selección congelada con nuevas particiones')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    strengths = list(STRENGTHS)
    sampling_seed, model_seeds = 20261023, [400, 401, 402]
    selection = None
    if args.selection:
        selection = json.loads(args.selection.read_text())
        if selection['selected_strength'] is None:
            raise ValueError('El piloto no habilitó ninguna intensidad para confirmar')
        strengths = [0., selection['selected_strength']]
        sampling_seed, model_seeds = 20261024, [500, 501, 502]
    tasks = make_tasks(sampling_seed, model_seeds)
    if args.smoke:
        tasks = tasks[:1]
    args.output.mkdir(parents=True, exist_ok=False)
    data = ROOT/'data/creadores.csv'
    frame = pd.read_csv(data)
    if len(frame) != 50:
        raise ValueError('El protocolo requiere 50 creadores')
    source_paths = list((ROOT/'src/tp4').glob('*.py')) + [Path(__file__),
        ROOT/'analysis/confirm_exercise1.py', ROOT/'analysis/som_regions.py',
        ROOT/'analysis/som_campanas.py', ROOT/'analysis/eda.py', ROOT/'analysis/kohonen_experiments.py']
    protocol = dict(smoke=args.smoke, stage='confirmation' if selection else 'pilot',
        data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        model=asdict(MODEL), strengths=strengths, tasks=tasks, fits=len(tasks)*len(strengths),
        sampling_seed=sampling_seed, neighbors=3, intact_probability=.5,
        primary='QE de los diez perfiles reales excluidos; menor es mejor',
        gates=dict(min_relative_qe_improvement=.01, min_split_win_fraction=.70,
                   max_te_increase=.02, max_art_overlap_decrease=.02),
        selection=selection,
        limitations=['Las particiones se superponen y provienen del mismo catálogo de 50.',
                     'ARI y campañas alinean etiquetas contra una referencia completa; son descriptivos.',
                     'No se usa ningún excluido para escalar, elegir vecinos ni generar copias.',
                     'Las columnas de entrenamiento no incluyen las derivadas descartadas por Preprocessor.',
                     'Interpolaciones coherentes no garantizan que los perfiles sintéticos existan.',
                     'El criterio de aceptación se fija antes de ver los resultados.'])
    (args.output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    x = Preprocessor().fit_transform(frame)
    reference = Kohonen(MODEL).fit(x)
    labels = contiguous_ward(reference.weights_, reference.grid.adjacency, 5)
    winners = reference.predict(x)
    labels = _order_by_median(labels, winners, frame.followers.to_numpy())[winners]
    (args.output/'reference.json').write_text(json.dumps(dict(labels=labels.tolist(),
        selections=selections(frame, labels)), indent=2)+'\n')
    print(f'Protocolo fijado: {protocol["fits"]} entrenamientos + referencia', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output/'runs.jsonl').open('w') as stream:
        futures = [pool.submit(worker, task, str(data), strengths, labels.tolist()) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            stream.write(json.dumps(result, allow_nan=False)+'\n')
            stream.flush()
            if completed % 5 == 0 or completed == len(tasks):
                print(f'{completed}/{len(tasks)} pares de condiciones; {completed*len(strengths)} modelos', flush=True)


if __name__ == '__main__':
    main()
