"""Matched KL warmup ablation: six new runs, six existing NSynth controls."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from src.analysis.nsynth_cnn_protocol import write_immutable,train_one
from src.data.art_search import assert_loss_unchanged

ROOT=Path(__file__).resolve().parents[2]
SEEDS=(7,17,27)

def without_identifiers(value):
    result=copy.deepcopy(value)
    result.pop('name',None);result.pop('output',None)
    return result

def prepare(root=ROOT):
    from src.experiments.runner import _prepare
    root=Path(root);assert_loss_unchanged(root)
    folder=root/'configs/nsynth_warmup'
    records=[]
    for architecture in ('mlp','conv'):
        control=json.loads((root/f'configs/nsynth_cnn/vae_{architecture}.json').read_text())
        assert control['training']['epochs']==200
        assert control['model']['kl']=={'weight':.5,'warmup_epochs':50}
        assert control['seeds']==list(SEEDS)
        new=copy.deepcopy(control)
        new['name']=f'ej2_nsynth_cnn_vae_{architecture}_no_warmup'
        new['output']=f'results/{new["name"]}'
        new['model']['kl']['warmup_epochs']=0
        compared=without_identifiers(new);compared['model']['kl']['warmup_epochs']=50
        assert compared==without_identifiers(control),'Confounded ablation'
        write_immutable(folder/f'vae_{architecture}_no_warmup.json',new)
        for seed in SEEDS:
            single=copy.deepcopy(new);single.pop('seeds');single['seed']=seed
            write_immutable(folder/'runs'/f'{seed:02d}_vae_{architecture}_no_warmup.json',single)
            signature_path=root/f'results/nsynth_warmup_study/initialization/{architecture}_{seed}.json'
            if signature_path.exists():
                records.append(json.loads(signature_path.read_text()));continue
            first=without_identifiers(single)
            second=copy.deepcopy(first);second['model']['kl']['warmup_epochs']=50
            a=_prepare(first);b=_prepare(second)
            assert a.dataset.labels==b.dataset.labels
            assert np.array_equal(a.dataset.samples,b.dataset.samples)
            digest=hashlib.sha256()
            for x,y in zip(a.model.parameters(),b.model.parameters()):
                assert np.array_equal(x,y),'Initial weights differ'
                digest.update(x.tobytes())
            assert a.shuffle_rng.bit_generator.state==b.shuffle_rng.bit_generator.state
            assert a.corruption_rng.bit_generator.state==b.corruption_rng.bit_generator.state
            assert a.model._rng.bit_generator.state==b.model._rng.bit_generator.state
            record={'architecture':architecture,'seed':seed,'initial_weights_sha256':digest.hexdigest(),
                    'initial_weights_equal':True,'shuffle_corruption_sampling_rng_equal':True,
                    'only_changed_hyperparameter':'model.kl.warmup_epochs:50 -> 0'}
            write_immutable(signature_path,record);records.append(record)
            print(f'Paired initialization verified: {architecture} seed={seed}',flush=True)
            del a,b
    protocol={'study':'nsynth_warmup_v1','warmups':[0,50],'final_beta':.5,'epochs':200,
              'architectures':['mlp','conv'],'seeds':list(SEEDS),'new_runs':6,'reused_controls':6,
              'initialization_audit':records,'heldout_split':'same exact instruments as main NSynth study',
              'metrics':'heldout reconstruction, standard-prior generation, posterior KL and active latent dimensions',
              'interpretation':'Paired training from scratch; no inference from fine-tuning or cancelled beta pilot. '
                               'Total losses before epoch50 have different beta and are not directly comparable.'}
    write_immutable(root/'results/nsynth_warmup_study/protocol.json',protocol)
    return protocol

def run(root=ROOT,jobs=2):
    root=Path(root);prepare(root)
    tasks=[(f'vae_{architecture}_no_warmup',seed,root/f'configs/nsynth_warmup/runs/{seed:02d}_vae_{architecture}_no_warmup.json')
           for seed in SEEDS for architecture in ('mlp','conv')]
    with ProcessPoolExecutor(max_workers=jobs,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(train_one,str(path)):(condition,seed) for condition,seed,path in tasks}
        for future in as_completed(futures):
            condition,seed=futures[future];future.result()
            from src.analysis.nsynth_cnn_evaluation import evaluate_run
            from src.analysis.nsynth_warmup_report import report
            evaluate_run((str(root),condition,seed));report(root)
            print(f'Warmup ablation completed: {condition} seed={seed}',flush=True)
    from src.experiments.runner import run_experiments
    run_experiments([root/f'configs/nsynth_warmup/vae_{a}_no_warmup.json' for a in ('mlp','conv')])
    from src.analysis.nsynth_warmup_report import report
    report(root);assert_loss_unchanged(root)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    if args.prepare:print(json.dumps(prepare(),indent=2))
    else:run(jobs=args.jobs)
