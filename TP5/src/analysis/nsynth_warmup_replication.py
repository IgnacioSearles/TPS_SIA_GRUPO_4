"""Independent replication of warmup 0/50 with three new, paired NSynth seeds."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import traceback

import numpy as np

from src.analysis.nsynth_cnn_protocol import write_immutable,train_one
from src.data.art_search import assert_loss_unchanged

ROOT=Path(__file__).resolve().parents[2]
NEW_SEEDS=(37,47,57)

def prepare(root=ROOT):
    from src.experiments.runner import _prepare
    root=Path(root);assert_loss_unchanged(root)
    folder=root/'configs/nsynth_warmup_replication';audits=[]
    for architecture in ('mlp','conv'):
        original=json.loads((root/f'configs/nsynth_cnn/vae_{architecture}.json').read_text())
        assert original['training']['epochs']==200 and original['model']['kl']=={'weight':.5,'warmup_epochs':50}
        for warmup in (0,50):
            condition=f'vae_{architecture}'+('_no_warmup' if warmup==0 else '')
            config=copy.deepcopy(original);config['seeds']=list(NEW_SEEDS)
            config['name']=f'ej2_nsynth_cnn_{condition}';config['output']=f'results/{config["name"]}'
            config['model']['kl']['warmup_epochs']=warmup
            write_immutable(folder/f'{architecture}_warmup{warmup}.json',config)
            for seed in NEW_SEEDS:
                single=copy.deepcopy(config);single.pop('seeds');single['seed']=seed
                write_immutable(folder/'runs'/f'{seed:02d}_{architecture}_warmup{warmup}.json',single)
        for seed in NEW_SEEDS:
            audit_path=root/f'results/nsynth_warmup_replication/initialization/{architecture}_{seed}.json'
            if audit_path.exists():audits.append(json.loads(audit_path.read_text()));continue
            specs=[]
            for warmup in (0,50):
                c=json.loads((folder/'runs'/f'{seed:02d}_{architecture}_warmup{warmup}.json').read_text())
                specs.append({k:v for k,v in c.items() if k not in ('name','output')})
            check=copy.deepcopy(specs[0]);check['model']['kl']['warmup_epochs']=50
            assert check==specs[1],'Confounded warmup replication'
            a,b=(_prepare(c) for c in specs)
            assert np.array_equal(a.dataset.samples,b.dataset.samples)
            digest=hashlib.sha256()
            for x,y in zip(a.model.parameters(),b.model.parameters()):
                assert np.array_equal(x,y);digest.update(x.tobytes())
            for x,y in ((a.shuffle_rng,b.shuffle_rng),(a.model._rng,b.model._rng)):
                assert x.bit_generator.state==y.bit_generator.state
            record={'architecture':architecture,'seed':seed,'initial_weights_sha256':digest.hexdigest(),
                    'weights_and_training_rng_paired':True,'only_changed_hyperparameter':'warmup_epochs'}
            write_immutable(audit_path,record);audits.append(record)
            print(f'Paired replication initialization: {architecture}, seed={seed}',flush=True)
            del a,b
    protocol={'study':'nsynth_warmup_independent_replication_v1','new_seeds':list(NEW_SEEDS),
              'all_seeds':[7,17,27,37,47,57],'warmups':[0,50],'epochs':200,'final_beta':.5,
              'architectures':['mlp','conv'],'new_runs':12,'reused_runs':12,'initialization_audit':audits,
              'purpose':'Replication across seeds, not hyperparameter search; same fixed instrument-held-out split.',
              'decoder_note':'Remote codec default changed to PGHI; encoding, training inputs and MSE metrics unchanged. '
                             'Listening exports identify decoder phase and pair all methods under the same phase.'}
    write_immutable(root/'results/nsynth_warmup_replication/protocol.json',protocol)
    return protocol

def run(root=ROOT,jobs=6):
    root=Path(root);prepare(root);errors=[]
    tasks=[(f'vae_{a}'+('_no_warmup' if w==0 else ''),s,
            root/f'configs/nsynth_warmup_replication/runs/{s:02d}_{a}_warmup{w}.json')
           for s in NEW_SEEDS for a in ('mlp','conv') for w in (0,50)]
    with ProcessPoolExecutor(max_workers=jobs,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(train_one,str(p)):(c,s) for c,s,p in tasks}
        for future in as_completed(futures):
            c,s=futures[future]
            try:
                future.result()
                from src.analysis.nsynth_cnn_evaluation import evaluate_run
                from src.analysis.nsynth_warmup_report import report
                evaluate_run((str(root),c,s));report(root)
                print(f'Replication evaluated: {c}, seed={s}',flush=True)
            except Exception as exc:
                traceback.print_exc();errors.append({'condition':c,'seed':s,'error':str(exc)})
                (root/'results/nsynth_warmup_replication/errors.json').write_text(json.dumps(errors,indent=2))
    from src.analysis.nsynth_warmup_report import report
    report(root);assert_loss_unchanged(root)
    if errors:raise RuntimeError(f'{len(errors)} replication errors, see results/nsynth_warmup_replication/errors.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--jobs',type=int,default=6)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    if args.prepare:print(json.dumps(prepare(),indent=2))
    else:run(jobs=args.jobs)
