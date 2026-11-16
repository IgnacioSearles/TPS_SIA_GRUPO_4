"""Matched NSynth CNN/dense reconstruction, generation and denoising study.

No changes to existing models, training loss, or the main experiment runner.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import json
from multiprocessing import get_context
from pathlib import Path
import traceback

from src.data.art_search import assert_loss_unchanged

ROOT=Path(__file__).resolve().parents[2]
CONDITIONS=('vae_mlp','vae_conv','dae_mlp','dae_conv')
SEEDS=(7,17,27)

def make_configuration(settings,condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    is_vae=condition.startswith('vae')
    architecture=condition.split('_')[1]
    model={'type':'variational' if is_vae else 'autoencoder','architecture':architecture,
           'latent_size':16,'hidden_layers':[512,128],'activation':'relu',
           'initializer':'he','loss':'mse','precision':'float32'}
    if is_vae:
        model['kl']={'weight':.5,'warmup_epochs':50}
    if architecture=='conv':
        model.update(image_shape=[128,64,1],conv_channels=[8,16])
    training={'epochs':200,'batch_size':64,'optimizer':{'type':'adam','learning_rate':.001},'log_every':10}
    if not is_vae:
        training['noise']={'type':'gaussian','level':.1}
    return {'name':f'ej2_nsynth_cnn_{condition}','output':f'results/ej2_nsynth_cnn_{condition}',
            'seeds':list(SEEDS),'dataset':copy.deepcopy(settings),'model':model,
            'training':training,'evaluation':{}}

def write_immutable(path,value):
    if path.exists():
        if json.loads(path.read_text())!=json.loads(json.dumps(value)):
            raise ValueError(f'Refusing to replace study protocol: {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2)+'\n')

def prepare(root=ROOT):
    from src.data.nsynth_cnn_study import build_study_data
    root=Path(root);assert_loss_unchanged(root)
    data=build_study_data(root)
    configs=root/'configs/nsynth_cnn'
    for condition in CONDITIONS:
        full=make_configuration(data['train_settings'],condition)
        write_immutable(configs/f'{condition}.json',full)
        for seed in SEEDS:
            single=copy.deepcopy(full);single.pop('seeds');single['seed']=seed
            write_immutable(configs/'runs'/f'{seed:02d}_{condition}.json',single)
    protocol={'study':'nsynth_cnn_v1','conditions':CONDITIONS,'seeds':SEEDS,
              'epochs':200,'latent_size':16,'training_noise':'gaussian std0.1 on clipped normalized mel for DAE only',
              'training_count':len(data['train']),'test_count':len(data['test']),
              'split':'held-out exact instruments within family; same split for all conditions',
              'generation':'64 paired standard-normal latent codes per seed; VAE only',
              'metrics':'heldout MSE, denoising vs identity, generation nearest-set distances and diversity; no novelty claim',
              'no_training_loss_changes':True,'original_presentation_untouched':True,
              'prior_models_not_reused':'Old NSynth models saw the full selection, including these test instruments.'}
    write_immutable(root/'results/nsynth_cnn_study/protocol.json',protocol)
    print(json.dumps(protocol,indent=2),flush=True)
    return protocol

def train_one(config):
    from src.experiments.runner import run_experiment
    return str(run_experiment(Path(config),checkpoint_every=5))

def run(root=ROOT,jobs=6):
    root=Path(root)
    prepare(root)
    tasks=[(condition,seed,root/f'configs/nsynth_cnn/runs/{seed:02d}_{condition}.json')
           for seed in SEEDS for condition in CONDITIONS]
    errors=[]
    print(f'{len(tasks)} runs, {jobs} workers, interleaved by seed and condition',flush=True)
    with ProcessPoolExecutor(max_workers=jobs,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(train_one,str(path)):(condition,seed) for condition,seed,path in tasks}
        for future in as_completed(futures):
            condition,seed=futures[future]
            try:
                future.result()
                print(f'Training complete: {condition} seed={seed}',flush=True)
                # Evaluate each finished run promptly while other workers keep training.
                from src.analysis.nsynth_cnn_evaluation import evaluate_run,report
                evaluate_run((str(root),condition,seed))
                report(root)
                print(f'Evaluation complete: {condition} seed={seed}',flush=True)
            except Exception as exc:
                traceback.print_exc()
                errors.append({'condition':condition,'seed':seed,'error':str(exc)})
                (root/'results/nsynth_cnn_study/errors.json').write_text(json.dumps(errors,indent=2))
    # Restore one full three-seed summary.csv per condition, reusing completed runs.
    if not errors:
        from src.experiments.runner import run_experiments
        run_experiments([root/f'configs/nsynth_cnn/{c}.json' for c in CONDITIONS])
    from src.analysis.nsynth_cnn_evaluation import report
    report(root)
    assert_loss_unchanged(root)
    if errors:
        raise RuntimeError(f'{len(errors)} task(s) need attention; see results/nsynth_cnn_study/errors.json')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--jobs',type=int,default=6)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    if args.prepare:prepare()
    else:run(jobs=args.jobs)

if __name__=='__main__':main()
