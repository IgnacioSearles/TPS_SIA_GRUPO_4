"""Nine-seed reconstruction replication; only 67/77/87 require new training.

python -m src.analysis.nsynth_reconstruction_replication --jobs 3
python -m src.analysis.nsynth_reconstruction_replication --report-only
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import csv
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time
import traceback

import numpy as np

from src.analysis.nsynth_cnn_protocol import write_immutable
from src.data.art_search import assert_loss_unchanged
from src.evaluation.context import write_csv

ROOT=Path(__file__).resolve().parents[2]
SEEDS=(7,17,27,37,47,57,67,77,87)
NEW_SEEDS=(67,77,87)
REUSED_SEEDS=(7,17,27,37,47,57)
ARCHITECTURES=('mlp','conv')
OUTPUT=Path('reports/nsynth_reconstruction_replication')
PROTOCOL='nsynth_reconstruction_nine_seed_v1'


def read(path):return json.loads(path.read_text())
def sha(value):return hashlib.sha256(value).hexdigest()


def run_config(config,seed):
    value=copy.deepcopy(config)
    value.pop('name',None);value.pop('output',None);value.pop('seeds',None)
    value['seed']=seed
    return value


def validate_base(config,architecture):
    expected={'type':'variational','architecture':architecture,'latent_size':16,'hidden_layers':[512,128],
              'activation':'relu','initializer':'he','loss':'mse','precision':'float32',
              'kl':{'weight':.5,'warmup_epochs':50}}
    if architecture=='conv':expected.update(image_shape=[128,64,1],conv_channels=[8,16])
    if config['model']!=expected or 'initial_parameters' in config:
        raise ValueError('Unexpected architecture, KL schedule, loss or pretrained initialization')
    training={'epochs':200,'batch_size':64,'optimizer':{'type':'adam','learning_rate':.001},'log_every':10}
    if config['training']!=training:
        raise ValueError('Only new seeds may change; training must match the original 200-epoch protocol')


def progress(directory,expected):
    if (directory/'summary.json').exists():
        if read(directory/'config.json')!=expected:raise ValueError(f'Incompatible completed run {directory}')
        with (directory/'history.csv').open() as stream:history=list(csv.DictReader(stream))
        if not history or int(float(history[-1]['epoch']))!=200:raise ValueError('Incomplete training history')
        return {'status':'training_complete','epoch':200}
    path=directory/'checkpoint.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:metadata=json.loads(str(saved['metadata']))
        if metadata['signature']['config']!=sha(json.dumps(expected,sort_keys=True).encode()):
            raise ValueError(f'Incompatible running checkpoint {directory}')
        return {'status':'training_pending','epoch':metadata['epoch']}
    return {'status':'queued','epoch':0}


def prepare(root=ROOT):
    from src.data.nsynth_cnn_study import build_study_data
    root=Path(root);assert_loss_unchanged(root)
    data=build_study_data(root)
    if len(data['train'])!=884 or len(data['test'])!=224:
        raise ValueError('The replication requires the same fixed 884/224 instrument split')
    configs=root/'configs/nsynth_reconstruction_replication';reused=[]
    for architecture in ARCHITECTURES:
        base=read(root/f'configs/nsynth_cnn/vae_{architecture}.json');validate_base(base,architecture)
        if base['dataset']!=data['train_settings']:raise ValueError('Training data differ from the original study')
        for seed in REUSED_SEEDS:
            expected=run_config(base,seed)
            directory=root/f'results/ej2_nsynth_cnn_vae_{architecture}/base__seed={seed}'
            state=progress(directory,expected)
            if seed in (37,47,57):
                assigned=root/f'configs/nsynth_warmup_replication/runs/{seed:02d}_{architecture}_warmup50.json'
                if not assigned.exists() or run_config(read(assigned),seed)!=expected:
                    raise ValueError('Reused warmup50 replication is not identically configured')
            reused.append({'architecture':architecture,'seed':seed,**state,
                           'path':str(directory.relative_to(root)),'scheduled_here':False})
        for seed in NEW_SEEDS:
            single=copy.deepcopy(base);single.pop('seeds',None);single['seed']=seed
            write_immutable(configs/f'{seed:02d}_{architecture}.json',single)
    protocol={'protocol':PROTOCOL,'all_seeds':SEEDS,'new_seeds':NEW_SEEDS,'reused_seeds':REUSED_SEEDS,
              'new_trainings':6,'reused_trainings':12,'architectures':ARCHITECTURES,'epochs':200,
              'warmup_epochs':50,'beta':.5,'split':{'train':884,'test':224,'unit':'exact instrument'},
              'reused_runs_audit':reused,'primary_metric':'MSE of every held-out mel spectrogram',
              'independence':'Nine trained seeds per architecture; 224 notes are not 224 independent trainings.',
              'purpose':'Additional reconstruction replication, without changing beta/warmup or searching hyperparameters.'}
    protocol_path=root/'results/nsynth_reconstruction_replication/protocol.json'
    if protocol_path.exists():
        previous=read(protocol_path)
        if {k:v for k,v in previous.items() if k!='reused_runs_audit'}!=json.loads(json.dumps({k:v for k,v in protocol.items() if k!='reused_runs_audit'})):
            raise ValueError('The reconstruction replication protocol changed')
    else:write_immutable(protocol_path,protocol)
    (protocol_path.parent/'latest_reused_runs.json').write_text(json.dumps(reused,indent=2)+'\n')
    return data,protocol


def train_one(config_path):
    # Use the existing run implementation, without rewriting the shared experiment summary.csv.
    from src.experiments.grid import expand_runs
    from src.experiments.runner import _obtain_summary
    config=read(Path(config_path));output=ROOT/config.pop('output');config.pop('name',None)
    runs=expand_runs(config)
    if len(runs)!=1:raise ValueError('One seed per worker is required')
    spec=runs[0]
    _obtain_summary(spec,output/spec.label,False,False,checkpoint_every=5)
    return str(output/spec.label)


def evaluate_reconstruction(root,architecture,seed,data):
    from src.experiments.runner import load_run
    root=Path(root);condition=f'vae_{architecture}'
    directory=root/f'results/ej2_nsynth_cnn_{condition}/base__seed={seed}'
    base=read(root/f'configs/nsynth_cnn/{condition}.json')
    state=progress(directory,run_config(base,seed))
    if state['status']!='training_complete':return {'architecture':architecture,'seed':seed,**state}
    weights=sha((directory/'model.npz').read_bytes())
    fingerprint=sha(json.dumps(data['manifest'],sort_keys=True).encode())
    signature={'protocol':PROTOCOL,'model_sha256':weights,'data_sha256':fingerprint}
    output=root/OUTPUT/f'{condition}/seed={seed}';output.mkdir(parents=True,exist_ok=True)
    cache=output/'summary.json'
    if cache.exists() and read(cache).get('signature')==signature:return read(cache)
    legacy=root/f'reports/nsynth_cnn_study/{condition}/seed={seed}'
    legacy_summary=read(legacy/'summary.json') if (legacy/'summary.json').exists() else {}
    rows=None;source='frozen_model_inference'
    if legacy_summary.get('model_sha256')==weights and legacy_summary.get('data_sha256')==fingerprint and (legacy/'per_note.csv').exists():
        with (legacy/'per_note.csv').open() as stream:
            selected=[r for r in csv.DictReader(stream) if r['task']=='reconstruction' and r['method']=='model']
        if sorted(r['label'] for r in selected)!=sorted(data['test'].labels):raise ValueError('Held-out cache labels differ')
        rows=[{'architecture':architecture,'seed':seed,'label':r['label'],'family':r['family'],
               'mse':float(r['mse']),'valid':str(r['valid'])=='True'} for r in selected]
        source='reused_existing_per_note_csv'
    if rows is None:
        model,training=load_run(directory)
        if training.labels!=data['train'].labels:raise ValueError('Unexpected model training split')
        samples=data['test'].samples;values=[]
        for start in range(0,len(samples),32):values.append(model.reconstruct(samples[start:start+32].astype(model.precision)))
        decoded=np.concatenate(values);valid=np.all(np.isfinite(decoded),axis=1)
        errors=np.mean((decoded-samples)**2,axis=1)
        rows=[{'architecture':architecture,'seed':seed,'label':label,'family':family,
               'mse':float(errors[i]) if valid[i] else float('inf'),'valid':bool(valid[i])}
              for i,(label,family) in enumerate(zip(data['test'].labels,data['test'].groups))]
    if sha((directory/'model.npz').read_bytes())!=weights:raise ValueError('Model changed during reconstruction evaluation')
    write_csv(output/'per_note.csv',rows)
    families=[]
    for family in sorted({r['family'] for r in rows}):
        selected=[r for r in rows if r['family']==family]
        families.append({'architecture':architecture,'seed':seed,'family':family,'notes':len(selected),
                         'mse':float(np.mean([r['mse'] for r in selected])),'invalid':sum(not r['valid'] for r in selected)})
    write_csv(output/'per_family.csv',families)
    result={'architecture':architecture,'seed':seed,'status':'evaluated','epoch':200,'signature':signature,
            'mse':float(np.mean([r['mse'] for r in rows])),'notes':len(rows),'families':len(families),
            'invalid':sum(not r['valid'] for r in rows),'source':source}
    cache.write_text(json.dumps(result,indent=2)+'\n')
    return result


def paired(rows):
    values=[]
    for seed in SEEDS:
        pair={r['architecture']:r for r in rows if r['seed']==seed and r['status']=='evaluated'}
        if len(pair)!=2:continue
        if pair['mlp']['signature']['data_sha256']!=pair['conv']['signature']['data_sha256']:
            raise ValueError('Dense and CNN evaluations use different held-out data')
        values.append({'seed':seed,'mlp_mse':pair['mlp']['mse'],'conv_mse':pair['conv']['mse'],
                       'difference_conv_minus_mlp':pair['conv']['mse']-pair['mlp']['mse']})
    return values


def report(root=ROOT,data=None):
    from src.data.nsynth_cnn_study import build_study_data
    root=Path(root);assert_loss_unchanged(root)
    if data is None:data=build_study_data(root)
    output=root/OUTPUT;output.mkdir(parents=True,exist_ok=True)
    rows=[evaluate_reconstruction(root,a,s,data) for s in SEEDS for a in ARCHITECTURES]
    pairs=paired(rows);evaluated=[r for r in rows if r['status']=='evaluated']
    per_note=[];per_family=[]
    for row in evaluated:
        folder=output/f'vae_{row["architecture"]}/seed={row["seed"]}'
        for name,collection in [('per_note.csv',per_note),('per_family.csv',per_family)]:
            with (folder/name).open() as stream:collection.extend(csv.DictReader(stream))
    if per_note:write_csv(output/'per_note.csv',per_note);write_csv(output/'per_family.csv',per_family)
    write_csv(output/'runs.csv',rows)
    if pairs:write_csv(output/'paired_seeds.csv',pairs)
    result={'protocol':PROTOCOL,'status':'complete' if len(evaluated)==18 else 'incomplete','expected_models':18,
            'evaluated_models':len(evaluated),'paired_seeds':len(pairs),'expected_paired_seeds':9,'runs':rows,'pairs':pairs}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Reconstrucción VAE NSynth: nueve semillas',f'Estado: {result["status"]}, {len(evaluated)}/18 modelos y {len(pairs)}/9 pares evaluados.',
           '','Se reutilizan las semillas 7,17,27 y las réplicas warmup50 de37,47,57. Solo se entrenan seis modelos nuevos: '
           '67,77,87 para denso y CNN. Todos usan 200 épocas, beta0.5, warmup50 y el mismo split de884/224 por instrumento.',
           '','| Semilla | MSE denso | MSE CNN | Δ CNN menos denso |','|---|---:|---:|---:|']
    for row in pairs:lines.append(f'| {row["seed"]} | {row["mlp_mse"]:.6f} | {row["conv_mse"]:.6f} | {row["difference_conv_minus_mlp"]:.6f} |')
    pending=[r for r in rows if r['status']!='evaluated']
    if pairs:
        differences=np.array([r['difference_conv_minus_mlp'] for r in pairs])
        lines += ['',f'Entre los {len(pairs)} pares disponibles, Δ media={differences.mean():.6f}; '
                  f'CNN con menor MSE en {sum(differences<0)}/{len(pairs)} semillas. '
                  'Estos recuentos se actualizan solo cuando ambas arquitecturas de una semilla están evaluadas.']
    if pending:
        lines += ['','## Pendientes','']+[f'- {r["architecture"]}, semilla {r["seed"]}: {r["status"]}, época {r["epoch"]}/200.' for r in pending]
    lines += ['','La MSE global pondera las224notas por igual; el CSV por familia conserva los desbalances. '
              'Hay nueve inicializaciones independientes por arquitectura cuando termine el plan, no9×224 entrenamientos. '
              'La ampliación reusa el mismo conjunto de instrumentos reservados y evalúa estabilidad frente a semillas, no generalización externa.',
              'No se declara significación estadística ni se elige ganador con pares faltantes. '
              'La evaluación trabaja en mel: no cambia con la inicialización de fase al escuchar WAV. '
              'Una mejor MSE no certifica calidad auditiva ni novedad. Se conservan fallos e inválidos. '
              'La pérdida VAE y el ejecutor de entrenamiento no se modificaron; no se reentrenan réplicas warmup50 existentes.']
    (output/'informe.md').write_text('\n'.join(lines)+'\n')
    if pairs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for row in pairs:axes[0].plot([0,1],[row['mlp_mse'],row['conv_mse']],marker='o',label=str(row['seed']))
        axes[0].set_xticks([0,1],['Denso','CNN']);axes[0].set_ylabel('MSE held-out por semilla');axes[0].set_ylim(bottom=0)
        axes[0].legend(title='Semilla',ncol=3,fontsize=8)
        axes[1].bar([str(r['seed']) for r in pairs],[r['difference_conv_minus_mlp'] for r in pairs],color='#176a98')
        axes[1].axhline(0,color='#555555');axes[1].set_ylabel('Δ MSE: CNN menos denso');axes[1].set_xlabel('Semilla')
        fig.suptitle(f'Reconstrucción NSynth: {len(pairs)}/9 pares disponibles,200 épocas, warmup50')
        fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    return result


def run(root=ROOT,jobs=3):
    root=Path(root);data,_=prepare(root);errors=[]
    tasks=[(a,s,root/f'configs/nsynth_reconstruction_replication/{s:02d}_{a}.json') for s in NEW_SEEDS for a in ARCHITECTURES]
    print(f'Reconstruction replication:6 new trainings,{jobs} workers; warmup50 existing seeds not rescheduled.',flush=True)
    with ProcessPoolExecutor(max_workers=jobs,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(train_one,str(path)):(a,s) for a,s,path in tasks}
        for future in as_completed(futures):
            architecture,seed=futures[future]
            try:
                future.result();report(root,data)
                print(f'Reconstruction replication evaluated: {architecture},seed={seed}',flush=True)
            except Exception as exc:
                traceback.print_exc();errors.append({'architecture':architecture,'seed':seed,'error':str(exc)})
                (root/'results/nsynth_reconstruction_replication/errors.json').write_text(json.dumps(errors,indent=2))
    result=report(root,data)
    if errors:raise RuntimeError(f'{len(errors)} reconstruction errors; see errors.json')
    while result['status']!='complete':
        print(f'Waiting for reused warmup50 replicas: {result["evaluated_models"]}/18 models evaluated.',flush=True)
        time.sleep(30);result=report(root,data)
    assert_loss_unchanged(root)
    print('Nine-seed reconstruction comparison complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=3);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if args.jobs<1:parser.error('jobs must be positive')
    if args.prepare:print(json.dumps(prepare()[1],indent=2))
    elif args.report_only:print(json.dumps({k:v for k,v in report().items() if k not in ('runs','pairs')},indent=2))
    else:run(jobs=args.jobs)
