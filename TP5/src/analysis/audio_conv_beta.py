"""Short, paired beta continuations from frozen omitted-note CNN checkpoints.

No new loss, supervision or final-reference access. New Adam phase, not resume.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.analysis.audio_convolution import summarize_rows
from src.data.art_search import assert_loss_unchanged
from src.evaluation.context import write_csv

ROOT=Path(__file__).resolve().parents[2]
SEEDS=(7,17,27)
ARMS={'beta005':.05,'beta050':.5,'beta100':1.}
PROTOCOL='audio_conv_beta_pilot_v1'

def read(path):
    return json.loads(path.read_text())

def freeze_checkpoint(source, destination, seed):
    """Read an atomic checkpoint once; never read a changing file twice."""
    with np.load(source,allow_pickle=False) as saved:
        metadata=json.loads(str(saved['metadata']))
        arrays={key:saved[key].copy() for key in saved.files if key.startswith('parameter_')}
    if not 100 <= metadata['epoch'] <= 200:
        raise ValueError(f'Unexpected parent epoch for seed {seed}: {metadata["epoch"]}')
    if not arrays:
        raise ValueError('Empty checkpoint')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise ValueError('Refusing to overwrite frozen weights')
    np.savez_compressed(destination,**arrays)
    return {'seed':seed,'parent':str(source.relative_to(ROOT)),'parent_epoch':metadata['epoch'],
            'weights':str(destination.relative_to(ROOT)),
            'weights_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'parent_elapsed_seconds':metadata['elapsed']}

def prepare(root=ROOT):
    root=Path(root)
    assert_loss_unchanged(root)
    folder=root/'results/audio_conv_beta_pilot';folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'protocol.json'
    if manifest.exists():
        value=read(manifest)
        for row in value['parents']:
            if hashlib.sha256((root/row['weights']).read_bytes()).hexdigest()!=row['weights_sha256']:
                raise ValueError('Frozen weights changed')
        return value
    base=read(root/'configs/audio_conv/ej2_audio_conv_conv_omitted.json')
    parents=[]
    for seed in SEEDS:
        source=root/f'results/ej2_audio_conv_conv_omitted/base__seed={seed}/checkpoint.npz'
        parents.append(freeze_checkpoint(source,folder/f'parents/seed={seed}/model.npz',seed))
    config_dir=root/'configs/audio_conv_beta_pilot';config_dir.mkdir(exist_ok=True)
    for name,beta in ARMS.items():
        config=copy.deepcopy(base)
        config['model']['kl']={'weight':beta,'warmup_epochs':0}
        config['training']['epochs']=20
        config['training']['optimizer']['learning_rate']=.0003
        config['training']['log_every']=5
        config['initial_parameters']='results/audio_conv_beta_pilot/parents/seed={seed}/model.npz'
        config['name']=f'ej2_audio_conv_beta_pilot_{name}'
        config['output']=f'results/ej2_audio_conv_beta_pilot_{name}'
        (config_dir/f'{name}.json').write_text(json.dumps(config,indent=2)+'\n')
    value={'protocol':PROTOCOL,'seeds':SEEDS,'arms':ARMS,'parents':parents,
           'epochs_per_arm':20,'learning_rate':.0003,'optimizer_reset':True,
           'rng_reset_identical_across_arms':True,'same_training_data_across_arms':True,
           'targets_validation':[51,57,63],'reserved_not_used':[55,61,67],
           'interpretation':'Exploratory continuation from the same frozen weights per seed. '
                            'Parent epochs can differ across seeds because snapshots come from a running study; '
                            'all beta arms within a seed use exactly the same parent. '
                            'Not equivalent to training each beta from scratch, nor to exact Adam resume. '
                            'Compare with beta0.5 fork, not main epoch200 run. No automatic final-test selection.'}
    manifest.write_text(json.dumps(value,indent=2)+'\n')
    return value

def report(root=ROOT):
    root=Path(root);assert_loss_unchanged(root)
    protocol=read(root/'results/audio_conv_beta_pilot/protocol.json')
    output=root/'reports/audio_conv_beta_pilot';output.mkdir(parents=True,exist_ok=True)
    groups=[];seeds=[];missing=[];reliable=True
    for name,beta in ARMS.items():
        combined=[]
        for seed in SEEDS:
            run=root/f'results/ej2_audio_conv_beta_pilot_{name}/base__seed={seed}'
            if not (run/'summary.json').exists():
                missing.append(str(run.relative_to(root)));continue
            config=read(run/'config.json')
            if config['model']['kl']!={'weight':beta,'warmup_epochs':0}:
                raise ValueError('Unexpected beta')
            training=read(run/'training.json')
            parent=next(r for r in protocol['parents'] if r['seed']==seed)
            if training['initial_parameters']['sha256']!=parent['weights_sha256']:
                raise ValueError('Unmatched parent')
            manifest=read(run/'pitch_manifest.json')
            forbidden=protocol['targets_validation']+protocol['reserved_not_used']
            if any(n['split']=='training' and any(abs(n['pitch']-p)<=.75+1e-9 for p in forbidden) for n in manifest['notes']):
                raise ValueError('Training contamination')
            reliable &= read(run/'pitch_summary.json')['pitch_calibration_reliable']
            with (run/'pitch_interpolation.csv').open() as stream:
                rows=[r for r in csv.DictReader(stream) if r['method']=='vae_interpolation' and float(r['fraction'])==.5]
            if len(rows)!=9:
                raise ValueError('Missing midpoints')
            seeds.append({'arm':name,'beta':beta,'seed':seed,**summarize_rows(rows)})
            combined.extend(rows)
        if combined:
            groups.append({'arm':name,'beta':beta,**summarize_rows(combined)})
    if groups:
        write_csv(output/'comparison.csv',groups)
        write_csv(output/'per_seed.csv',seeds)
    value={'status':'incomplete' if missing else 'complete','missing':missing,
           'completed_runs':9-len(missing),'expected_runs':9,'codec_reliable':reliable if groups else None,
           'aggregates':groups,'protocol':protocol}
    (output/'summary.json').write_text(json.dumps(value,indent=2)+'\n')
    lines=['# Piloto CNN de audio: peso de KL',f'Estado: {value["status"]}, {9-len(missing)}/9 corridas terminadas.',
           '','Cada semilla comparte pesos congelados entre sus tres ramas. Épocas padre por semilla: '+str({p['seed']:p['parent_epoch'] for p in protocol['parents']})+'. Tres ramas de20 épocas con Adam reiniciado y LR0.0003. '
           'El control es beta0.5 con idéntico reinicio. No comparar como entrenamiento desde cero ni como continuación exacta.',
           '','| Beta | Altura | Espectro | Ambos | Total | MSE espectral |',
           '|---|---:|---:|---:|---:|---:|']
    lines += [f'| {r["beta"]} | {r["pitch_hits"]} | {r["spectral_hits"]} | {r["joint_hits"]} | {r["trials"]} | {r["spectral_mse"]:.6f} |' for r in groups]
    lines += ['','Las variantes no son semillas independientes. Inválidos cuentan como fallos. '
              'No se buscan mejores fracciones: solo el punto medio. Se mantienen MSE+betaKL y la representación mel. '
              'Las notas55,61,67 permanecen reservadas sin evaluar. Un piloto de20 épocas puede detectar sensibilidad, '
              'pero un resultado negativo no descarta otro beta entrenado desde cero. No se modifica la presentación.']
    (output/'informe.md').write_text('\n'.join(lines)+'\n')
    return value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    value=prepare() if args.prepare else report()
    print(json.dumps({k:v for k,v in value.items() if k not in ('parents','protocol','aggregates')},indent=2))

if __name__=='__main__':main()
