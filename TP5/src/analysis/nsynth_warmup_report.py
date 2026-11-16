"""Paired NSynth VAE warmup 0 vs 50 report, without retraining or audio inversion.

python -m src.analysis.nsynth_warmup_report
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.art_search import assert_loss_unchanged
from src.evaluation.context import write_csv
from src.experiments.runner import load_run

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path('reports/nsynth_warmup')
ORIGINAL_SEEDS=(7,17,27)
EXTENSION_SEEDS=(37,47,57)
SEEDS=ORIGINAL_SEEDS+EXTENSION_SEEDS
EXPECTED_MODELS=4*len(SEEDS)
SEED_COLORS=('#176a98','#d4602a','#4d8b5a','#8b5da8','#ad8335','#4e7e86')
ARCHITECTURES=('mlp','conv')
WARMUPS=(0,50)
ACTIVE_THRESHOLD=1e-2
PROTOCOL='nsynth_warmup_v1'


def read(path):
    return json.loads(path.read_text())


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def condition_name(architecture,warmup):
    return f'vae_{architecture}'+('_no_warmup' if warmup==0 else '')


def normalized_config(config):
    """Only the KL ramp may differ in a matched pair of actual run configs."""
    result=json.loads(json.dumps(config))
    result['model']['kl'].pop('warmup_epochs')
    return result


def validate_matched_pair(config_zero,config_fifty):
    for config,warmup in ((config_zero,0),(config_fifty,50)):
        if config['model']['type']!='variational' or config['model']['kl']!={'weight':.5,'warmup_epochs':warmup}:
            raise ValueError('Unexpected model or KL schedule')
        if config['training']['epochs']!=200:
            raise ValueError('Warmup comparison requires 200 epochs in both arms')
        if 'initial_parameters' in config:
            raise ValueError('Warmup ablation must start from scratch')
    if normalized_config(config_zero)!=normalized_config(config_fifty):
        raise ValueError('Unmatched warmup comparison: data, architecture, seed or training settings differ')


def posterior_statistics(model,samples,batch_size=32):
    """Train-posterior statistics, no latent sampling and no decoder calls."""
    means=[];log_variances=[]
    for start in range(0,len(samples),batch_size):
        mu,log_variance=model.encode_distribution(samples[start:start+batch_size].astype(model.precision))
        means.append(np.asarray(mu,dtype=np.float64).copy())
        log_variances.append(np.asarray(log_variance,dtype=np.float64).copy())
    mu=np.concatenate(means);log_variance=np.concatenate(log_variances)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(log_variance)):
        raise ValueError('Non-finite posterior statistics')
    variance_mu=np.var(mu,axis=0)
    per_dimension_kl=np.mean(.5*(np.exp(log_variance)+mu**2-1-log_variance),axis=0)
    if not np.all(np.isfinite(per_dimension_kl)):
        raise ValueError('Non-finite posterior KL')
    return {'train_samples':len(samples),'latent_size':mu.shape[1],
            'active_threshold':ACTIVE_THRESHOLD,'active_dimensions':int(np.sum(variance_mu>ACTIVE_THRESHOLD)),
            'mean_posterior_kl':float(np.sum(per_dimension_kl)),
            'variance_mu_per_dimension':variance_mu.tolist(),
            'mean_mu_per_dimension':np.mean(mu,axis=0).tolist(),
            'mean_log_variance_per_dimension':np.mean(log_variance,axis=0).tolist(),
            'mean_kl_per_dimension':per_dimension_kl.tolist()}


def latent_statistics(root,directory,condition,seed,weights_sha):
    root=Path(root)
    train_config=read(directory/'config.json')
    # Dataset config fixes selection and exact label order; hashes also bind source selection.
    manifest_path=root/'results/nsynth_cnn_study/data/manifest.json'
    manifest_hash=hash_bytes(manifest_path.read_bytes())
    signature={'protocol':PROTOCOL,'weights_sha256':weights_sha,'manifest_sha256':manifest_hash,
               'dataset_config_sha256':hash_bytes(json.dumps(train_config['dataset'],sort_keys=True).encode()),
               'active_threshold':ACTIVE_THRESHOLD}
    cache=root/OUTPUT/f'latent/{condition}/seed={seed}.json'
    if cache.exists():
        value=read(cache)
        if value.get('signature')==signature:return value['statistics']
    model,dataset=load_run(directory)
    manifest=read(manifest_path)
    if dataset.labels!=manifest['train_labels'] or set(dataset.labels).intersection(manifest['test_labels']):
        raise ValueError('Latent diagnostics require the original training split, without held-out examples')
    statistics=posterior_statistics(model,dataset.samples)
    if hash_bytes((directory/'model.npz').read_bytes())!=weights_sha:
        raise ValueError('Weights changed while evaluating posterior')
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps({'signature':signature,'statistics':statistics},indent=2)+'\n')
    return statistics


def reconstruction_mse(evaluation_folder):
    path=evaluation_folder/'per_note.csv'
    if not path.exists():return None
    with path.open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['task']=='reconstruction' and r['method']=='model']
    if not rows:return None
    if len({r['label'] for r in rows})!=len(rows):raise ValueError('Duplicate held-out reconstruction rows')
    return float(np.mean([float(r['mse']) for r in rows]))


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['architecture'],row['warmup']].append(row)
    fields=('heldout_mse','final_kl','final_reconstruction','final_loss','mean_posterior_kl','active_dimensions',
            'generation_train_mse','generation_test_mse','generation_pairwise_mse')
    aggregates=[]
    for (architecture,warmup),group in sorted(groups.items()):
        aggregate={'architecture':architecture,'warmup':warmup,'trained_models':len(group),
                   'evaluated_models':sum(r['evaluation_ready'] for r in group)}
        for field in fields:
            values=[r[field] for r in group if r.get(field) is not None]
            aggregate[f'{field}_mean']=float(np.mean(values)) if values else None
            aggregate[f'{field}_seeds']=len(values)
        aggregates.append(aggregate)
    return aggregates


def paired_differences(rows,configs):
    indexed={(r['architecture'],r['warmup'],r['seed']):r for r in rows}
    differences=[]
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            zero=indexed.get((architecture,0,seed));fifty=indexed.get((architecture,50,seed))
            if zero is None or fifty is None:continue
            validate_matched_pair(configs[architecture,0,seed],configs[architecture,50,seed])
            if zero['data_sha256'] is not None and fifty['data_sha256'] is not None and zero['data_sha256']!=fifty['data_sha256']:
                raise ValueError('Unmatched evaluation split')
            if zero['generation_codes_sha256'] is not None and fifty['generation_codes_sha256'] is not None and zero['generation_codes_sha256']!=fifty['generation_codes_sha256']:
                raise ValueError('Unmatched generation codes')
            difference={'architecture':architecture,'seed':seed,'definition':'warmup0 minus warmup50'}
            for field in ('heldout_mse','final_kl','final_reconstruction','final_loss','mean_posterior_kl',
                          'active_dimensions','generation_train_mse','generation_test_mse','generation_pairwise_mse'):
                difference[field]=zero[field]-fifty[field] if zero.get(field) is not None and fifty.get(field) is not None else None
            differences.append(difference)
    return differences


def paired_descriptions(paired):
    """Describe seed-level differences; no p-values or automatic winner selection."""
    groups={'original_three':ORIGINAL_SEEDS,'extension_three':EXTENSION_SEEDS,'all_six':SEEDS}
    descriptions=[]
    for scope,seeds in groups.items():
        for architecture in ARCHITECTURES:
            selected=[r for r in paired if r['architecture']==architecture and r['seed'] in seeds]
            for field in ('heldout_mse','mean_posterior_kl','active_dimensions','generation_train_mse',
                          'generation_test_mse','generation_pairwise_mse'):
                values=np.asarray([r[field] for r in selected if r.get(field) is not None],dtype=float)
                if not len(values):continue
                descriptions.append({'scope':scope,'architecture':architecture,'metric':field,
                    'expected_pairs':len(seeds),'available_pairs':len(values),'complete_scope':len(values)==len(seeds),
                    'definition':'warmup0 minus warmup50','mean_difference':float(np.mean(values)),
                    'median_difference':float(np.median(values)),'minimum_difference':float(np.min(values)),
                    'maximum_difference':float(np.max(values)),'negative_differences':int(np.sum(values<0)),
                    'zero_differences':int(np.sum(values==0)),'positive_differences':int(np.sum(values>0))})
    return descriptions


def plot(rows,path):
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields=[('heldout_mse','MSE en instrumentos reservados'),('mean_posterior_kl','KL posterior media en train'),
            ('active_dimensions','Dimensiones activas, umbral 0,01'),('generation_pairwise_mse','Diversidad MSE de generación')]
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for row_index,architecture in enumerate(ARCHITECTURES):
        for ax,(field,title) in zip(axes[row_index],fields):
            for seed,color in zip(SEEDS,SEED_COLORS):
                positions=[];values=[]
                for warmup in WARMUPS:
                    row=next((r for r in rows if r['architecture']==architecture and r['seed']==seed and r.get(field) is not None and r['warmup']==warmup),None)
                    if row is not None:positions.append(warmup);values.append(row[field])
                if values:ax.plot(positions,values,marker='o',color=color,label=f'Semilla {seed}')
            ax.set_title(f'{architecture.upper()}: {title}',fontsize=10)
            ax.set_xticks([0,50]);ax.set_xlabel('Épocas de warmup');ax.set_ylim(bottom=0)
            if field=='active_dimensions':ax.set_ylim(0,16.5)
            handles,_=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=7,ncol=2)
    present=sorted({r['seed'] for r in rows})
    fig.suptitle(f'NSynth: beta final 0,5 y 200 épocas. Semillas disponibles {present}; plan: 6 por condición.')
    fig.savefig(path,dpi=140);plt.close(fig)


def plot_histories(root,rows,path):
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for row_index,architecture in enumerate(ARCHITECTURES):
        for ax,(field,title) in zip(axes[row_index],[('reconstruction','Reconstrucción por muestra'),
                                                    ('kl','KL por muestra'),('loss','Recon + 0,5 KL, desde época 50')]):
            labeled=set()
            for row in rows:
                if row['architecture']!=architecture:continue
                directory=root/f'results/ej2_nsynth_cnn_{row["condition"]}/base__seed={row["seed"]}'
                with (directory/'history.csv').open() as stream:history=list(csv.DictReader(stream))
                points=[r for r in history if field in r and r[field] and (field!='loss' or float(r['epoch'])>=50)]
                if not points:continue
                warmup=row['warmup'];label=f'Warmup {warmup}' if warmup not in labeled else None;labeled.add(warmup)
                ax.plot([float(r['epoch']) for r in points],[float(r[field]) for r in points],
                        color='#176a98' if warmup==50 else '#d4602a',
                        linestyle={7:'-',17:'--',27:':',37:'-',47:'--',57:':'}[row['seed']],
                        alpha=.8 if row['seed'] in ORIGINAL_SEEDS else .5,label=label)
            ax.set_title(f'{architecture.upper()}: {title}',fontsize=11)
            ax.set_xlabel('Época');ax.set_ylim(bottom=0)
            if field!='loss':ax.axvline(50,color='#999999',linestyle=':',alpha=.5)
            handles,_=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=8)
    fig.suptitle('Reconstrucción y KL separadas. Antes de 50 los pesos KL difieren. Extensión 37/47/57: líneas más claras.')
    fig.savefig(path,dpi=140);plt.close(fig)


def report(root=ROOT,compute_latents=True):
    root=Path(root);assert_loss_unchanged(root)
    output=root/OUTPUT;output.mkdir(parents=True,exist_ok=True)
    rows=[];missing=[];configs={};dimensions=[]
    for architecture in ARCHITECTURES:
        for warmup in WARMUPS:
            condition=condition_name(architecture,warmup)
            for seed in SEEDS:
                directory=root/f'results/ej2_nsynth_cnn_{condition}/base__seed={seed}'
                if not (directory/'summary.json').exists():
                    missing.append({'condition':condition,'seed':seed,'status':'training_pending'});continue
                config=read(directory/'config.json');configs[architecture,warmup,seed]=config
                if (config['seed']!=seed or config['model']['kl']!={'weight':.5,'warmup_epochs':warmup}
                        or config['model']['type']!='variational'
                        or config['model'].get('architecture','mlp')!=architecture
                        or config['training']['epochs']!=200 or 'initial_parameters' in config):
                    raise ValueError('Unexpected completed warmup configuration')
                with (directory/'history.csv').open() as stream:history=list(csv.DictReader(stream))
                if not history or int(float(history[-1]['epoch']))!=200:
                    raise ValueError('Completed model does not have 200 epochs in its recorded history')
                training=read(directory/'training.json')
                weights_sha=hash_bytes((directory/'model.npz').read_bytes())
                evaluation_folder=root/f'reports/nsynth_cnn_study/{condition}/seed={seed}'
                evaluation=read(evaluation_folder/'summary.json') if (evaluation_folder/'summary.json').exists() else None
                if evaluation is not None and evaluation.get('model_sha256')!=weights_sha:
                    raise ValueError('Stale audio/reconstruction evaluation for changed model weights')
                latent=latent_statistics(root,directory,condition,seed,weights_sha) if compute_latents else None
                row={'architecture':architecture,'warmup':warmup,'condition':condition,'seed':seed,
                     'evaluation_ready':evaluation is not None,'model_sha256':weights_sha,
                     'heldout_mse':reconstruction_mse(evaluation_folder) if evaluation else None,
                     'final_kl':training.get('final_kl'),'final_reconstruction':training.get('final_reconstruction'),
                     'final_loss':training.get('final_loss'),'active_dimensions':latent['active_dimensions'] if latent else None,
                     'mean_posterior_kl':latent['mean_posterior_kl'] if latent else None,
                     'data_sha256':evaluation.get('data_sha256') if evaluation else None,
                     'generation_codes_sha256':evaluation.get('generation_codes_sha256') if evaluation else None,
                     **{key:evaluation.get(key) if evaluation else None for key in ('generation_train_mse','generation_test_mse',
                          'generation_pairwise_mse','generation_distinct_train_neighbors','generation_distinct_test_neighbors')}}
                rows.append(row)
                if latent:
                    for dimension in range(latent['latent_size']):
                        dimensions.append({'condition':condition,'seed':seed,'architecture':architecture,'warmup':warmup,
                                           'dimension':dimension,'variance_mu':latent['variance_mu_per_dimension'][dimension],
                                           'active':latent['variance_mu_per_dimension'][dimension]>ACTIVE_THRESHOLD,
                                           'mean_kl':latent['mean_kl_per_dimension'][dimension]})
    paired=paired_differences(rows,configs)
    aggregates=summarize(rows)
    original_aggregates=summarize([r for r in rows if r['seed'] in ORIGINAL_SEEDS])
    extension_aggregates=summarize([r for r in rows if r['seed'] in EXTENSION_SEEDS])
    descriptions=paired_descriptions(paired)
    if rows:write_csv(output/'per_seed.csv',rows);write_csv(output/'comparison.csv',aggregates)
    if original_aggregates:write_csv(output/'comparison_original_three.csv',original_aggregates)
    if extension_aggregates:write_csv(output/'comparison_extension_three.csv',extension_aggregates)
    if descriptions:write_csv(output/'paired_descriptions.csv',descriptions)
    if dimensions:write_csv(output/'latent_dimensions.csv',dimensions)
    if paired:write_csv(output/'paired_differences.csv',paired)
    ready=sum(r['evaluation_ready'] for r in rows)
    result={'protocol':PROTOCOL,'status':'complete' if len(rows)==EXPECTED_MODELS and ready==EXPECTED_MODELS else 'incomplete',
            'trained_models':len(rows),'evaluated_models':ready,'expected_models':EXPECTED_MODELS,'missing':missing,
            'report_scope':'six_seed_extension','original_seeds':ORIGINAL_SEEDS,'extension_seeds':EXTENSION_SEEDS,
            'original_trained_models':sum(r['seed'] in ORIGINAL_SEEDS for r in rows),
            'extension_trained_models':sum(r['seed'] in EXTENSION_SEEDS for r in rows),
            'posterior_statistics_computed':compute_latents,'rows':rows,'aggregates':aggregates,
            'original_seed_aggregates':original_aggregates,'extension_seed_aggregates':extension_aggregates,
            'paired_differences':paired,'paired_descriptions':descriptions}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    def value(v):return 'pendiente' if v is None else f'{v:.6f}' if isinstance(v,float) else str(v)
    lines=['# NSynth: ablación de warmup KL',f'Estado: {result["status"]}, {len(rows)}/{EXPECTED_MODELS} modelos entrenados y {ready}/{EXPECTED_MODELS} evaluados.',
           '','Dos arquitecturas, warmup 0 o 50, 200 épocas desde cero y beta final 0,5. '
           'La tanda original usa semillas 7, 17, 27 y 12 modelos; su ampliación agrega semillas 37, 47, 57 y 12 modelos. '
           'Los seis controles warmup 50 originales se reutilizan del estudio NSynth principal. El plan completo tiene seis semillas por condición.',
           '',f'Tanda original: {result["original_trained_models"]}/12 entrenados. Ampliación: {result["extension_trained_models"]}/12 entrenados. '
           'Mientras falten semillas nuevas, el agregado disponible sigue reflejando las semillas terminadas; no se presenta como un resultado de seis semillas.',
           '','| Arquitectura | Warmup | Semilla | MSE test | KL final train | Recon final train | Loss final train | KL posterior media | Dims activas |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        fields=('architecture','warmup','seed','heldout_mse','final_kl','final_reconstruction','final_loss','mean_posterior_kl','active_dimensions')
        lines.append('| '+' | '.join(value(row[field]) for field in fields)+' |')
    lines += ['','## Diferencias pareadas de MSE held-out','',
              'Δ = MSE con warmup 0 menos MSE con warmup 50. Un valor negativo favorece warmup 0 en esa semilla. '
              'Se muestran por separado las semillas originales, las nuevas y todas las disponibles. No se calcula significación estadística.',
              '','| Grupo | Arquitectura | Pares disponibles/esperados | Δ media | Δ mínima | Δ máxima | Δ negativas | Δ positivas |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    scope_labels={'original_three':'Original: 7, 17, 27','extension_three':'Ampliación: 37, 47, 57','all_six':'Plan completo: seis semillas'}
    for row in descriptions:
        if row['metric']!='heldout_mse':continue
        lines.append(f'| {scope_labels[row["scope"]]} | {row["architecture"]} | {row["available_pairs"]}/{row["expected_pairs"]} | '
                     f'{row["mean_difference"]:.6f} | {row["minimum_difference"]:.6f} | {row["maximum_difference"]:.6f} | '
                     f'{row["negative_differences"]} | {row["positive_differences"]} |')
    if missing:
        lines += ['','## Modelos pendientes','']
        lines += [f'- {item["condition"]}, semilla {item["seed"]}: entrenamiento pendiente.' for item in missing]
    lines += ['','## Generación libre, mismos códigos normales estándar','',
              '| Arquitectura | Warmup | Semilla | MSE vecino train | MSE vecino test | MSE entre pares | Vecinos train distintos /64 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        fields=('architecture','warmup','seed','generation_train_mse','generation_test_mse','generation_pairwise_mse','generation_distinct_train_neighbors')
        lines.append('| '+' | '.join(value(row[field]) for field in fields)+' |')
    lines += ['','Cada par de la misma arquitectura y semilla debe compartir configuración completa salvo warmup. '
              'La misma semilla fija inicialización, orden de minibatches y secuencia de ruido de reparametrización; después del primer gradiente diferente los pesos pueden divergir. '
              'Los datos se separan por instrumento y las métricas test no intervienen en entrenamiento.',
              'Dimensión activa: varianza poblacional de μ_j(x) entre notas de entrenamiento estrictamente mayor que0.01. '
              'KL posterior media evalúa analíticamente q(z|x) contra N(0,I) con pesos finales, sin muestrear ni decodificar. '
              'Estos indicadores pueden revelar uso débil del latente, pero no establecen por sí solos un colapso posterior ni una relación musical correcta.',
              'Los términos finales de entrenamiento proceden de la última época registrada y usan reconstrucción sumada sobre características por muestra. '
              'La MSE held-out promedia características. Tienen escalas y procedimientos distintos y no deben compararse como si fueran la misma métrica.',
              'Durante las primeras50 épocas, beta es0.5 desde el inicio sin warmup y crece como0.5×min(1,época/50) con warmup. '
              'Por eso no se comparan las pérdidas totales de esas primeras épocas como si usaran el mismo objetivo ponderado. '
              'Las curvas muestran reconstrucción y KL por separado, y pérdida total únicamente desde la época50. El total final corresponde al mismo beta0.5.',
              'La generación reutiliza la evaluación cacheada: no repite WAV ni Griffin-Lim. Menor distancia al vecino y mayor diversidad no certifican calidad ni novedad. '
              'Train y test contienen distinto número de referencias, por lo que sus distancias mínimas no prueban generalización.',
              'Las métricas de este informe operan sobre espectrogramas mel y distribuciones latentes; no dependen del método de fase usado para exportar WAV. '
              'Una comparación de escucha debe usar el mismo método de fase en todos los modelos y registrarlo por separado. '
              'Los WAV históricos no se reinterpretan como PGHI ni se regeneran en este reporte.',
              'Los CSV conservan todas las semillas y las diferencias warmup0 menos warmup50. Un resultado peor o inconcluso sigue siendo válido. '
              'Mientras falten pares no se declara ganador. El plan llega a seis entrenamientos independientes por condición; '
              'las224 notas de prueba de cada modelo no son224 entrenamientos independientes. '
              'La ampliación usa el mismo conjunto de prueba que la tanda original: evalúa estabilidad frente a inicialización, '
              'pero no equivale a una validación externa nueva ni garantiza generalización a otros instrumentos o datasets.']
    (output/'informe.md').write_text('\n'.join(lines)+'\n')
    plot(rows,output/'comparison.png')
    plot_histories(root,rows,output/'training_curves.png')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-latent',action='store_true',help='Report cached evaluation without computing posterior statistics')
    args=parser.parse_args()
    result=report(compute_latents=not args.skip_latent)
    print(json.dumps({key:result[key] for key in ('status','trained_models','evaluated_models','expected_models')},indent=2))
