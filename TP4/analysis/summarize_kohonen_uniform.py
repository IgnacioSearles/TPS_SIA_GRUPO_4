"""Comparación emparejada de vecindades: no entrena ni modifica presentation/."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .kohonen_experiments import adjusted_rand_index
from .kohonen_uniform_experiments import ROOT, VARIANTS, digest, fingerprint, task_key

NAMES = {'gaussian':'Gaussiana 2→0,5', 'uniform_matched':'Uniforme 2→0,5', 'uniform_neighbors':'Uniforme 2→1'}


def flattened(task, record, gaussian):
    excluded = record.get('excluded_metrics', {})
    train = record['train_metrics']
    primary = excluded or train
    comparison = gaussian.get('excluded_metrics') or gaussian['train_metrics']
    sizes = record.get('selection_sizes') or {k:len(v) for k,v in record['selections'].items()}
    labels = np.asarray(record['labels'])
    indices = task.get('train',list(range(50)))
    row = dict(stage=task['stage'], candidate=task.get('candidate','rect4x4'), sample=task.get('sample',-1),
        seed=task['seed'], variant=record['variant'], strength=record['strength'],
        train_qe=train['quantization_error'], train_te=train['topographic_error'],
        excluded_qe=excluded.get('quantization_error'), excluded_te=excluded.get('topographic_error'),
        qe_gap=None if not excluded else excluded['quantization_error']-train['quantization_error'],
        ari_reference=record['ari_reference'], ari_gaussian_pair=adjusted_rand_index(labels,gaussian['labels']),
        occupied_neurons=record['occupied_neurons'], occupied_regions=len(np.unique(labels[indices])),
        delta_qe_vs_gaussian=primary['quantization_error']-comparison['quantization_error'],
        delta_te_vs_gaussian=primary['topographic_error']-comparison['topographic_error'],
        mean_displacement_z=record.get('mean_displacement_z',0.), epochs=record['epochs'],updates=record['updates'])
    for topic in ('football','makeup','dogs','art'):
        row[topic+'_overlap']=record['selection_overlap'][topic]
        row[topic+'_complete']=record['selection_overlap'][topic]==1.
        row[topic+'_short']=sizes[topic]<3
    return row


def independent_units(group):
    # Las tres semillas de una submuestra se promedian antes de contar victorias.
    columns=group.select_dtypes(include='number').columns.drop(['sample','seed','strength'])
    if group.iloc[0]['stage']=='finalists':
        return group[columns]
    return group.groupby('sample')[columns].mean()


def summary_table(frame,label_lookup):
    rows=[]
    metrics=['train_qe','train_te','excluded_qe','excluded_te','qe_gap','ari_reference','ari_gaussian_pair',
             'occupied_neurons','occupied_regions','mean_displacement_z']
    for (stage,candidate,variant,strength), group in frame.groupby(['stage','candidate','variant','strength'],sort=False):
        units=independent_units(group)
        row=dict(stage=stage,candidate=candidate,variant=variant,strength=strength,fits=len(group),units=len(units))
        for metric in metrics:
            if group[metric].notna().any():
                row[metric]=group[metric].mean()
        row.update(delta_qe_vs_gaussian=units.delta_qe_vs_gaussian.mean(),
            delta_qe_p05=units.delta_qe_vs_gaussian.quantile(.05),delta_qe_p95=units.delta_qe_vs_gaussian.quantile(.95),
            qe_win_fraction_vs_gaussian=(units.delta_qe_vs_gaussian<0).mean(),
            delta_te_vs_gaussian=units.delta_te_vs_gaussian.mean())
        assignments=[]
        for entry in group.itertuples():
            assignments.append((entry.sample,label_lookup[(stage,candidate,entry.sample,entry.seed,variant,strength)]))
        if stage=='finalists':
            aris=[adjusted_rand_index(a[1],b[1]) for a,b in combinations(assignments,2)]
        else:
            aris=[adjusted_rand_index(a[1],b[1]) for a,b in combinations(assignments,2) if a[0]==b[0]]
        row['ari_between_seeds']=float(np.mean(aris)) if aris else None
        for topic in ('football','makeup','dogs','art'):
            for suffix in ('overlap','complete','short'):
                row[topic+'_'+suffix]=group[topic+'_'+suffix].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def augmentation_table(frame,gates):
    rows=[]
    for variant in ['gaussian',*VARIANTS]:
        group=frame[(frame.stage=='augmentation')&(frame.variant==variant)]
        base=group[group.strength==0].set_index(['sample','seed'])
        for strength in [.05,.15,.30]:
            condition=group[group.strength==strength].set_index(['sample','seed'])
            if len(condition)!=len(base):
                raise ValueError('Falta contraparte sin augmentation')
            delta=condition[['excluded_qe','excluded_te','art_overlap']]-base[['excluded_qe','excluded_te','art_overlap']]
            split=delta.groupby('sample').mean()
            row=dict(variant=variant,strength=strength,fits=len(condition),
                excluded_qe=condition.excluded_qe.mean(),baseline_qe=base.excluded_qe.mean(),
                relative_qe_improvement=1-condition.excluded_qe.mean()/base.excluded_qe.mean(),
                mean_delta_qe=split.excluded_qe.mean(),delta_qe_p05=split.excluded_qe.quantile(.05),
                delta_qe_p95=split.excluded_qe.quantile(.95),split_win_fraction=(split.excluded_qe<0).mean(),
                mean_te_increase=split.excluded_te.mean(),mean_art_overlap_decrease=-split.art_overlap.mean())
            row['passes_qe']=bool(row['relative_qe_improvement']>=gates['min_relative_qe_improvement'])
            row['passes_wins']=bool(row['split_win_fraction']>=gates['min_split_win_fraction'])
            row['passes_te']=bool(row['mean_te_increase']<=gates['max_te_increase'])
            row['passes_art']=bool(row['mean_art_overlap_decrease']<=gates['max_art_overlap_decrease'])
            row['eligible']=all(row[k] for k in ['passes_qe','passes_wins','passes_te','passes_art'])
            rows.append(row)
    return pd.DataFrame(rows)


def figures(output,frame,references):
    variants=['gaussian',*VARIANTS]
    colors=['#536678','#277b96','#b77729']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    full=frame[(frame.stage=='finalists')&(frame.candidate=='rect4x4')]
    for ax,metric,title in [(axes[0,0],'train_qe','4×4: error de cuantización'),
                             (axes[1,0],'train_te','4×4: error topográfico')]:
        ax.boxplot([full.loc[full.variant==v,metric] for v in variants],tick_labels=[NAMES[v] for v in variants])
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    sub=frame[frame.stage=='subsamples']
    axes[0,1].boxplot([sub[sub.variant==v].groupby('sample').delta_qe_vs_gaussian.mean() for v in VARIANTS],
                       tick_labels=[NAMES[v] for v in VARIANTS])
    axes[0,1].axhline(0,color='#b24c42',linestyle='--')
    axes[0,1].set(title='100 submuestras: QE excluidos − gaussiana',ylabel='Diferencia media de tres semillas')
    aug=frame[frame.stage=='augmentation']
    for v,color in zip(variants,colors):
        means=aug[aug.variant==v].groupby('strength').excluded_qe.mean()
        axes[1,1].plot(means.index*100,means.values,'o-',label=NAMES[v],color=color)
    axes[1,1].set(title='Augmentation: error en perfiles excluidos',xlabel='Desplazamiento máximo hacia vecino (%)',ylabel='QE medio')
    axes[1,1].legend(fontsize=9);axes[1,1].grid(alpha=.2)
    fig.suptitle('Vecindad uniforme frente a gaussiana — mismas semillas, datos y actualizaciones')
    fig.tight_layout();fig.savefig(output/'comparacion.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    arrays=[np.asarray(references[v]['u_matrix']) for v in VARIANTS]
    lo,hi=min(a.min() for a in arrays),max(a.max() for a in arrays)
    for ax,(v,reference),array in zip(axes,references.items(),arrays):
        view=ax.imshow(array,vmin=lo,vmax=hi,cmap='viridis')
        counts=np.bincount(np.asarray(reference['labels']),minlength=5)
        ax.set_title(f'{NAMES[v]}\nQE {reference["train_metrics"]["quantization_error"]:.3f}; regiones ocupadas {np.count_nonzero(counts)}')
        ax.set_xticks(range(4));ax.set_yticks(range(4))
    fig.colorbar(view,ax=axes.tolist(),label='Distancia media (U)',shrink=.8)
    fig.savefig(output/'matrices_u.png',dpi=150,bbox_inches='tight');plt.close(fig)


def summarize(output):
    protocol=json.loads((output/'protocol.json').read_text())
    if protocol['smoke']:
        raise ValueError('La tanda smoke no es un experimento completo')
    if fingerprint(ROOT/'presentation')!=protocol['presentation_fingerprint']:
        raise ValueError('Cambió la presentación; revisar su integridad')
    if digest(ROOT/'data/creadores.csv')!=protocol['data_sha256']:
        raise ValueError('Cambió el catálogo')
    for name,expected in protocol['sources'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f'Cambió la fuente {name}')
    runs=[json.loads(s) for s in (output/'runs.jsonl').read_text().splitlines()]
    if sorted(task_key(r['task']) for r in runs)!=sorted(task_key(t) for t in protocol['tasks']):
        raise ValueError('Tareas faltantes o duplicadas')
    controls=[json.loads(s) for s in (output/'controls.jsonl').read_text().splitlines()]
    baseline={(task_key(c['task']),c['strength']):c for c in controls}
    records=[];lookup={};failures=[]
    for row in controls:
        task=row['task'];record=flattened(task,row,row)
        records.append(record)
        lookup[(record['stage'],record['candidate'],record['sample'],record['seed'],'gaussian',record['strength'])]=row['labels']
    total=0
    for run in runs:
        task=run['task']
        expected={(v,s) for v in VARIANTS for s in ([0.,.05,.15,.30] if task['stage']=='augmentation' else [0.])}
        observed=[(r['variant'],r['strength']) for r in run['results']]
        if len(observed)!=len(expected) or set(observed)!=expected:
            raise ValueError('Condiciones incompletas o duplicadas')
        for result in run['results']:
            total+=1
            if result['status']!='ok':
                failures.append(dict(task=task,**result));continue
            expected_updates=50000 if task['stage']=='finalists' else 40000
            if result['epochs']!=1000 or result['updates']!=expected_updates:
                raise ValueError('Presupuesto distinto del protocolo')
            paired=baseline[(task_key(task),result['strength'])]
            record=flattened(task,result,paired)
            records.append(record)
            lookup[(record['stage'],record['candidate'],record['sample'],record['seed'],record['variant'],record['strength'])]=result['labels']
    if total!=protocol['fits']:
        raise ValueError('Cantidad de modelos incorrecta')
    (output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    if failures:
        raise ValueError(f'{len(failures)} modelos fallidos; no se genera una comparación incompleta')
    frame=pd.DataFrame(records)
    frame.to_csv(output/'runs.csv',index=False)
    table=summary_table(frame,lookup)
    table.to_csv(output/'summary.csv',index=False)
    for stage in ('finalists','subsamples','augmentation'):
        table[table.stage==stage].to_csv(output/f'{stage}.csv',index=False)
    augmentation=augmentation_table(frame,protocol['augmentation_gates'])
    augmentation.to_csv(output/'augmentation_gates.csv',index=False)
    references=json.loads((output/'uniform_references.json').read_text())
    figures(output,frame,references)
    summary=dict(new_fits=total,reused_gaussian_fits=len(controls),reference_fits=2,failures=len(failures),
        presentation_unchanged=True,rows=json.loads(table.to_json(orient='records',double_precision=12)),
        augmentation_gates=json.loads(augmentation.to_json(orient='records',double_precision=12)))
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(table[['stage','candidate','variant','strength','train_qe','train_te','excluded_qe','ari_between_seeds',
                 'art_overlap','art_short']].to_string(index=False))
    print(augmentation[['variant','strength','relative_qe_improvement','split_win_fraction','eligible']].to_string(index=False))
    print(json.dumps(dict(new_fits=total,reused_gaussian_fits=len(controls),presentation_unchanged=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    summarize(parser.parse_args().output)
