"""Consolida el versus anidado, sin entrenar ni modificar presentación."""
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
from .kohonen_neighborhood_versus import candidate_summary,choose,FAMILIES
from .kohonen_uniform_experiments import ROOT,digest,fingerprint


def assert_protocol(output,protocol):
    if protocol['smoke'] or not (output/'COMPLETED').exists():
        raise ValueError('Se requiere la tanda completa terminada')
    if digest(ROOT/'data/creadores.csv')!=protocol['data_sha256']:
        raise ValueError('Cambió el catálogo')
    for name,expected in protocol['sources'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f'Cambió la fuente {name}')
    if fingerprint(ROOT/'presentation')!=protocol['presentation_fingerprint']:
        raise ValueError('Cambió presentation/')


def summarize(output):
    protocol=json.loads((output/'protocol.json').read_text())
    assert_protocol(output,protocol)
    calibration=[json.loads(s) for s in (output/'calibration.jsonl').read_text().splitlines()]
    outer=[json.loads(s) for s in (output/'confirmation.jsonl').read_text().splitlines()]
    candidates={c['id']:c for c in protocol['candidates']}
    partitions={o['index']:o for o in protocol['outer_partitions']}
    expected={(i,c) for i in partitions for c in candidates}
    actual=[(r['outer_index'],r['candidate_id']) for r in calibration]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Faltan o se duplican calibraciones')
    for row in calibration:
        part=partitions[row['outer_index']]
        for inner in part['inner']:
            observed=sorted(r['seed'] for r in row['models'] if r['inner']==inner['index'])
            if observed!=sorted(inner['seeds']):
                raise ValueError('Semillas internas distintas de las declaradas')
    if len(outer)!=len(partitions) or {o['outer_index'] for o in outer}!=set(partitions):
        raise ValueError('Faltan o se duplican bloques externos')
    calibrated=[candidate_summary(r) for r in calibration]
    frozen=json.loads((output/'selections.json').read_text())
    if frozen['data_sha256']!=protocol['data_sha256']:
        raise ValueError('La selección pertenece a otro catálogo')
    if digest(output/'calibration.jsonl')!=frozen['calibration_sha256']:
        raise ValueError('Cambió calibración tras congelar elecciones')
    selected={(s['outer_index'],s['family']):s for s in frozen['selection']}
    if len(selected)!=len(partitions)*2:
        raise ValueError('Selecciones incompletas')
    for index in partitions:
        rows=[r for r in calibrated if r['outer_index']==index]
        for family in FAMILIES:
            check=choose(rows,family)
            old=selected[index,family]
            if any(check[k]!=old[k] for k in ('candidate_id','fallback','eligible_count')):
                raise ValueError('Elección no reproducible con los criterios previos')
    model_rows=[];point_rows=[];region_rows=[]
    for row in outer:
        index=row['outer_index'];part=partitions[index]
        observed=[(r['family'],r['seed']) for r in row['models']]
        expected={(f,s) for f in FAMILIES for s in part['seeds']}
        if len(observed)!=len(expected) or set(observed)!=expected:
            raise ValueError('Modelos externos incompletos')
        for record in row['models']:
            family=record['family'];selection=selected[index,family]
            if record['candidate_id']!=selection['candidate_id']:
                raise ValueError('Se evaluó otro modelo que el elegido')
            if record['epochs']!=1000 or record['updates']!=40000:
                raise ValueError('Presupuesto externo incorrecto')
            np.testing.assert_allclose(np.mean(record['point_qe']),record['test_metrics']['quantization_error'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.mean(record['point_te']),record['test_metrics']['topographic_error'],rtol=0,atol=1e-12)
            candidate=candidates[record['candidate_id']]
            model_rows.append(dict(outer_index=index,repetition=part['repetition'],fold=part['fold'],
                family=family,candidate_id=record['candidate_id'],seed=record['seed'],radius=candidate['radius'],rate=candidate['rate'],
                fallback=selection['fallback'],train_qe=record['train_metrics']['quantization_error'],
                train_te=record['train_metrics']['topographic_error'],test_qe=record['test_metrics']['quantization_error'],
                test_te=record['test_metrics']['topographic_error'],empty_fraction=record['empty_fraction']))
            if len(record['point_qe'])!=len(part['test']) or len(record['point_te'])!=len(part['test']):
                raise ValueError('Métricas individuales de prueba incompletas')
            for creator,qe,te in zip(part['test'],record['point_qe'],record['point_te']):
                point_rows.append(dict(outer_index=index,repetition=part['repetition'],fold=part['fold'],family=family,
                    seed=record['seed'],creator_index=creator,qe=qe,te=te))
        for family in FAMILIES:
            group=[r['labels'] for r in row['models'] if r['family']==family]
            aris=[adjusted_rand_index(a,b) for a,b in combinations(group,2)]
            region_rows.append(dict(outer_index=index,family=family,region_ari=np.mean(aris)))
    if sum(len(r['models']) for r in calibration)!=protocol['calibration_fits'] or len(model_rows)!=protocol['confirmation_fits']:
        raise ValueError('Cantidad de modelos incorrecta')
    models=pd.DataFrame(model_rows);points=pd.DataFrame(point_rows);stability=pd.DataFrame(region_rows)
    counts=points.groupby(['family','creator_index']).size()
    if len(counts)!=100 or not (counts==9).all():
        raise ValueError('Cada creador debe tener nueve evaluaciones externas por familia')
    models.to_csv(output/'confirmation.csv',index=False)
    points.to_csv(output/'creator_evaluations.csv',index=False)
    stability.to_csv(output/'region_stability.csv',index=False)
    columns=['train_qe','train_te','test_qe','test_te','empty_fraction']
    block=models.groupby(['outer_index','family'])[columns].mean().reset_index()
    block=block.merge(stability,on=['outer_index','family'],validate='one_to_one')
    block.to_csv(output/'outer_means.csv',index=False)
    gaussian=block[block.family=='gaussian'].set_index('outer_index')
    uniform=block[block.family=='uniform'].set_index('outer_index')
    differences=uniform[columns+['region_ari']]-gaussian[columns+['region_ari']]
    differences= differences.rename(columns=lambda c:'delta_'+c)
    differences.to_csv(output/'paired_differences.csv')
    per_creator=points.groupby(['creator_index','family'])[['qe','te']].mean().unstack('family')
    creator=pd.DataFrame(dict(delta_qe=per_creator['qe']['uniform']-per_creator['qe']['gaussian'],
                              delta_te=per_creator['te']['uniform']-per_creator['te']['gaussian']))
    creator.to_csv(output/'paired_creators.csv')
    selection_table=pd.DataFrame(frozen['selection'])
    frequency=selection_table.groupby(['family','candidate_id']).agg(outer_blocks=('outer_index','size'),
                fallback_blocks=('fallback','sum')).reset_index()
    frequency.to_csv(output/'selection_frequency.csv',index=False)
    calib=pd.DataFrame(calibrated)
    calib=calib.merge(pd.DataFrame([dict(candidate_id=c['id'],radius=c['radius'],rate=c['rate']) for c in candidates.values()]),on='candidate_id',validate='many_to_one')
    averages=calib.groupby(['family','candidate_id','radius','rate']).agg(validation_qe=('validation_qe','mean'),
        validation_te=('validation_te','mean'),empty_fraction=('empty_fraction','mean'),region_ari=('region_ari','mean'),
        eligible_blocks=('eligible','sum')).reset_index()
    averages.to_csv(output/'candidate_averages.csv',index=False)
    paired_calib=averages.pivot(index=['radius','rate'],columns='family',values=['validation_qe','validation_te','region_ari'])
    paired_calib.columns=['_'.join(c) for c in paired_calib.columns]
    paired_calib.to_csv(output/'matched_calibration.csv')
    summary_rows=[]
    for family in FAMILIES:
        g=block[block.family==family]
        summary_rows.append(dict(family=family,models=int((models.family==family).sum()),outer_blocks=len(g),
            train_qe=g.train_qe.mean(),test_qe=g.test_qe.mean(),train_te=g.train_te.mean(),test_te=g.test_te.mean(),
            empty_fraction=g.empty_fraction.mean(),region_ari=g.region_ari.mean(),
            fallback_blocks=int(selection_table.loc[selection_table.family==family,'fallback'].sum())))
    comparison=dict(qe_difference_uniform_minus_gaussian=differences.delta_test_qe.mean(),
        qe_difference_p05=differences.delta_test_qe.quantile(.05),qe_difference_p95=differences.delta_test_qe.quantile(.95),
        uniform_wins_outer_qe=int((differences.delta_test_qe<0).sum()),outer_blocks=len(differences),
        te_difference_uniform_minus_gaussian=differences.delta_test_te.mean(),
        uniform_wins_outer_te=int((differences.delta_test_te<0).sum()),
        uniform_wins_creators_qe=int((creator.delta_qe<0).sum()),creator_count=len(creator),
        relative_qe_change=float(differences.delta_test_qe.mean()/gaussian.test_qe.mean()))
    both_valid=[i for i in partitions if not selected[i,'gaussian']['fallback'] and not selected[i,'uniform']['fallback']]
    valid=differences.loc[both_valid]
    comparison['both_families_eligible_blocks']=len(valid)
    comparison['qe_difference_both_eligible']=float(valid.delta_test_qe.mean()) if len(valid) else None
    comparison['uniform_wins_both_eligible']=int((valid.delta_test_qe<0).sum())
    summary=dict(calibration_fits=protocol['calibration_fits'],confirmation_fits=protocol['confirmation_fits'],
        models=summary_rows,paired=comparison,presentation_unchanged=True,
        selection_frequency=json.loads(frequency.to_json(orient='records')),
        note='Los percentiles describen 15 particiones dependientes. No son intervalos de confianza ni creadores nuevos.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    pd.DataFrame(summary_rows).to_csv(output/'summary.csv',index=False)
    plot_results(output,block,differences,frequency,candidates)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(json.dumps(comparison,indent=2))
    print(frequency.to_string(index=False))


def plot_results(output,block,differences,frequency,candidates):
    names=['Gaussiana calibrada','Uniforme calibrada']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,metric,title in [(axes[0],'test_qe','QE externo (media de tres semillas)'),
                             (axes[1],'test_te','TE externo (media de tres semillas)')]:
        ax.boxplot([block.loc[block.family==f,metric] for f in FAMILIES],tick_labels=names)
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    values=differences.delta_test_qe.to_numpy()
    axes[2].bar(np.arange(15)+1,values,color=['#b87537' if v>0 else '#287b96' for v in values])
    axes[2].axhline(0,color='#555',linewidth=1)
    axes[2].set(title='QE uniforme − gaussiana',xlabel='Partición externa',ylabel='Negativo: favorece a uniforme',xticks=[1,5,10,15])
    fig.suptitle('Comparación anidada — prueba externa excluida de la calibración')
    fig.tight_layout();fig.savefig(output/'versus_externo.png',dpi=150);plt.close(fig)
    radii=['const05','const1','const2','decay205','decay21']
    titles=['Constante 0,5','Constante 1','Constante 2','2 → 0,5','2 → 1']
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,family,name in zip(axes,FAMILIES,['Gaussiana','Uniforme']):
        matrix=np.zeros((2,5),dtype=int)
        for r in frequency[frequency.family==family].itertuples():
            c=candidates[r.candidate_id]
            matrix[0 if c['rate']==.1 else 1,radii.index(c['radius'])]=r.outer_blocks
        ax.imshow(matrix,vmin=0,vmax=15,cmap='Blues',aspect='auto')
        for (y,x),n in np.ndenumerate(matrix):ax.text(x,y,str(n),ha='center',va='center',color='white' if n>7 else 'black')
        ax.set(xticks=range(5),xticklabels=titles,yticks=[0,1],yticklabels=['η₀ = 0,1','η₀ = 0,3'],title=name)
        ax.tick_params(axis='x',labelrotation=25)
    fig.suptitle('Elecciones internas en 15 particiones — sin usar su prueba externa')
    fig.tight_layout();fig.savefig(output/'selecciones.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    summarize(parser.parse_args().output)
