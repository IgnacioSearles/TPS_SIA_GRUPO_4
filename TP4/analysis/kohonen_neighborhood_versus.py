"""Versus anidado y equilibrado: uniforme frente a gaussiana, sin augmentation.

Se calibra dentro de 40 creadores y se confirma sobre diez excluidos de la
calibración. 3 repeticiones × 5 particiones externas. No toca presentation/.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.tp4.kohonen import Kohonen
from src.tp4.numerics import distances
from src.tp4.preprocessing import Preprocessor
from .confirm_exercise1 import regions
from .kohonen_experiments import adjusted_rand_index
from .kohonen_uniform_experiments import ROOT, digest, fingerprint
from .som_campanas import MODEL

GATES=dict(max_validation_te=.30,max_empty_fraction=.20,min_region_ari=.60)
FAMILIES=('gaussian','uniform')


def candidates():
    definitions=[('const05',.5,.5,'constant'),('const1',1.,1.,'constant'),
                 ('const2',2.,2.,'constant'),('decay205',2.,.5,'exponential'),
                 ('decay21',2.,1.,'exponential')]
    result=[]
    for family in FAMILIES:
        for name,start,end,decay in definitions:
            for rate in (.1,.3):
                cfg=replace(MODEL,neighborhood=family,radius_start=start,radius_end=end,
                            radius_decay=decay,learning_rate_start=rate)
                result.append(dict(id=f'{family}_{name}_eta{int(rate*1000)}',family=family,
                                   radius=name,rate=rate,config=asdict(cfg)))
    return result


def outer_partitions():
    rng=np.random.default_rng(2026102601)
    result=[]
    for repetition in range(3):
        tests=np.array_split(rng.permutation(50),5)
        for fold,test in enumerate(tests):
            index=repetition*5+fold
            test=sorted(test.tolist())
            train=sorted(set(range(50))-set(test))
            inner_rng=np.random.default_rng([20261026,index,777])
            inner=[]
            for number in range(3):
                fit=sorted(inner_rng.choice(train,32,replace=False).tolist())
                validation=sorted(set(train)-set(fit))
                inner.append(dict(index=number,train=fit,validation=validation,
                                  seeds=[8000+index*100+number*10+s for s in range(2)]))
            result.append(dict(index=index,repetition=repetition,fold=fold,train=train,test=test,
                               inner=inner,seeds=[12000+index*100+s for s in range(3)]))
    return result


def configured(candidate,seed):
    # Partir del mismo modelo y sobrescribir la configuración declarada.
    return replace(MODEL,**{k:v for k,v in candidate['config'].items() if k not in ['seed','early_stopping']},seed=seed)


def calibrate(job,data):
    frame=pd.read_csv(data)
    outer,candidate=job['outer'],job['candidate']
    records=[]
    for inner in outer['inner']:
        prep=Preprocessor().fit(frame.iloc[inner['train']])
        fit=prep.transform(frame.iloc[inner['train']])
        validation=prep.transform(frame.iloc[inner['validation']])
        # ARI usa solo los 40 externos de entrenamiento, nunca los diez de prueba.
        comparison=prep.transform(frame.iloc[outer['train']])
        for seed in inner['seeds']:
            start=time.perf_counter()
            model=Kohonen(configured(candidate,seed)).fit(fit)
            records.append(dict(inner=inner['index'],seed=seed,train_metrics=asdict(model.evaluate(fit)),
                validation_metrics=asdict(model.evaluate(validation)),
                empty_fraction=float((model.occupancy(fit)==0).mean()),
                labels=regions(model,comparison).tolist(),epochs=model.epochs_,updates=model.updates_,
                seconds=time.perf_counter()-start))
    return dict(outer_index=outer['index'],candidate_id=candidate['id'],family=candidate['family'],models=records)


def candidate_summary(row):
    models=row['models']
    aris=[]
    for index in range(3):
        group=[r for r in models if r['inner']==index]
        if len(group)!=2:
            raise ValueError('Cada validación interna debe tener dos semillas')
        aris.append(adjusted_rand_index(group[0]['labels'],group[1]['labels']))
    if any(r['epochs']!=1000 or r['updates']!=32000 for r in models) or len(models)!=6:
        raise ValueError('Presupuesto interno incorrecto')
    result=dict(outer_index=row['outer_index'],candidate_id=row['candidate_id'],family=row['family'],
        validation_qe=float(np.mean([r['validation_metrics']['quantization_error'] for r in models])),
        validation_te=float(np.mean([r['validation_metrics']['topographic_error'] for r in models])),
        empty_fraction=float(np.mean([r['empty_fraction'] for r in models])),region_ari=float(np.mean(aris)))
    result['eligible']=(result['validation_te']<=GATES['max_validation_te'] and
                        result['empty_fraction']<=GATES['max_empty_fraction'] and
                        result['region_ari']>=GATES['min_region_ari'])
    return result


def choose(rows,family):
    rows=[r for r in rows if r['family']==family]
    eligible=[r for r in rows if r['eligible']]
    if eligible:
        winner=min(eligible,key=lambda r:(r['validation_qe'],r['validation_te'],r['candidate_id']))
    else:
        # Fallback predeclarado: permite medir la familia, sin afirmar que sea apta.
        winner=min(rows,key=lambda r:(r['validation_te'],r['validation_qe'],r['empty_fraction'],r['candidate_id']))
    return dict(**winner,eligible_count=len(eligible),fallback=not bool(eligible))


def confirm(job,data):
    frame=pd.read_csv(data)
    outer=job['outer']
    prep=Preprocessor().fit(frame.iloc[outer['train']])
    x=prep.transform(frame)
    records=[]
    for candidate in job['candidates']:
        for seed in outer['seeds']:
            start=time.perf_counter()
            model=Kohonen(configured(candidate,seed)).fit(x[outer['train']])
            test=x[outer['test']]
            ds=distances(test,model.weights_)
            first_two=np.argsort(ds,axis=1,kind='stable')[:,:2]
            point_te=~model.grid.adjacency[first_two[:,0],first_two[:,1]]
            records.append(dict(family=candidate['family'],candidate_id=candidate['id'],seed=seed,
                train_metrics=asdict(model.evaluate(x[outer['train']])),test_metrics=asdict(model.evaluate(test)),
                point_qe=ds.min(axis=1).tolist(),point_te=point_te.astype(int).tolist(),
                labels=regions(model,x).tolist(),empty_fraction=float((model.occupancy(x[outer['train']])==0).mean()),
                epochs=model.epochs_,updates=model.updates_,seconds=time.perf_counter()-start))
    return dict(outer_index=outer['index'],models=records)


def run_jobs(jobs,path,worker,data,workers,key):
    completed=[]
    if path.exists():
        completed=[json.loads(line) for line in path.read_text().splitlines()]
    seen={key(r) for r in completed}
    pending=[job for job in jobs if key(job) not in seen]
    with ProcessPoolExecutor(max_workers=workers) as pool,path.open('a') as stream:
        futures=[pool.submit(worker,job,data) for job in pending]
        for count,future in enumerate(as_completed(futures),len(completed)+1):
            row=future.result();completed.append(row)
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if count%10==0 or count==len(jobs):
                print(f'{path.stem}: {count}/{len(jobs)} bloques',flush=True)
    if len(completed)!=len(jobs) or len({key(r) for r in completed})!=len(jobs):
        raise ValueError('Bloques incompletos o duplicados')
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    outer=outer_partitions();options=candidates()
    if args.smoke:
        outer=outer[:1]
        options=[c for c in options if c['radius']=='decay21' and c['rate']==.3]
    source_paths=list((ROOT/'src/tp4').glob('*.py'))+[Path(__file__),ROOT/'analysis/confirm_exercise1.py',
        ROOT/'analysis/kohonen_uniform_experiments.py',ROOT/'analysis/kohonen_experiments.py',
        ROOT/'analysis/som_campanas.py',ROOT/'analysis/som_regions.py',ROOT/'analysis/eda.py']
    protocol=dict(smoke=args.smoke,data_sha256=digest(ROOT/'data/creadores.csv'),
        sources={str(p.relative_to(ROOT)):digest(p) for p in source_paths},
        presentation_fingerprint=fingerprint(ROOT/'presentation'),candidates=options,outer_partitions=outer,
        calibration_fits=len(outer)*len(options)*6,confirmation_fits=len(outer)*2*3,gates=GATES,
        selection_rule='Dentro de cada familia: menor QE de validación entre aptos; desempate TE e id.',
        fallback_rule='Sin aptos: menor TE, después QE, vacías e id; se marca fallback y no se declara apto.',
        limitations=['Solo catálogo fijo de 50; tres repeticiones de cinco particiones externas.',
            'Cada creador aparece tres veces como prueba externa en la tanda completa.',
            'Los diez externos quedan excluidos de toda calibración de ese bloque.',
            'La confirmación compara procedimientos calibrados; la configuración puede variar por partición.',
            'La segunda etapa usa exclusivamente selecciones congeladas antes de medir pruebas externas.',
            'Se mantienen grilla rectangular 4x4, datos, presupuesto y calendario; no hay augmentation.',
            'No se elige ni se despliega automáticamente una única configuración final global.'])
    if args.resume:
        if json.loads((args.output/'protocol.json').read_text())!=protocol:
            raise ValueError('Cambió el protocolo; no se reanuda')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
    data=str(ROOT/'data/creadores.csv')
    jobs=[dict(outer=o,candidate=c,outer_index=o['index'],candidate_id=c['id']) for o in outer for c in options]
    key=lambda row:(row['outer_index'],row['candidate_id'])
    print(f'Plan anidado: {protocol["calibration_fits"]} calibraciones + {protocol["confirmation_fits"]} confirmaciones',flush=True)
    calibration=run_jobs(jobs,args.output/'calibration.jsonl',calibrate,data,args.workers,key)
    summaries=[candidate_summary(row) for row in calibration]
    selections=[]
    for o in outer:
        group=[r for r in summaries if r['outer_index']==o['index']]
        for family in FAMILIES:
            selections.append(choose(group,family))
    selection_file=args.output/'selections.json'
    payload=dict(gates=GATES,selection=selections,data_sha256=protocol['data_sha256'],
                 calibration_sha256=digest(args.output/'calibration.jsonl'))
    if selection_file.exists() and json.loads(selection_file.read_text())!=payload:
        raise ValueError('Cambió la selección congelada')
    selection_file.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    pd.DataFrame(summaries).to_csv(args.output/'calibration_summary.csv',index=False)
    selected_ids={(s['outer_index'],s['family']):s['candidate_id'] for s in selections}
    confirmation_jobs=[]
    for o in outer:
        chosen=[next(c for c in options if c['id']==selected_ids[o['index'],f]) for f in FAMILIES]
        confirmation_jobs.append(dict(outer=o,candidates=chosen,outer_index=o['index']))
    print('Selecciones congeladas. Comienza la evaluación externa.',flush=True)
    confirmation=run_jobs(confirmation_jobs,args.output/'confirmation.jsonl',confirm,data,args.workers,lambda row:row['outer_index'])
    if any(len(r['models'])!=6 for r in confirmation):
        raise ValueError('Faltan modelos externos')
    if fingerprint(ROOT/'presentation')!=protocol['presentation_fingerprint']:
        raise RuntimeError('Cambió presentation/; revisar antes de concluir')
    (args.output/'COMPLETED').write_text('Finalizaron calibración y confirmación; el consolidado verifica métricas.\n')


if __name__=='__main__':
    main()
