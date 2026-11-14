"""Búsqueda progresiva de calidad, con fórmulas AE/VAE protegidas."""

import argparse
import json
import csv
import copy
from pathlib import Path

import numpy as np

from src import plotting
from src.data.art_capacity import read_json,file_hash,load_probe
from src.data.art_search import (CANDIDATES,GENERATION_SEEDS,prepare_search,write_config,assert_loss_unchanged)
from src.data.factory import build_dataset
from src.data.art import render_art
from src.evaluation.art_geometry import reconstruction_scores,summarize_geometry
from src.evaluation.art_cleanliness import cleanliness,clean_pass
from src.evaluation.art_factorial import load_source_model
from src.evaluation.art_search_cache import audited_legacy_cache
from src.data.art_search_expanded import validate_expanded_training
from src.evaluation.context import write_csv
from src.evaluation.generative import nearest_training
from src.experiments.grid import expand_runs
from src.options import ConfigurationError,Options

ROOT=Path(__file__).resolve().parents[2]
FAMILIES=('circle','ellipse','triangle','rectangle')


def native_references(samples,records,size):
    if size==32:return samples,records
    result=copy.deepcopy(records)
    for record in result:
        for obj in record['objects']:
            for key in ('center_x','center_y','half_width','half_height'):obj[key]*=size/32
    return np.array([render_art(size,r['objects']).ravel() for r in result],dtype=np.float32),result


def evaluate(root,name,final=False):
    root=Path(root);assert_loss_unchanged(root)
    config=read_json(root/f'configs/ej2_search_art_{name}.json')
    seeds=config['seeds']
    split='reserved_final' if final else 'validation'
    data=root/'results/art_search/data'; manifest=read_json(data/'manifest.json')
    samples,records=load_probe(data,split)
    size=config['dataset']['image_size'];shape=(size,size,3)
    samples,records=native_references(samples,records,size)
    output=root/f"results/art_search/{split}/{name}/seeds_{'_'.join(map(str,seeds))}"
    sources=[]
    for run in expand_runs(config):
        path=root/f'results/ej2_search_art_{name}/{run.label}'
        if read_json(path/'config.json')!=run.config:raise ConfigurationError('Config guardada incompatible')
        read_json(path/'summary.json')
        sources.append({'seed':run.seed,'run':str(path.relative_to(root)),
                        'model_sha256':file_hash(path/'model.npz'),'config_sha256':file_hash(path/'config.json')})
    inputs={'sources':sources,'data_manifest_sha256':file_hash(data/'manifest.json'),
            'loss_guard':assert_loss_unchanged(root),
            'evaluation_sha256':{p:file_hash(root/p) for p in ['src/evaluation/art_geometry.py',
                'src/evaluation/art_cleanliness.py','src/nn/spatial.py','src/models/factory.py']}}
    if size!=32:
        import hashlib
        inputs['native_reference_transform']={'version':1,'size':size,'samples_sha256':hashlib.sha256(samples.tobytes()).hexdigest()}
    unbounded=config['model'].get('output_activation','sigmoid')=='identity'
    if config['dataset']['train_count']>4096:
        inputs['expanded_data_manifest_sha256']=file_hash(root/'results/art_search/expanded_data_manifest.json')
    if config['model']['kl']['weight']!=.5:
        inputs['objective_hyperparameters']={'formula':'MSE + beta * KL','beta':config['model']['kl']['weight'],
                                            'warmup_epochs':config['model']['kl']['warmup_epochs']}
    if unbounded:inputs['raw_output_policy']={'range_rmse_max':.005,'corrective_postprocessing':False}
    if (output/'status.json').exists():
        old=read_json(output/'status.json')
        if old['inputs']!=inputs and not audited_legacy_cache(root,name,old['inputs'],inputs):
            raise ConfigurationError('Cambió una evaluación ya registrada')
        return output,old
    training=build_dataset(Options(config['dataset'],'dataset'),root)
    if config['dataset']['train_count']>4096:validate_expanded_training(root,training)
    if {r.tobytes() for r in training.samples}.intersection(r.tobytes() for r in samples):
        raise ConfigurationError('Las referencias entraron al entrenamiento')
    rows=[];family_rows=[];all_scores=[]
    output.mkdir(parents=True,exist_ok=True)
    epsilon=np.random.default_rng(GENERATION_SEEDS[split]).standard_normal((64,config['model']['latent_size']))
    shown=np.array([0,1,2,3,256,257,258,259])
    for source in sources:
        seed=source['seed'];print(f'Evaluando {name}, {split}, semilla {seed}',flush=True)
        model=load_source_model(root,source)
        reconstructed=np.concatenate([model.reconstruct(samples[i:i+64].astype(model.precision))
                                      for i in range(0,len(samples),64)])
        scores=[{'label':r['label'],'family':r['family'],
                 **reconstruction_scores(a.reshape(shape),b.reshape(shape),r)}
                for a,b,r in zip(samples,reconstructed,records)]
        summary,families=summarize_geometry(scores)
        generated=model.decode(epsilon.astype(model.precision))
        clean=[{'sample':i,**cleanliness(image.reshape(shape))} for i,image in enumerate(generated)]
        for row,image in zip(clean,generated):
            row['clean']=clean_pass(row,manifest['cleanliness_limits'])
            if unbounded:
                row['range_rmse']=float(np.sqrt(np.mean((image-np.clip(image,0,1))**2)))
                row['clean']=bool(row['clean'] and row['range_rmse']<=.005)
        rate=float(np.mean([row['clean'] for row in clean]))
        directory=output/f'seed={seed}';directory.mkdir(exist_ok=True)
        write_csv(directory/'reconstruction.csv',scores);write_csv(directory/'cleanliness.csv',clean)
        nearest,indices=nearest_training(generated,training.samples)
        write_csv(directory/'nearest.csv',[{'sample':i,'label':training.labels[j],'mse':float(d)}
                  for i,(d,j) in enumerate(zip(nearest,indices))])
        np.savez_compressed(directory/'generated.npz',samples=generated,coordinates=epsilon)
        plotting.save_image_mosaic(generated,(8,8),shape,False,
                                  f'{name}, semilla {seed}: 64 salidas sin filtrar',directory/'generated.png')
        plotting.save_image_rows([samples[shown],reconstructed[shown]],['Original','Reconstrucción'],
                                 [records[i]['family'] for i in shown],shape,False,directory/'reconstruction.png')
        pick=np.arange(0,64,8)
        plotting.save_image_rows([generated[pick],training.samples[indices[pick]]],['Generada','Entrenamiento próximo'],
                                 [str(i) for i in pick],shape,False,directory/'nearest.png')
        rows.append({'seed':seed,'clean_rate':rate,**summary})
        family_rows.extend({'seed':seed,**r} for r in families);all_scores.extend(clean)
    criterion=manifest['criteria']
    ious=[r['macro_silhouette_iou'] for r in rows];rates=[r['clean_rate'] for r in rows]
    polygon=[r['silhouette_iou'] for r in family_rows if r['family'] in ('triangle','rectangle')]
    polygon_means={family:float(np.mean([r['silhouette_iou'] for r in family_rows if r['family']==family]))
                   for family in ('triangle','rectangle')}
    passed=(np.mean(ious)>=criterion['macro_iou_mean'] and min(ious)>=criterion['macro_iou_min']
            and all(v>=criterion['polygon_iou_mean'] for v in polygon_means.values()) and min(polygon)>=criterion['polygon_iou_min']
            and np.mean(rates)>=criterion['clean_rate_mean'] and min(rates)>=criterion['clean_rate_min'])
    status={'candidate':name,'seeds':seeds,'inputs':inputs,'split':split,'resolution':size,
            'macro_iou_mean':float(np.mean(ious)),'polygon_iou_mean':float(np.mean(polygon)),
            'polygon_iou_means':polygon_means,
            'clean_rate_mean':float(np.mean(rates)),'automatic_pass':bool(passed),
            'confirmation_seeds_complete':seeds==[7,17,27],
            'manual_diversity_review_required':True,'model_samples_filtered':False}
    write_csv(output/'runs.csv',rows);write_csv(output/'families.csv',family_rows)
    (output/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in status.items() if k!='inputs'},indent=2),flush=True)
    return output,status


def check_coverage(path,evaluation):
    coverage=read_json(path)
    for seed in (7,17,27):
        labels=coverage.get(str(seed),{})
        with (evaluation/f'seed={seed}/cleanliness.csv').open() as stream:
            clean={int(row['sample']):row['clean']=='True' for row in csv.DictReader(stream)}
        used=set()
        for family in FAMILIES:
            indices=labels.get(family,[])
            if len(set(indices))<5 or any(type(i) is not int or not 0<=i<64 for i in indices):
                raise ConfigurationError(f'Falta evidencia de diversidad: semilla {seed}, {family}')
            if used.intersection(indices):raise ConfigurationError('Una muestra recibió dos familias distintas')
            if not all(clean.get(i,False) for i in indices):raise ConfigurationError('La evidencia de diversidad incluye muestras no limpias')
            used.update(indices)
    return coverage


def freeze(root,name,coverage_path):
    directory,status=evaluate(root,name)
    if not status['automatic_pass'] or not status['confirmation_seeds_complete']:
        raise ConfigurationError('La configuración no cumple el criterio automático en las tres semillas')
    coverage=check_coverage(coverage_path,directory)
    selection={'candidate':name,'validation_status_sha256':file_hash(directory/'status.json'),
               'coverage':coverage,'inputs':status['inputs'],'criterion_fixed_before_search':True}
    path=root/'results/art_search/selection.json'
    if path.exists() and read_json(path)!=selection:raise ConfigurationError('La elección final ya está congelada')
    path.write_text(json.dumps(selection,indent=2),encoding='utf-8')
    return selection


def final_check(root):
    selection=read_json(root/'results/art_search/selection.json')
    validation,status=evaluate(root,selection['candidate'])
    if file_hash(validation/'status.json')!=selection['validation_status_sha256']:
        raise ConfigurationError('Cambió la selección antes de evaluar final')
    return evaluate(root,selection['candidate'],final=True)


def accept_final(root,coverage_path):
    output,status=final_check(root)
    if not status['automatic_pass'] or not status['confirmation_seeds_complete']:
        raise ConfigurationError('La confirmación final no cumplió los criterios')
    coverage=check_coverage(coverage_path,output)
    result={'status':'satisfactory','candidate':status['candidate'],'final_status_sha256':file_hash(output/'status.json'),
            'coverage':coverage,'criteria':read_json(root/'results/art_search/data/manifest.json')['criteria'],
            'interpretation':'Predefined cleanliness and reconstruction criteria plus explicit visual family review (reviewer identified in coverage); not universal validity.'}
    (root/'results/art_search/satisfactory.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare',action='store_true')
    actions.add_argument('--evaluate',nargs='+',choices=list(CANDIDATES))
    actions.add_argument('--promote',choices=list(CANDIDATES))
    actions.add_argument('--freeze',choices=list(CANDIDATES))
    actions.add_argument('--final',action='store_true')
    actions.add_argument('--accept-final',action='store_true')
    parser.add_argument('--coverage',type=Path)
    args=parser.parse_args()
    try:
        if args.prepare:prepare_search(ROOT)
        elif args.evaluate:
            for name in args.evaluate:evaluate(ROOT,name)
        elif args.promote:print(write_config(ROOT,args.promote,(7,17,27)))
        elif args.freeze:
            if args.coverage is None:raise ConfigurationError('--freeze requiere --coverage')
            freeze(ROOT,args.freeze,args.coverage)
        elif args.accept_final:
            if args.coverage is None:raise ConfigurationError('--accept-final requiere --coverage')
            accept_final(ROOT,args.coverage)
        else:final_check(ROOT)
    except (ConfigurationError,ValueError,FileNotFoundError) as error:
        parser.exit(1,f'Error: {error}\n')


if __name__=='__main__':main()
