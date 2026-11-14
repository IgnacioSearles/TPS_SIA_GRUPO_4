"""Auditoría adicional de muestreo en validación, sin abrir el conjunto final."""

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

from src import plotting
from src.data.art_capacity import read_json,file_hash
from src.data.art_search import CANDIDATES,assert_loss_unchanged
from src.evaluation.art_cleanliness import cleanliness,clean_pass
from src.evaluation.art_factorial import load_source_model
from src.evaluation.context import write_csv
from src.options import ConfigurationError

ROOT=Path(__file__).resolve().parents[2]


def ensure_protocol(root):
    protocol={'version':1,'count':256,'seed':2026111800,'distribution':'N(0,I)',
              'data_manifest_sha256':file_hash(root/'results/art_search/data/manifest.json'),
              'primary_criteria_unchanged':True,'final_split_used':False,
              'purpose':'Additional validation Monte Carlo diagnostic, never a replacement for the primary 64 cases.'}
    path=root/'results/art_search/audit_protocol.json'
    if path.exists() and read_json(path)!=protocol:raise ConfigurationError('Cambió la auditoría registrada')
    path.write_text(json.dumps(protocol,indent=2))
    return protocol


def audit(root,name,seed=7):
    root=Path(root);assert_loss_unchanged(root)
    protocol=ensure_protocol(root)
    source=root/f'results/ej2_search_art_{name}/base__seed={seed}'
    read_json(source/'summary.json')
    config=read_json(source/'config.json');size=config['dataset']['image_size'];shape=(size,size,3)
    inputs={'model_sha256':file_hash(source/'model.npz'),'config_sha256':file_hash(source/'config.json'),
            'protocol':protocol,'code':{p:file_hash(root/p) for p in (
                'src/analysis/art_search_audit.py','src/evaluation/art_cleanliness.py',
                'src/models/factory.py','src/nn/spatial.py')}}
    output=root/f'results/art_search/validation_audit/{name}/seed={seed}'
    if (output/'summary.json').exists():
        result=read_json(output/'summary.json')
        if result['inputs']==inputs:return result
        revision=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
        output=output/f'revision_{revision}'
        if (output/'summary.json').exists():
            result=read_json(output/'summary.json')
            if result['inputs']!=inputs:raise ConfigurationError('Colisión de versiones de auditoría')
            return result
    model=load_source_model(root,{'run':str(source.relative_to(root))})
    coordinates=np.random.default_rng(protocol['seed']).standard_normal((protocol['count'],config['model']['latent_size']))
    generated=np.concatenate([model.decode(z.astype(model.precision)) for z in np.array_split(coordinates,4)])
    limits=read_json(root/'results/art_search/data/manifest.json')['cleanliness_limits']
    rows=[]
    for i,sample in enumerate(generated):
        row={'sample':i,**cleanliness(sample.reshape(shape))}
        row['clean']=clean_pass(row,limits)
        if config['model'].get('output_activation','sigmoid')=='identity':
            row['range_rmse']=float(np.sqrt(np.mean((sample-np.clip(sample,0,1))**2)))
            row['clean']=bool(row['clean'] and row['range_rmse']<=.005)
        rows.append(row)
    n=len(rows);count=sum(r['clean'] for r in rows);p=count/n;z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    result={'candidate':name,'seed':seed,'inputs':inputs,'count':n,'clean_count':count,'clean_rate':p,
            'output_directory':str(output.relative_to(root)),
            'conditional_wilson95':[float(center-half),float(center+half)],
            'interpretation':'Sampling interval conditional on this fitted model; ignores hyperparameter selection and does not certify semantic shape validity.',
            'all_cases_saved':True,'corrective_postprocessing':False}
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'cleanliness.csv',rows)
    np.savez_compressed(output/'generated.npz',samples=generated,coordinates=coordinates)
    for page in range(4):
        plotting.save_image_mosaic(generated[page*64:(page+1)*64],(8,8),shape,False,
                                  f'{name}, auditoría: {page*64}–{(page+1)*64-1}',output/f'page_{page}.png')
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates',nargs='+',choices=list(CANDIDATES))
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    for name in args.candidates:
        result=audit(ROOT,name,args.seed)
        print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2),flush=True)
