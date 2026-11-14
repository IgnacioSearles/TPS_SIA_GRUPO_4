"""Momentos posteriores de entrenamiento; diagnóstico, sin modificar pesos."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.art_capacity import read_json,file_hash
from src.data.art_search import assert_loss_unchanged,CANDIDATES
from src.data.factory import build_dataset
from src.evaluation.art_factorial import load_source_model
from src.evaluation.context import write_csv
from src.options import Options,ConfigurationError

ROOT=Path(__file__).resolve().parents[2]


def diagnose(root,name,seed=7):
    root=Path(root);assert_loss_unchanged(root)
    source=root/f'results/ej2_search_art_{name}/base__seed={seed}'
    config=read_json(source/'config.json')
    if config['model']['type']!='variational':raise ConfigurationError('El diagnóstico requiere un VAE')
    dataset=build_dataset(Options(config['dataset'],'dataset'),root)
    fingerprint={'model':file_hash(source/'model.npz'),'config':file_hash(source/'config.json'),
                 'training_samples':hashlib.sha256(dataset.samples.tobytes()).hexdigest(),
                 'diagnostic_code':file_hash(Path(__file__))}
    output=root/f'results/art_search/latent_diagnostics/{name}/seed={seed}'
    if (output/'summary.json').exists():
        old=read_json(output/'summary.json')
        if old['inputs']!=fingerprint:raise ConfigurationError('Cambió un diagnóstico registrado')
        return old
    model=load_source_model(root,{'run':str(source.relative_to(root))})
    pairs=[model.encode_distribution(dataset.samples[i:i+64].astype(model.precision))
           for i in range(0,len(dataset),64)]
    mean=np.concatenate([p[0] for p in pairs]).astype(np.float64)
    logvar=np.concatenate([p[1] for p in pairs]).astype(np.float64)
    variance=np.exp(logvar)
    if not np.isfinite(variance).all():raise ConfigurationError('Varianzas no finitas')
    mu=mean.mean(axis=0);between=mean.var(axis=0);within=variance.mean(axis=0)
    rows=[{'dimension':i,'mean_mu':float(a),'variance_mu':float(b),
           'mean_posterior_variance':float(c),'aggregate_variance':float(b+c)}
          for i,(a,b,c) in enumerate(zip(mu,between,within))]
    summary={'candidate':name,'seed':seed,'inputs':fingerprint,'training_count':len(dataset),
             'mean_posterior_variance':float(within.mean()),
             'aggregate_mean':mu.tolist(),'aggregate_variance':(between+within).tolist(),
             'mean_kl':float(np.mean(-.5*np.sum(1+logvar-mean**2-variance,axis=1))),
             'interpretation':'Training q(z|x) moments. Matching mean/variance to N(0,I) does not prove joint Gaussianity or good generation.',
             'weights_modified':False,'heldout_used':False}
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'dimensions.csv',rows)
    np.savez_compressed(output/'posterior.npz',mean=mean,log_variance=logvar)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates',nargs='+',choices=list(CANDIDATES))
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    for name in args.candidates:
        result=diagnose(ROOT,name,args.seed)
        print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2),flush=True)
