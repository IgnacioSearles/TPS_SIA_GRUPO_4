"""Diagnóstico de muestreo de un VAE fijo; no entrena ni cambia su prior.

Compara N(0,I), momentos del posterior agregado, mezcla de posteriores y
reconstrucción desde medias. Los últimos dos usan índices de entrenamiento:
son referencias diagnósticas, no pruebas de novedad ni generación desde el prior.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src import plotting
from src.data.art_capacity import file_hash,read_json
from src.data.art_search import assert_loss_unchanged
from src.evaluation.art_cleanliness import cleanliness,clean_pass
from src.evaluation.art_geometry import reconstruction_scores
from src.evaluation.context import write_csv
from src.evaluation.generative import nearest_training,pairwise_mse
from src.experiments.runner import load_run
from src.options import ConfigurationError

ROOT=Path(__file__).resolve().parents[2]
CANDIDATE='cnn4k_lowbeta_conv2_50'
METHODS=('prior','moment_gaussian','posterior_mixture','mean_replay')
LABELS=('Prior N(0,I)','Gaussiana de momentos','Mezcla de posteriores','Medias: reconstrucción')
PROTOCOL={'version':1,'count':256,'seeds':[7,17,27],
          'epsilon_seed':2026112200,'training_index_seed':2026112100,
          'display_indices':list(range(0,256,32)),
          'fit_data':'training only','reserved_final_used':False,
          'purpose':'Diagnosis of a fixed decoder, not replacement of the trained prior or a new acceptance test.'}


def aggregate_moments(means,log_variances):
    means=np.asarray(means,dtype=np.float64)
    logs=np.asarray(log_variances,dtype=np.float64)
    if (means.ndim!=2 or logs.shape!=means.shape or len(means)==0
            or not np.isfinite(means).all() or not np.isfinite(logs).all()):
        raise ConfigurationError('Se requieren medias y log-varianzas finitas N×D')
    with np.errstate(over='ignore'):
        variances=np.exp(logs)
    if not np.isfinite(variances).all():
        raise ConfigurationError('Varianza posterior no finita')
    center=means.mean(axis=0)
    offsets=means-center
    # Exact moments of the finite uniform mixture: denominator N, not N-1.
    covariance=offsets.T@offsets/len(means)+np.diag(variances.mean(axis=0))
    values,vectors=np.linalg.eigh(covariance)
    square_root=(vectors*np.sqrt(np.maximum(values,0)))@vectors.T
    return center,covariance,square_root


def sampling_coordinates(means,log_variances,epsilon,indices):
    means=np.asarray(means,dtype=np.float64)
    logs=np.asarray(log_variances,dtype=np.float64)
    center,covariance,square_root=aggregate_moments(means,logs)
    epsilon=np.asarray(epsilon,dtype=np.float64)
    indices=np.asarray(indices)
    if (epsilon.ndim!=2 or epsilon.shape[1]!=means.shape[1] or not np.isfinite(epsilon).all()
            or indices.shape!=(len(epsilon),) or not np.issubdtype(indices.dtype,np.integer)
            or np.any(indices<0) or np.any(indices>=len(means))):
        raise ConfigurationError('Coordenadas o índices de muestreo inválidos')
    coordinates={'prior':epsilon.copy(),
                 'moment_gaussian':epsilon@square_root.T+center,
                 'posterior_mixture':means[indices]+np.exp(.5*logs[indices])*epsilon,
                 'mean_replay':means[indices].copy()}
    return coordinates,(center,covariance,square_root)


def decode_batches(model,coordinates):
    return np.concatenate([model.decode(z.astype(model.precision)) for z in np.array_split(coordinates,4)])


def diagnose(root=ROOT):
    root=Path(root);assert_loss_unchanged(root)
    output=root/'results/art_sampling_diagnostic'
    output.mkdir(parents=True,exist_ok=True)
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and read_json(protocol_path)!=PROTOCOL:
        raise ConfigurationError('Cambió el protocolo del diagnóstico')
    protocol_path.write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    limits=read_json(root/'results/art_search/data/manifest.json')['cleanliness_limits']
    code={p:file_hash(root/p) for p in ('src/analysis/art_sampling_control.py','src/data/art.py',
         'src/nn/spatial.py','src/models/factory.py','src/evaluation/art_cleanliness.py',
         'src/evaluation/art_geometry.py')}
    summaries=[]
    for seed in PROTOCOL['seeds']:
        source=root/f'results/ej2_search_art_{CANDIDATE}/base__seed={seed}'
        read_json(source/'summary.json')
        model,data=load_run(source)
        inputs={'model_sha256':file_hash(source/'model.npz'),
                'config_sha256':file_hash(source/'config.json'),
                'training_samples_sha256':hashlib.sha256(data.samples.tobytes()).hexdigest(),
                'protocol':PROTOCOL,'code':code}
        with np.load(source/'checkpoint.npz',allow_pickle=False) as checkpoint:
            training_state=json.loads(str(checkpoint['metadata']))
        if training_state['signature']['samples']!=inputs['training_samples_sha256']:
            raise ConfigurationError('Los datos actuales difieren de los usados al entrenar')
        directory=output/f'seed={seed}'
        if (directory/'summary.json').exists():
            saved=read_json(directory/'summary.json')
            if saved['inputs']!=inputs:raise ConfigurationError('Cambió un diagnóstico terminado')
            summaries.extend(saved['methods']);continue
        directory.mkdir(exist_ok=True)
        pairs=[model.encode_distribution(data.samples[i:i+64].astype(model.precision))
               for i in range(0,len(data.samples),64)]
        means=np.concatenate([p[0] for p in pairs]);logs=np.concatenate([p[1] for p in pairs])
        epsilon=np.random.default_rng(PROTOCOL['epsilon_seed']).standard_normal((PROTOCOL['count'],model.latent_size))
        indices=np.random.default_rng(PROTOCOL['training_index_seed']).integers(len(data.samples),size=len(epsilon))
        coordinates,moments=sampling_coordinates(means,logs,epsilon,indices)
        np.savez_compressed(directory/'fitted_distribution.npz',mean=moments[0],covariance=moments[1],
                            square_root=moments[2],posterior_means=means,posterior_log_variances=logs,
                            source_indices=indices,epsilon=epsilon)
        print(f'Diagnóstico, semilla {seed}: posteriores de {len(means)} imágenes',flush=True)
        rows=[];selected=[]
        for method,label in zip(METHODS,LABELS):
            generated=decode_batches(model,coordinates[method])
            target=directory/method;target.mkdir(exist_ok=True)
            nearest,nearest_indices=nearest_training(generated,data.samples)
            cases=[]
            for i,image in enumerate(generated):
                item={'sample':i,**cleanliness(image.reshape(data.image_shape))}
                item['clean']=clean_pass(item,limits)
                item.update(nearest_train_mse=float(nearest[i]),nearest_train_label=data.labels[nearest_indices[i]])
                if method in ('posterior_mixture','mean_replay'):
                    index=int(indices[i]);record=data.generative_study.train_records[index]
                    scores=reconstruction_scores(data.samples[index].reshape(data.image_shape),
                                                 image.reshape(data.image_shape),record)
                    item.update(source_index=index,source_label=data.labels[index],
                                source_family=record['family'],reference_iou=scores['silhouette_iou'])
                cases.append(item)
            row={'seed':seed,'method':method,'count':len(cases),'clean_count':sum(r['clean'] for r in cases),
                 'clean_rate':float(np.mean([r['clean'] for r in cases])),
                 'nearest_train_mse':float(nearest.mean()),'pairwise_mse':pairwise_mse(generated)}
            if method in ('posterior_mixture','mean_replay'):
                row['reference_iou']=float(np.mean([r['reference_iou'] for r in cases]))
            rows.append(row)
            write_csv(target/'cases.csv',cases)
            np.savez_compressed(target/'generated.npz',samples=generated,coordinates=coordinates[method])
            for page in range(4):
                plotting.save_image_mosaic(generated[page*64:(page+1)*64],(8,8),data.image_shape,
                                          False,f'{label}; semilla {seed}; casos {page*64}–{page*64+63}',
                                          target/f'page_{page}.png')
            selected.append(generated[PROTOCOL['display_indices']])
            print(json.dumps(row),flush=True)
        fig,axes=plt.subplots(4,8,figsize=(13,8.2))
        for row_index,(images,label) in enumerate(zip(selected,LABELS)):
            for column,(image,index) in enumerate(zip(images,PROTOCOL['display_indices'])):
                axis=axes[row_index,column]
                axis.imshow(image.reshape(data.image_shape),interpolation='nearest')
                axis.set_xticks([]);axis.set_yticks([])
                if row_index==0:axis.set_title(str(index))
                if column==0:axis.set_ylabel(label,fontsize=10)
        fig.suptitle(f'Mismo decoder, semilla {seed}; índices prefijados, sin filtros')
        fig.tight_layout();fig.savefig(directory/'comparison.png',dpi=150,bbox_inches='tight');plt.close(fig)
        if file_hash(source/'model.npz')!=inputs['model_sha256']:
            raise ConfigurationError('Los pesos fuente cambiaron')
        result={'inputs':inputs,'seed':seed,'methods':rows,
                'weights_modified':False,'training_performed':False,
                'interpretation':'Prior is the canonical generator. Other samplers are training-fitted diagnostics; '
                                 'mean replay is reconstruction, not new generation. No novelty or heldout claim.'}
        (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        summaries.extend(rows)
    write_csv(output/'methods.csv',summaries)
    (output/'summary.json').write_text(json.dumps({'protocol':PROTOCOL,'methods':summaries},indent=2)+'\n')
    return summaries


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    diagnose()
