"""Prior estándar vs. gaussiana ajustada, para AE y VAE, sobre las mismas notas NSynth.

No mide generalización ni calidad perceptual: los datos son los de entrenamiento
del protocolo original. No cambia pesos, pérdida ni el códec.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src import plotting
from src.analysis.latent_sampling import fit_gaussian
from src.data.art_capacity import file_hash
from src.data.audio import write_waveform
from src.evaluation.context import write_csv
from src.evaluation.generative import nearest_training, pairwise_mse
from src.experiments.runner import load_run
from src.options import ConfigurationError

ROOT=Path(__file__).resolve().parents[2]
METHODS=('ae_standard','ae_fitted','vae_standard','vae_fitted')
LABELS=('AE: N(0,I)','AE: gaussiana ajustada','VAE: N(0,I)','VAE: gaussiana ajustada')


def validate_pair(configs,datasets):
    """Solo la parte variacional y la evaluación pueden variar entre fuentes."""
    if [config['model']['type'] for config in configs]!=['autoencoder','variational']:
        raise ConfigurationError('Las fuentes deben ser AE y VAE en ese orden')
    comparable=[]
    for config in configs:
        item=copy.deepcopy(config)
        item.pop('evaluation',None)
        item['model'].pop('type',None)
        item['model'].pop('kl',None)
        comparable.append(item)
    if comparable[0]!=comparable[1]:
        raise ConfigurationError('AE/VAE difieren en datos, arquitectura o entrenamiento')
    if (list(datasets[0].labels)!=list(datasets[1].labels)
            or not np.array_equal(datasets[0].samples,datasets[1].samples)):
        raise ConfigurationError('AE/VAE no recibieron las mismas notas en el mismo orden')


def compare(root=ROOT):
    root=Path(root)
    sources=[root/f'results/ej2_nsynth_sampling_control_{name}/base__seed=7' for name in ('ae','vae')]
    configs=[json.loads((p/'config.json').read_text()) for p in sources]
    for p in sources:json.loads((p/'summary.json').read_text())
    loaded=[load_run(p) for p in sources]
    models,datasets=zip(*loaded)
    validate_pair(configs,datasets)
    ae,vae=models;data=datasets[0]
    fingerprint=hashlib.sha256(data.samples.tobytes()).hexdigest()
    for source in sources:
        with np.load(source/'checkpoint.npz',allow_pickle=False) as checkpoint:
            metadata=json.loads(str(checkpoint['metadata']))
        if metadata['signature']['samples']!=fingerprint:
            raise ConfigurationError('Las notas actuales difieren de las usadas al entrenar')
    inputs={'version':2,'count':256,'generation_seed':2026111900,'preview_indices':list(range(6)),
            'training_samples_sha256':fingerprint,
            'model_seed':7,'covariance':'full sample N-1, fitted to the training codes of each model (VAE: posterior means)',
            'sources':[{'run':str(p.relative_to(root)),
                        'model_sha256':file_hash(p/'model.npz'),
                        'config_sha256':file_hash(p/'config.json')} for p in sources],
            'code':{p:file_hash(root/p) for p in ('src/analysis/nsynth_sampling.py',
                    'src/analysis/latent_sampling.py','src/data/audio.py')}}
    output=root/'results/nsynth_sampling_control'
    if (output/'manifest.json').exists():
        old=json.loads((output/'manifest.json').read_text())
        if old['inputs']!=inputs:raise ConfigurationError('Cambió un control de audio ya terminado')
        return old
    if data.audio_codec is None:raise ConfigurationError('El control requiere audio')
    epsilon=np.random.default_rng(inputs['generation_seed']).standard_normal((inputs['count'],ae.latent_size))
    output.mkdir(parents=True,exist_ok=True)
    coordinates,samples=[],[]
    for name,model in (('ae',ae),('vae',vae)):
        mean,covariance,square_root=fit_gaussian(model.encode(data.samples.astype(model.precision)))
        np.savez_compressed(output/f'fitted_gaussian_{name}.npz',mean=mean,covariance=covariance,
                            square_root=square_root,epsilon=epsilon)
        for z in (epsilon,epsilon@square_root.T+mean):
            coordinates.append(z.astype(model.precision))
            samples.append(model.decode(coordinates[-1]))
    rows=[]
    shape=data.image_shape
    figure,axes=plt.subplots(len(METHODS),6,figsize=(13,9))
    for method_index,(method,label,generated,z) in enumerate(zip(METHODS,LABELS,samples,coordinates)):
        directory=output/method;directory.mkdir(exist_ok=True)
        distances,indices=nearest_training(generated,data.samples)
        rows.append({'method':method,'count':len(generated),
                     'nearest_train_mse':float(distances.mean()),
                     'pairwise_mse':pairwise_mse(generated),
                     'mean_value':float(generated.mean())})
        write_csv(directory/'samples.csv',[{'sample':i,'nearest_train_mse':float(d),
                  'nearest_train_label':data.labels[j]} for i,(d,j) in enumerate(zip(distances,indices))])
        np.savez_compressed(directory/'generated.npz',samples=generated,coordinates=z)
        for page in range(4):
            plotting.save_image_mosaic(generated[page*64:(page+1)*64],(8,8),shape,False,
                                      f'{label}, casos {page*64}–{page*64+63}',directory/f'page_{page}.png')
        for column in range(6):
            axes[method_index,column].imshow(generated[column].reshape(shape),origin='lower',
                                             aspect='auto',cmap='magma',vmin=0,vmax=1)
            axes[method_index,column].set_xticks([]);axes[method_index,column].set_yticks([])
            if column==0:axes[method_index,column].set_ylabel(label)
            if method_index==0:axes[method_index,column].set_title(f'Índice {column}')
            for name,vector in [('generated',generated[column]),('nearest_codec',data.samples[indices[column]])]:
                waveform=data.audio_codec.decode(vector.reshape(shape))
                write_waveform(directory/f'{name}_{column}.wav',waveform,data.audio_codec.sample_rate)
        print(json.dumps(rows[-1]),flush=True)
    figure.suptitle('Mismos vectores normales base; seis primeros índices, sin seleccionar por calidad')
    figure.tight_layout();figure.savefig(output/'comparison.png',dpi=150);plt.close(figure)
    write_csv(output/'summary.csv',rows)
    for p,item in zip(sources,inputs['sources']):
        if file_hash(p/'model.npz')!=item['model_sha256']:
            raise ConfigurationError('Los pesos cambiaron durante la evaluación')
    result={'status':'completed','inputs':inputs,'rows':rows,'training_during_evaluation':False,
            'train_count':len(data.samples),'independent_model_seeds':1,
            'codec':data.audio_codec.encoding_parameters(),
            'interpretation':'Distances/diversity are descriptive, not perceptual quality, novelty or generalization. '
                             'One model seed; a fitted Gaussian can cross low-density latent regions. '
                             'Both models get the same fitted-Gaussian treatment, so the comparison is symmetric. '
                             'New reproductions, not the missing original presentation weights.'}
    (output/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(compare(),indent=2,ensure_ascii=False))
