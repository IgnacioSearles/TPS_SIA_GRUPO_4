"""Densidades pequeñas sobre códigos del VAE fijo: comparación exploratoria.

Las mezclas cambian la distribución usada al generar. No cambian el prior
del entrenamiento ni constituyen aprobación del experimento VAE estándar.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src import plotting
from src.analysis.gaussian_mixture import fit_mixture,sample_mixture
from src.analysis.art_sampling_control import decode_batches
from src.data.art_capacity import file_hash,read_json
from src.data.art_search import assert_loss_unchanged
from src.evaluation.art_cleanliness import cleanliness,clean_pass
from src.evaluation.context import write_csv
from src.evaluation.generative import nearest_training,pairwise_mse
from src.experiments.runner import load_run

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL={'version':1,'components':[1,4,8],'model_seeds':[7,17,27],
          'count':256,'posterior_draw_seed':2026112300,'fit_seed':2026112500,
          'epsilon_seed':2026112200,'component_uniform_seed':2026112400,
          'restarts':3,'iterations':100,'ridge':1e-5,'tolerance':1e-5,
          'fit_data':'one posterior sample per training image; no labels',
          'restart_selection':'training likelihood only','reserved_final_used':False,
          'display_indices':list(range(0,256,32))}


def compare(root=ROOT):
    root=Path(root);assert_loss_unchanged(root)
    output=root/'results/art_latent_mixture';output.mkdir(parents=True,exist_ok=True)
    path=output/'protocol.json'
    if path.exists():assert read_json(path)==PROTOCOL,'Cambió el protocolo'
    path.write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    limits=read_json(root/'results/art_search/data/manifest.json')['cleanliness_limits']
    rows=[]
    for seed in PROTOCOL['model_seeds']:
        source=root/f'results/ej2_search_art_cnn4k_lowbeta_conv2_50/base__seed={seed}'
        previous=root/f'results/art_sampling_diagnostic/seed={seed}'
        posterior=previous/'fitted_distribution.npz'
        old=read_json(previous/'summary.json')
        assert old['inputs']['model_sha256']==file_hash(source/'model.npz')
        inputs={'model_sha256':file_hash(source/'model.npz'),
                'config_sha256':file_hash(source/'config.json'),
                'posterior_sha256':file_hash(posterior),'protocol':PROTOCOL,
                'code':{p:file_hash(root/p) for p in ('src/analysis/art_latent_mixture.py',
                        'src/analysis/gaussian_mixture.py','src/nn/spatial.py','src/models/factory.py',
                        'src/evaluation/art_cleanliness.py')}}
        directory=output/f'seed={seed}'
        if (directory/'summary.json').exists():
            cached=read_json(directory/'summary.json');assert cached['inputs']==inputs
            rows.extend(cached['methods']);continue
        directory.mkdir(exist_ok=True)
        model,data=load_run(source)
        with np.load(posterior) as saved:
            means=saved['posterior_means'].copy();logs=saved['posterior_log_variances'].copy()
        assert means.shape==(len(data.samples),model.latent_size)
        check_mean,check_logs=model.encode_distribution(data.samples[:64].astype(model.precision))
        np.testing.assert_array_equal(check_mean,means[:64]);np.testing.assert_array_equal(check_logs,logs[:64])
        training_noise=np.random.default_rng(PROTOCOL['posterior_draw_seed']).standard_normal(means.shape)
        fitted_samples=means.astype(float)+np.exp(.5*logs.astype(float))*training_noise
        epsilon=np.random.default_rng(PROTOCOL['epsilon_seed']).standard_normal((PROTOCOL['count'],model.latent_size))
        uniforms=np.random.default_rng(PROTOCOL['component_uniform_seed']).random(len(epsilon))
        selected=[];seed_rows=[]
        for components in PROTOCOL['components']:
            fitted=fit_mixture(fitted_samples,components,seed=PROTOCOL['fit_seed'],
                 restarts=PROTOCOL['restarts'],max_iterations=PROTOCOL['iterations'],
                 tolerance=PROTOCOL['tolerance'],ridge=PROTOCOL['ridge'])
            coordinates,chosen=sample_mixture(fitted,epsilon,uniforms)
            generated=decode_batches(model,coordinates)
            target=directory/f'components={components}';target.mkdir(exist_ok=True)
            nearest,indices=nearest_training(generated,data.samples)
            cases=[]
            for i,sample in enumerate(generated):
                case={'sample':i,'component':int(chosen[i]),**cleanliness(sample.reshape(data.image_shape))}
                case.update(clean=clean_pass(case,limits),nearest_train_mse=float(nearest[i]),
                            nearest_train_label=data.labels[indices[i]])
                cases.append(case)
            row={'seed':seed,'components':components,'count':len(cases),
                 'clean_count':sum(c['clean'] for c in cases),
                 'nearest_train_mse':float(nearest.mean()),'pairwise_mse':pairwise_mse(generated),
                 'train_log_likelihood':fitted['train_log_likelihood'],'fit_iterations':fitted['iterations']}
            seed_rows.append(row);rows.append(row)
            write_csv(target/'cases.csv',cases)
            np.savez_compressed(target/'density.npz',weights=fitted['weights'],means=fitted['means'],
                                covariances=fitted['covariances'])
            np.savez_compressed(target/'generated.npz',samples=generated,coordinates=coordinates,epsilon=epsilon,uniforms=uniforms)
            for page in range(4):
                plotting.save_image_mosaic(generated[page*64:(page+1)*64],(8,8),data.image_shape,False,
                    f'Mezcla {components} componentes, semilla {seed}; casos {64*page}–{64*page+63}',target/f'page_{page}.png')
            selected.append(generated[PROTOCOL['display_indices']])
            print(json.dumps(row),flush=True)
        with np.load(previous/'prior/generated.npz') as saved:
            prior=saved['samples'][PROTOCOL['display_indices']].copy()
            np.testing.assert_array_equal(saved['coordinates'],epsilon)
        figure,axes=plt.subplots(4,8,figsize=(14,8.6))
        for row,(images,label) in enumerate(zip([prior,*selected],['Prior\nN(0,I)','GMM\n1 componente','GMM\n4 componentes','GMM\n8 componentes'])):
            for column,(image,index) in enumerate(zip(images,PROTOCOL['display_indices'])):
                axis=axes[row,column];axis.imshow(image.reshape(data.image_shape),interpolation='nearest')
                axis.set_xticks([]);axis.set_yticks([])
                if row==0:axis.set_title(str(index))
            axes[row,0].text(-.15,.5,label,ha='right',va='center',transform=axes[row,0].transAxes)
        figure.suptitle(f'Decoder fijo, semilla {seed}; densidades ajustadas sin etiquetas')
        figure.text(.57,.025,'Índices prefijados; sin correspondencia semántica entre filas.\n'
                    'Las GMM son distribuciones de generación distintas del prior entrenado.',ha='center',fontsize=10)
        figure.subplots_adjust(left=.16,right=.99,top=.92,bottom=.09,hspace=.15,wspace=.10)
        figure.savefig(directory/'comparison.png',dpi=140);plt.close(figure)
        assert file_hash(source/'model.npz')==inputs['model_sha256']
        result={'inputs':inputs,'methods':seed_rows,'training_neural_network':False,
                'interpretation':'Post-training latent density fitting. Not canonical-prior generation or proof of novelty.'}
        (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv(output/'methods.csv',rows)
    (output/'summary.json').write_text(json.dumps({'protocol':PROTOCOL,'methods':rows},indent=2)+'\n')
    return rows


if __name__=='__main__':compare()
