"""Held-out NSynth reconstruction, prior generation and mel-denoising evaluation.

python -m src.analysis.nsynth_cnn_evaluation --jobs 2
No training, loss changes, final pitch-reference access, or slide changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.art_search import assert_loss_unchanged
from src.data.audio import write_waveform
from src.data.noise import Noise
from src.evaluation.context import write_csv
from src.experiments.runner import load_run

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path('reports/nsynth_cnn_study')
CONDITIONS=('vae_mlp','vae_conv','dae_mlp','dae_conv')
SEEDS=(7,17,27)
NOISE_LEVELS=(0.,.05,.1,.2)
GENERATION_COUNT=64
PROTOCOL='nsynth_cnn_evaluation_v1'


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def corruption(samples,seed,level):
    """Same noise in both architectures, independent of model evaluation RNG."""
    rng=np.random.default_rng(np.random.SeedSequence([seed,20261115,2]))
    return Noise('gaussian',level).apply(samples,rng).astype(np.float32)


def generation_codes(seed,latent_size,count=GENERATION_COUNT):
    rng=np.random.default_rng(np.random.SeedSequence([seed,20261115,1]))
    return rng.standard_normal((count,latent_size)).astype(np.float32)


def reconstruct_batches(model,samples,batch_size=32):
    return np.concatenate([model.reconstruct(samples[i:i+batch_size].astype(model.precision))
                           for i in range(0,len(samples),batch_size)])


def nearest_distances(samples,reference,batch_size=16):
    """Exact feature-normalized MSE, bounded query batch; no sklearn dependency."""
    reference=np.asarray(reference,dtype=np.float64)
    reference_norm=np.sum(reference**2,axis=1)
    distances=[];indices=[]
    for start in range(0,len(samples),batch_size):
        query=np.asarray(samples[start:start+batch_size],dtype=np.float64)
        matrix=(np.sum(query**2,axis=1)[:,None]+reference_norm[None,:]-2*query@reference.T)/reference.shape[1]
        np.maximum(matrix,0,out=matrix)
        nearest=np.argmin(matrix,axis=1)
        distances.extend(matrix[np.arange(len(query)),nearest])
        indices.extend(nearest)
    return np.asarray(distances),np.asarray(indices,dtype=int)


def mean_pairwise_mse(samples):
    samples=np.asarray(samples,dtype=np.float64)
    if len(samples)<2:return 0.
    # Identity: sum ordered pair distances = 2N sum ||x-mean||².
    return float(2*np.sum((samples-samples.mean(axis=0))**2)/((len(samples)-1)*samples.shape[1]))


def per_note_rows(condition,seed,dataset,prediction,task,noise_level=0.,method='model'):
    clean=dataset.samples
    valid=np.all(np.isfinite(prediction),axis=1)
    errors=np.mean((np.asarray(prediction,dtype=float)-clean)**2,axis=1)
    return [{'condition':condition,'seed':seed,'task':task,'method':method,'label':label,
             'family':family,'noise_level':noise_level,'valid':bool(valid[i]),
             'mse':float(errors[i]) if valid[i] else float('inf')}
            for i,(label,family) in enumerate(zip(dataset.labels,dataset.groups))]


def example_labels(manifest,dataset):
    chosen=manifest.get('eval_example_labels')
    if chosen:return chosen
    chosen=[]
    for family in sorted(set(dataset.groups)):
        chosen.append(sorted(label for label,group in zip(dataset.labels,dataset.groups) if group==family)[0])
    return chosen[:6]


def save_audio(codec,path,spec):
    path.parent.mkdir(parents=True,exist_ok=True)
    write_waveform(path,codec.decode(spec.reshape(codec.image_shape)),codec.sample_rate)


def evaluate_run(task):
    root,condition,seed=task
    root=Path(root)
    from src.data.nsynth_cnn_study import build_study_data
    directory=root/f'results/ej2_nsynth_cnn_{condition}/base__seed={seed}'
    if not (directory/'summary.json').exists():
        return {'condition':condition,'seed':seed,'status':'missing'}
    study=build_study_data(root)
    train,test,manifest=study['train'],study['test'],study['manifest']
    weights_before=digest((directory/'model.npz').read_bytes())
    data_hash=digest(json.dumps(manifest,sort_keys=True).encode())
    config=read(directory/'config.json')
    if config['dataset']!=study['train_settings']:
        raise ValueError(f'Unexpected training selection in {directory}')
    model,used_dataset=load_run(directory)
    if used_dataset.labels!=train.labels or set(used_dataset.labels).intersection(test.labels):
        raise ValueError('Held-out leakage or incompatible training order')
    is_vae=condition.startswith('vae_')
    if config['model']['type']!=('variational' if is_vae else 'autoencoder'):
        raise ValueError('Unexpected model type')
    output=root/OUTPUT/f'{condition}/seed={seed}'
    output.mkdir(parents=True,exist_ok=True)
    signature={'protocol':PROTOCOL,'weights_sha256':weights_before,'data_sha256':data_hash,
               'noise_levels':NOISE_LEVELS,'generation_count':GENERATION_COUNT,'seed':seed}
    signature_path=output/'signature.json'
    completed=output/'summary.json'
    if completed.exists() and signature_path.exists() and read(signature_path)==json.loads(json.dumps(signature)):
        return read(completed)
    references=example_labels(manifest,test)
    examples=[test.index_of(label) for label in references]
    codec=test.audio_codec
    originals_path=root/'results/nsynth_cnn_study/data/example_originals.npz'
    if not originals_path.exists():
        raise FileNotFoundError('Missing original-waveform references from the study data preparation')
    with np.load(originals_path,allow_pickle=False) as stored:
        originals={str(label):waveform for label,waveform in zip(stored['labels'],stored['waveforms'])}
    for index,label in zip(examples,references):
        if label not in originals:raise ValueError('Missing fixed original example')
        original_path=output/f'audio/references/{label}_original.wav'
        original_path.parent.mkdir(parents=True,exist_ok=True)
        write_waveform(original_path,originals[label],codec.sample_rate)
        save_audio(codec,output/f'audio/references/{label}_codec.wav',test.samples[index])
    rows=[];extra={};predictions={}
    if is_vae:
        decoded=reconstruct_batches(model,test.samples)
        rows=per_note_rows(condition,seed,test,decoded,'reconstruction')
        predictions['reconstructed_examples']=decoded[examples]
        for index,label in zip(examples,references):
            save_audio(codec,output/f'audio/reconstruction/{label}.wav',decoded[index])
        codes=generation_codes(seed,model.latent_size)
        generated=model.decode(codes.astype(model.precision))
        if not np.all(np.isfinite(generated)):
            raise ValueError('Non-finite generated spectrogram: record training failure before continuing')
        train_distance,train_neighbor=nearest_distances(generated,train.samples)
        test_distance,test_neighbor=nearest_distances(generated,test.samples)
        generation_rows=[{'condition':condition,'seed':seed,'sample':i,'train_mse':float(train_distance[i]),
                          'test_mse':float(test_distance[i]),'nearest_train_label':train.labels[train_neighbor[i]],
                          'nearest_test_label':test.labels[test_neighbor[i]],
                          'nearest_train_family':train.groups[train_neighbor[i]],
                          'nearest_test_family':test.groups[test_neighbor[i]]} for i in range(len(generated))]
        write_csv(output/'generation.csv',generation_rows)
        np.savez_compressed(output/'generated.npz',samples=generated,coordinates=codes)
        extra={'generation_count':len(generated),'generation_train_mse':float(np.mean(train_distance)),
               'generation_test_mse':float(np.mean(test_distance)),
               'generation_pairwise_mse':mean_pairwise_mse(generated),
               'generation_distinct_train_neighbors':len(set(train_neighbor)),
               'generation_distinct_test_neighbors':len(set(test_neighbor)),
               'generation_codes_sha256':digest(codes.tobytes())}
        for index in range(6):
            save_audio(codec,output/f'audio/generation/sample_{index:02d}.wav',generated[index])
    else:
        noise_hashes={}
        for level in NOISE_LEVELS:
            noisy=corruption(test.samples,seed,level)
            noise_hashes[str(level)]=digest(noisy.tobytes())
            decoded=reconstruct_batches(model,noisy)
            rows.extend(per_note_rows(condition,seed,test,decoded,'denoising',level))
            rows.extend(per_note_rows(condition,seed,test,noisy,'denoising',level,'input'))
            if level==.1:
                predictions.update(noisy_examples=noisy[examples],denoised_examples=decoded[examples])
                for index,label in zip(examples,references):
                    save_audio(codec,output/f'audio/denoised/{label}.wav',decoded[index])
                    save_audio(codec,output/f'audio/noisy/{label}.wav',noisy[index])
        extra={'corrupted_inputs_sha256':noise_hashes}
    np.savez_compressed(output/'examples.npz',labels=np.asarray(references),clean_examples=test.samples[examples],**predictions)
    write_csv(output/'per_note.csv',rows)
    assert weights_before==digest((directory/'model.npz').read_bytes()),'Weights changed during evaluation'
    record={'condition':condition,'seed':seed,'status':'complete','train_count':len(train),'test_count':len(test),
            'model_sha256':weights_before,'data_sha256':data_hash,**extra}
    signature_path.write_text(json.dumps(signature,indent=2))
    completed.write_text(json.dumps(record,indent=2))
    return record


def aggregate_note_rows(rows):
    groups=defaultdict(list)
    for row in rows:
        key=(row['condition'],row['task'],row['method'],float(row['noise_level']))
        groups[key].append(row)
    return [{'condition':condition,'task':task,'method':method,'noise_level':level,'count':len(group),
             'invalid':sum(str(r['valid'])!='True' for r in group),'mse':float(np.mean([float(r['mse']) for r in group]))}
            for (condition,task,method,level),group in sorted(groups.items())]


def create_example_figures(root,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Fixed seed and first six manifest-selected examples, never best cases.
    for task,left,right,keys in [('reconstruction','vae_mlp','vae_conv',('clean_examples','reconstructed_examples','reconstructed_examples')),
                                ('denoising','dae_mlp','dae_conv',('noisy_examples','denoised_examples','denoised_examples'))]:
        paths=[root/OUTPUT/f'{condition}/seed=7/examples.npz' for condition in (left,right)]
        if not all(p.exists() for p in paths):continue
        with np.load(paths[0],allow_pickle=False) as a,np.load(paths[1],allow_pickle=False) as b:
            assert np.array_equal(a['labels'],b['labels'])
            values=(a[keys[0]],a[keys[1]],b[keys[2]])
            labels=a['labels']
            fig,axes=plt.subplots(len(labels),3,figsize=(12,2.2*len(labels)),squeeze=False,constrained_layout=True)
            for row,label in enumerate(labels):
                for col,(data,title) in enumerate(zip(values,('Referencia mel' if task=='reconstruction' else 'Entrada ruidosa','Denso','CNN'))):
                    axes[row,col].imshow(data[row].reshape(128,64),origin='lower',aspect='auto',vmin=0,vmax=1,cmap='magma')
                    axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                    if row==0:axes[row,col].set_title(title)
                axes[row,0].set_ylabel(str(label),fontsize=8)
            fig.savefig(root/OUTPUT/f'{task}_examples.png',dpi=130);plt.close(fig)
    paths=[root/OUTPUT/f'{condition}/seed=7/generated.npz' for condition in ('vae_mlp','vae_conv')]
    if all(p.exists() for p in paths):
        with np.load(paths[0],allow_pickle=False) as a,np.load(paths[1],allow_pickle=False) as b:
            assert np.array_equal(a['coordinates'],b['coordinates'])
            fig,axes=plt.subplots(2,6,figsize=(15,5),constrained_layout=True)
            for row,(data,name) in enumerate(((a['samples'],'Denso'),(b['samples'],'CNN'))):
                for col in range(6):
                    axes[row,col].imshow(data[col].reshape(128,64),origin='lower',aspect='auto',vmin=0,vmax=1,cmap='magma')
                    axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                    if row==0:axes[row,col].set_title(f'Código {col}')
                axes[row,0].set_ylabel(name)
            fig.savefig(root/OUTPUT/'generation_examples.png',dpi=130);plt.close(fig)


def plot_metrics(root,aggregates,per_seed,generation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    clean=[r for r in aggregates if r['task']=='reconstruction']
    if clean:
        fig,ax=plt.subplots(figsize=(6.5,4),constrained_layout=True)
        bars=ax.bar([r['condition'] for r in clean],[r['mse'] for r in clean],color=['#176a98','#d4602a'][:len(clean)])
        ax.set_ylabel('MSE en instrumentos reservados');ax.set_ylim(bottom=0)
        ax.set_title('Reconstrucción NSynth, todas las notas de prueba')
        for bar,row in zip(bars,clean):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{row["mse"]:.5f}',ha='center',va='bottom')
        fig.savefig(root/OUTPUT/'reconstruction_mse.png',dpi=150);plt.close(fig)
    noisy=[r for r in aggregates if r['task']=='denoising']
    if noisy:
        fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
        for condition,color in [('dae_mlp','#176a98'),('dae_conv','#d4602a')]:
            values=sorted([r for r in noisy if r['condition']==condition and r['method']=='model'],key=lambda r:r['noise_level'])
            if values:ax.plot([r['noise_level'] for r in values],[r['mse'] for r in values],marker='o',label=condition,color=color)
        baseline=next((condition for condition in ('dae_mlp','dae_conv') if any(r['condition']==condition for r in noisy)),None)
        values=sorted([r for r in noisy if r['condition']==baseline and r['method']=='input'],key=lambda r:r['noise_level'])
        ax.plot([r['noise_level'] for r in values],[r['mse'] for r in values],marker='o',label='Entrada sin limpiar',color='#777777',linestyle='--')
        ax.axvline(.1,color='#999999',linestyle=':',label='Ruido de entrenamiento')
        ax.set_xlabel('Desvío del ruido gaussiano en mel');ax.set_ylabel('MSE respecto de la referencia limpia')
        ax.set_title('Denoising NSynth, todas las notas de prueba');ax.set_ylim(bottom=0);ax.legend()
        fig.savefig(root/OUTPUT/'denoising_mse.png',dpi=150);plt.close(fig)
    if per_seed:
        fig,axes=plt.subplots(1,4,figsize=(16,4.3),constrained_layout=True)
        settings=[('reconstruction',0.,'vae','Reconstrucción limpia'),
                  ('denoising',.05,'dae','Denoising σ0.05'),('denoising',.1,'dae','Denoising σ0.1'),
                  ('denoising',.2,'dae','Denoising σ0.2')]
        for ax,(task,level,prefix,title) in zip(axes,settings):
            for seed,color in zip(SEEDS,('#176a98','#d4602a','#4d8b5a')):
                rows=[r for r in per_seed if r['seed']==seed and r['task']==task and r['noise_level']==level and r['method']=='model']
                positions=[];values=[]
                for position,arch in enumerate(('mlp','conv')):
                    match=next((r for r in rows if r['condition']==f'{prefix}_{arch}'),None)
                    if match is not None:positions.append(position);values.append(match['mse'])
                if values:ax.plot(positions,values,marker='o',color=color,label=f'Semilla {seed}')
                if task=='denoising':
                    baseline=next((r for r in per_seed if r['seed']==seed and r['task']==task and r['noise_level']==level and r['method']=='input'),None)
                    if baseline is not None:ax.axhline(baseline['mse'],color=color,linestyle=':',alpha=.5)
            ax.set_xticks([0,1],['Denso','CNN']);ax.set_xlim(-.2,1.2);ax.set_ylim(bottom=0);ax.set_title(title)
            ax.set_ylabel('MSE en instrumentos reservados')
            handles,_=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=8)
        fig.suptitle('Comparación por semilla. Líneas punteadas: entrada ruidosa sin limpiar')
        fig.savefig(root/OUTPUT/'paired_by_seed.png',dpi=150);plt.close(fig)
    if generation:
        fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
        for ax,(field,title) in zip(axes,[('generation_train_mse','MSE al vecino de entrenamiento'),
                                        ('generation_test_mse','MSE al vecino de prueba'),
                                        ('generation_pairwise_mse','MSE media entre pares generados')]):
            for seed,color in zip(SEEDS,('#176a98','#d4602a','#4d8b5a')):
                positions=[];values=[]
                for position,arch in enumerate(('mlp','conv')):
                    match=next((r for r in generation if r['seed']==seed and r['condition']==f'vae_{arch}'),None)
                    if match is not None:positions.append(position);values.append(match[field])
                if values:ax.plot(positions,values,marker='o',color=color,label=f'Semilla {seed}')
            ax.set_xticks([0,1],['Denso','CNN']);ax.set_xlim(-.2,1.2);ax.set_ylim(bottom=0);ax.set_title(title,fontsize=11)
            handles,_=ax.get_legend_handles_labels()
            if handles:ax.legend(fontsize=8)
        fig.suptitle('Generación libre: descripciones de distancia y diversidad, sin certificado de calidad o novedad')
        fig.savefig(root/OUTPUT/'generation_metrics.png',dpi=150);plt.close(fig)


def report(root=ROOT,results=None):
    import csv
    root=Path(root);output=root/OUTPUT;output.mkdir(parents=True,exist_ok=True)
    if results is None:
        results=[]
        for condition in CONDITIONS:
            for seed in SEEDS:
                path=output/f'{condition}/seed={seed}/summary.json'
                results.append(read(path) if path.exists() else {'condition':condition,'seed':seed,'status':'missing'})
    per_note=[];per_family=[];per_seed=[]
    for result in results:
        if result['status']!='complete':continue
        folder=output/f'{result["condition"]}/seed={result["seed"]}'
        with (folder/'per_note.csv').open() as stream:rows=list(csv.DictReader(stream))
        per_note.extend(rows)
        per_seed.extend([{**aggregate,'seed':result['seed']} for aggregate in aggregate_note_rows(rows)])
        by_family=defaultdict(list)
        for row in rows:
            by_family[row['family']].append(row)
        for family,group in by_family.items():
            for aggregate in aggregate_note_rows(group):
                per_family.append({**aggregate,'family':family,'seed':result['seed']})
    aggregates=aggregate_note_rows(per_note)
    if per_note:
        write_csv(output/'per_note.csv',per_note)
        write_csv(output/'per_family.csv',per_family)
        write_csv(output/'comparison.csv',aggregates)
        write_csv(output/'per_seed.csv',per_seed)
    complete=[r for r in results if r['status']=='complete']
    # Pairing checks across architectures; never compare unrelated corruptions or z draws.
    for prefix,field in [('vae','generation_codes_sha256'),('dae','corrupted_inputs_sha256')]:
        for seed in SEEDS:
            pair=[r for r in complete if r['seed']==seed and r['condition'] in (f'{prefix}_mlp',f'{prefix}_conv')]
            if len(pair)==2 and pair[0][field]!=pair[1][field]:
                raise ValueError('Unpaired evaluation randomness')
    generation=[r for r in complete if r['condition'].startswith('vae')]
    if generation:write_csv(output/'generation_summary.csv',generation)
    value={'protocol':PROTOCOL,'completed_runs':len(complete),'expected_runs':12,
           'status':'complete' if len(complete)==12 else 'incomplete','runs':results,'aggregates':aggregates,
           'notes':['Held-out instruments, not a new source dataset.','Noise is Gaussian in normalized mel magnitudes, not additive microphone noise.',
                    'Nearest-neighbor distances do not establish novelty or musical quality.','Only three independent training seeds per condition.']}
    (output/'summary.json').write_text(json.dumps(value,indent=2))
    lines=['# NSynth: arquitectura densa frente a CNN',f'Estado: {value["status"]}, {len(complete)}/12 entrenamientos evaluados.',
           '','## Reconstrucción y denoising en instrumentos reservados','',
           '| Condición | Tarea | Método | Ruido mel σ | Casos | MSE | Inválidos |',
           '|---|---|---|---:|---:|---:|---:|']
    lines += [f'| {r["condition"]} | {r["task"]} | {r["method"]} | {r["noise_level"]} | {r["count"]} | {r["mse"]:.6f} | {r["invalid"]} |' for r in aggregates]
    lines += ['','## Generación libre del VAE','',
              '| Condición | Semilla | MSE al vecino train | MSE al vecino test | Diversidad MSE entre pares | Vecinos train distintos /64 |',
              '|---|---:|---:|---:|---:|---:|']
    lines += [f'| {r["condition"]} | {r["seed"]} | {r["generation_train_mse"]:.6f} | {r["generation_test_mse"]:.6f} | {r["generation_pairwise_mse"]:.6f} | {r["generation_distinct_train_neighbors"]} |' for r in generation]
    lines += ['','La separación de instrumentos evita que notas del mismo instrumento estén a la vez en train y test. '
              'Las métricas se exportan por nota, familia y semilla. Las notas de un instrumento no son entrenamientos independientes.',
              'El denoising usa ruido gaussiano σ0.1 en espectrogramas mel durante entrenamiento. Evaluamos σ0,0.05,0.1,0.2 con entradas idénticas entre arquitecturas. '
              'Esto no equivale a entrenar o evaluar eliminación de ruido acústico de un micrófono.',
              'Generamos64 códigos normales estándar por semilla, idénticos entre VAE. Los primeros6 fijados se exportan a WAV. '
              'Las distancias a entrenamiento y prueba, su diversidad y los vecinos distintos describen las muestras, pero no certifican calidad, novedad ni ausencia de memorización. '
              'Los conjuntos train y test tienen tamaños distintos: las distancias mínimas no son directamente comparables como evidencia de generalización. '
              'La MSE global pondera cada nota por igual, no cada familia. El CSV por familia permite inspeccionar ese desequilibrio.',
              'Los espectrogramas comparativos usan semilla7 y ejemplos prefijados por familia, sin elegir los mejores. '
              'El códec mel introduce degradación incluso sin red. Los WAV de referencia deben distinguir original y referencia pasada por el códec.',
              'No se modificó ninguna fórmula de pérdida: VAE MSE+betaKL, AE MSE. No se evalúan aquí notas omitidas.']
    (output/'informe.md').write_text('\n'.join(lines)+'\n')
    create_example_figures(root,results)
    plot_metrics(root,aggregates,per_seed,generation)
    return value


def run(root=ROOT,jobs=2):
    root=Path(root);assert_loss_unchanged(root)
    tasks=[(str(root),condition,seed) for condition in CONDITIONS for seed in SEEDS]
    if jobs==1:results=[evaluate_run(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:results=list(pool.map(evaluate_run,tasks))
    value=report(root,results)
    assert_loss_unchanged(root)
    print(json.dumps({'completed_runs':value['completed_runs'],'expected_runs':12,'status':value['status']},indent=2))
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    run(jobs=args.jobs)
