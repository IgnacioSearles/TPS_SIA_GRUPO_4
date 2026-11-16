"""Evaluate stronger mel corruption with the same six frozen denoising models."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv,hashlib,json
import numpy as np

from src.analysis.nsynth_cnn_evaluation import corruption,reconstruct_batches
from src.data.nsynth_cnn_study import build_study_data
from src.data.art_search import assert_loss_unchanged
from src.evaluation.context import write_csv
from src.experiments.runner import load_run

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path('reports/nsynth_denoising_noise_sweep')
LEVELS=(0.,.05,.1,.2,.3,.4,.5)

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def evaluate(task):
    architecture,seed=task
    condition=f'dae_{architecture}'
    folder=ROOT/f'results/ej2_nsynth_cnn_{condition}/base__seed={seed}'
    prior=ROOT/f'reports/nsynth_cnn_study/{condition}/seed={seed}'
    digest=sha(folder/'model.npz');assert digest==read(prior/'summary.json')['model_sha256']
    data=build_study_data(ROOT);test=data['test']
    model,train=load_run(folder);assert train.labels==data['train'].labels
    with (prior/'per_note.csv').open() as stream:
        old={(r['label'],float(r['noise_level']),r['method']):r for r in csv.DictReader(stream)}
    rows=[];maximum_difference=0.
    for level in LEVELS:
        noisy=corruption(test.samples,seed,level)
        if level<=.2:
            assert hashlib.sha256(noisy.tobytes()).hexdigest()==read(prior/'summary.json')['corrupted_inputs_sha256'][str(level)]
        reconstructed=reconstruct_batches(model,noisy)
        for method,values in [('input',noisy),('model',reconstructed)]:
            valid=np.isfinite(values).all(axis=1)
            errors=np.mean((values.astype(float)-test.samples)**2,axis=1)
            for i,label in enumerate(test.labels):
                error=float(errors[i]) if valid[i] else float('inf')
                if (label,level,method) in old:
                    difference=abs(error-float(old[label,level,method]['mse']))
                    assert difference<1e-7
                    maximum_difference=max(maximum_difference,difference)
                rows.append({'condition':condition,'seed':seed,'label':label,'family':test.groups[i],
                             'noise_level':level,'method':method,'mse':error,'valid':bool(valid[i])})
        print(f'{condition} seed={seed} sigma={level} evaluated',flush=True)
    assert sha(folder/'model.npz')==digest
    return rows,{'condition':condition,'seed':seed,'model_sha256':digest,'weights_unchanged':True,
                 'original_levels_reproduced_max_mse_difference':maximum_difference}

def run():
    assert_loss_unchanged(ROOT);output=ROOT/OUTPUT;output.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(evaluate,[(a,s) for a in ('mlp','conv') for s in (7,17,27)]))
    rows=[r for part,_ in results for r in part]
    per_seed=[];means=[]
    for condition in ('dae_mlp','dae_conv'):
        for level in LEVELS:
            for method in ('input','model'):
                for seed in (7,17,27):
                    group=[r for r in rows if r['condition']==condition and r['noise_level']==level and r['method']==method and r['seed']==seed]
                    assert len(group)==224
                    per_seed.append({'condition':condition,'noise_level':level,'method':method,'seed':seed,
                                     'count':224,'mse':float(np.mean([r['mse'] for r in group]))})
                group=[r for r in per_seed if r['condition']==condition and r['noise_level']==level and r['method']==method]
                means.append({'condition':condition,'noise_level':level,'method':method,'mse':float(np.mean([r['mse'] for r in group]))})
    write_csv(output/'per_note.csv',rows);write_csv(output/'per_seed.csv',per_seed);write_csv(output/'comparison.csv',means)
    (output/'summary.json').write_text(json.dumps({'status':'complete','seeds':[7,17,27],'levels':LEVELS,
        'train_noise':.1,'no_training':True,'model_audits':[a for _,a in results],'aggregates':means},indent=2))
    lines=['# Más ruido, mismos modelos','', 'Evaluación sin reentrenar:224 notas, tres semillas y dos arquitecturas. Ruido de entrenamiento:σ0.1.',
           '', '| σ | Entrada | Denso | CNN | Reducción CNN frente a entrada |','|---|---:|---:|---:|---:|']
    for level in LEVELS:
        get=lambda c,m:next(r['mse'] for r in means if r['condition']==c and r['method']==m and r['noise_level']==level)
        baseline,dense,cnn=get('dae_mlp','input'),get('dae_mlp','model'),get('dae_conv','model')
        improvement=f'{100*(1-cnn/baseline):.1f}%' if baseline else '—'
        lines.append(f'| {level} | {baseline:.6f} | {dense:.6f} | {cnn:.6f} | {improvement} |')
    lines+=['','La reducción negativa significa que empeora. Las magnitudes se recortan a[0,1]: a ruido fuerte también hay saturación. '
              'Se evalúa corrupción mel, no ruido acústico. Esta ampliación no modifica pesos ni los resultados guardados del estudio inicial.']
    (output/'informe.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));assert_loss_unchanged(ROOT)

if __name__=='__main__':run()
