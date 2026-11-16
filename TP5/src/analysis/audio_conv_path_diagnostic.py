"""Frozen-model comparison of interpolating before versus after the decoder.

Run: OPENBLAS_NUM_THREADS=1 python -m src.analysis.audio_conv_path_diagnostic --jobs 2
Only complete-control models are used. No training or reserved-final evaluation.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.art_search import assert_loss_unchanged
from src.data.audio import write_waveform
from src.evaluation.context import write_csv
from src.evaluation.pitch import pitch_measurement
from src.evaluation.pitch_interpolation import spectral_measurement
from src.experiments.runner import load_run

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path('reports/audio_conv_path_diagnostic')
FRACTIONS=(.25,.5,.75)
METHODS=('latent_mix','decoded_neighbor_mix','spectrogram_mix')


def generate_paths(model, neighbors, fractions=FRACTIONS):
    """Only neighbor inputs enter this function. No target can affect generation."""
    neighbors=np.asarray(neighbors,dtype=model.precision)
    weights=np.asarray(fractions,dtype=model.precision)[:,None]
    means=model.encode(neighbors)
    endpoint_outputs=model.decode(means)
    return {
        'latent_mix':model.decode((1-weights)*means[0]+weights*means[1]),
        'decoded_neighbor_mix':(1-weights)*endpoint_outputs[0]+weights*endpoint_outputs[1],
        'spectrogram_mix':(1-weights)*neighbors[0]+weights*neighbors[1],
    }


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_run(task):
    root,architecture,seed=task
    root=Path(root)
    directory=root/f'results/ej2_audio_conv_{architecture}_complete/base__seed={seed}'
    if not (directory/'summary.json').is_file():
        return {'architecture':architecture,'seed':seed,'status':'missing','rows':[]}
    before=file_hash(directory/'model.npz')
    model,dataset=load_run(directory)
    study=dataset.pitch_study
    assert not study.omit_targets and list(study.targets)==[51,57,63]
    assert list(study.reserved_targets)==[55,61,67]
    codec=dataset.audio_codec
    output=root/OUTPUT/f'{architecture}_seed={seed}'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for variant in study.evaluation_variants:
        for target in study.targets:
            neighbors=dataset.samples[[dataset.index_of(study.note(p,variant).label) for p in (target-1,target+1)]]
            paths=generate_paths(model,neighbors)
            # Load target references only AFTER paths are generated.
            refs={p:codec.encode(study.note(p,variant).waveform).ravel() for p in (target-1,target,target+1)}
            for method,values in paths.items():
                for fraction,spec in zip(FRACTIONS,values):
                    row={'architecture':architecture,'condition':'complete','seed':seed,
                         'target_pitch':target,'variant':variant,'method':method,'fraction':fraction,
                         **spectral_measurement(spec,refs,target),'pitch_evaluated':fraction==.5}
                    if fraction==.5:
                        waveform=codec.decode(spec.reshape(codec.image_shape))
                        row.update(pitch_measurement(waveform,codec.sample_rate,target))
                        write_waveform(output/f'{target}_{variant}_{method}.wav',waveform,codec.sample_rate)
                    rows.append(row)
    after=file_hash(directory/'model.npz')
    assert before==after,'Frozen model was modified'
    write_csv(output/'rows.csv',rows)
    return {'architecture':architecture,'seed':seed,'status':'complete','model_sha256':before,'rows':rows}


def summarize(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['architecture'],row['method'],row['fraction']].append(row)
    aggregates=[]
    for (architecture,method,fraction),group in sorted(grouped.items()):
        measured=[r for r in group if r['pitch_evaluated']]
        aggregates.append({'architecture':architecture,'method':method,'fraction':fraction,'trials':len(group),
                           'spectral_hits':sum(r['spectral_hit'] for r in group),
                           'spectral_mse':float(np.mean([r['spectral_mse'] for r in group])),
                           'pitch_trials':len(measured),'pitch_hits':sum(r['hit'] for r in measured),
                           'invalid':sum(not r['valid'] for r in measured),
                           'joint_hits':sum(r['hit'] and r['spectral_hit'] for r in measured)})
    return aggregates


def plot(aggregates,path):
    """Show the pitch-only trap alongside the combined criterion, all cases included."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    central=[r for r in aggregates if r['fraction']==.5]
    names={'latent_mix':'Mezcla latente','decoded_neighbor_mix':'Mezcla reconstruida','spectrogram_mix':'Mezcla mel'}
    labels=[f"{r['architecture'].upper()}\n{names[r['method']]}" for r in central]
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,field,title in zip(axes,('pitch_hits','joint_hits'),('Altura ≤ 50 cents','Altura y cercanía espectral')):
        bars=ax.bar(range(len(central)),[100*r[field]/r['trials'] for r in central],
                    color=['#176a98' if r['architecture']=='mlp' else '#d4602a' for r in central])
        ax.set_ylim(0,112);ax.set_ylabel('% de todas las predicciones');ax.set_title(title)
        ax.set_xticks(range(len(central)),labels,rotation=30,ha='right',fontsize=9)
        for bar,row in zip(bars,central):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{row[field]}/{row['trials']}",ha='center',fontsize=10)
    fig.savefig(path,dpi=150);plt.close(fig)


def audit_existing(root=ROOT):
    """Validate a completed report against pre-existing metrics and codec calibration."""
    root=Path(root);output=root/OUTPUT
    with (output/'rows.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    numerical=[];calibrations=[];comparisons=[];per_seed=[]
    for architecture in ('mlp','conv'):
        for seed in (7,17,27):
            directory=root/f'results/ej2_audio_conv_{architecture}_complete/base__seed={seed}'
            if not (directory/'summary.json').is_file():continue
            detail=json.loads((directory/'pitch_summary.json').read_text())
            calibrations.append({'architecture':architecture,'seed':seed,
                                 'reliable':detail['pitch_calibration_reliable'],
                                 'by_variant':detail['calibration_by_variant']})
            with (directory/'pitch_interpolation.csv').open(newline='') as stream:
                old={(r['target_pitch'],r['variant']):r for r in csv.DictReader(stream)
                     if r['method']=='vae_interpolation' and float(r['fraction'])==.5}
            selected=[r for r in rows if r['architecture']==architecture and int(r['seed'])==seed]
            converted=[{k:float(v) if k in ('fraction','spectral_mse') else v=='True' if k in
                       ('pitch_evaluated','hit','valid','spectral_hit') else v for k,v in r.items()}
                       for r in selected]
            per_seed.extend(dict(seed=seed,**r) for r in summarize(converted))
            for row in selected:
                if row['method']!='latent_mix' or float(row['fraction'])!=.5:continue
                previous=old[row['target_pitch'],row['variant']]
                error=abs(float(row['spectral_mse'])-float(previous['spectral_mse']))
                numerical.append(error)
                comparison={'architecture':architecture,'seed':seed,'target_pitch':row['target_pitch'],
                            'variant':row['variant'],'spectral_mse_absolute_difference':error,
                            'pitch_valid_same':row['valid']==previous['valid'],
                            'pitch_hit_same':row['hit']==previous['hit'],
                            'spectral_hit_same':row['spectral_hit']==previous['spectral_hit'],
                            'frequency_absolute_difference_hz':None if not row['frequency_hz'] or not previous['frequency_hz']
                            else abs(float(row['frequency_hz'])-float(previous['frequency_hz']))}
                comparisons.append(comparison)
    # The codec initializes Griffin-Lim from the same fixed phase seed on every call.
    model,dataset=load_run(root/'results/ej2_audio_conv_mlp_complete/base__seed=7')
    study=dataset.pitch_study
    neighbors=dataset.samples[[dataset.index_of(study.note(p,'v00').label) for p in (50,52)]]
    spec=generate_paths(model,neighbors)['spectrogram_mix'][1]
    codec=dataset.audio_codec
    one=codec.decode(spec.reshape(codec.image_shape));two=codec.decode(spec.reshape(codec.image_shape))
    deterministic=np.array_equal(one,two)
    result={'calibration':calibrations,'all_codec_calibrations_reliable':all(r['reliable'] for r in calibrations),
            'comparisons':len(comparisons),'same_pitch_hits':all(r['pitch_hit_same'] for r in comparisons),
            'same_pitch_validity':all(r['pitch_valid_same'] for r in comparisons),
            'same_spectral_hits':all(r['spectral_hit_same'] for r in comparisons),
            'max_spectral_mse_absolute_difference':max(numerical),
            'max_frequency_difference_hz':max(r['frequency_absolute_difference_hz'] or 0 for r in comparisons),
            'codec_exactly_deterministic_on_repeated_input':deterministic,
            'codec_waveform_sha256':hashlib.sha256(one.tobytes()).hexdigest()}
    assert deterministic and result['same_pitch_hits'] and result['same_pitch_validity'] and result['same_spectral_hits']
    assert result['max_spectral_mse_absolute_difference']<1e-7
    write_csv(output/'comparison_to_original.csv',comparisons)
    write_csv(output/'per_seed.csv',per_seed)
    (output/'audit.json').write_text(json.dumps(result,indent=2))
    with (output/'informe.md').open('a') as stream:
        stream.write('\nVerificación: las 54 predicciones latentes centrales conservan aciertos de altura, validez y aciertos espectrales del CSV original.\n')
        stream.write(f"Diferencia máxima MSE: {max(numerical):.3g}. Diferencia máxima de frecuencia: {result['max_frequency_difference_hz']:.3g} Hz.\n")
        stream.write('El códec devuelve exactamente el mismo audio al repetir una entrada. Las seis calibraciones previas son fiables: ver audit.json para tasas por variante.\n')
    return result


def run(root=ROOT,jobs=2):
    root=Path(root);assert_loss_unchanged(root)
    tasks=[(str(root),architecture,seed) for architecture in ('mlp','conv') for seed in (7,17,27)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        results=list(pool.map(evaluate_run,tasks))
    rows=[r for result in results for r in result.pop('rows')]
    if not rows:raise RuntimeError('No finished models available')
    aggregates=summarize(rows)
    output=root/OUTPUT;output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'rows.csv',rows);write_csv(output/'comparison.csv',aggregates)
    plot(aggregates,output/'comparison.png')
    (output/'summary.json').write_text(json.dumps({'runs':results,'aggregates':aggregates,
        'fractions':FRACTIONS,'pitch_only_at_midpoint':True,'no_training':True,
        'scope':'validation only; target notes were present during complete-control training'},indent=2))
    lines=['# ¿Interpolar antes o después del decoder?','',
        'Modelos congelados del control completo: las notas objetivo estuvieron en entrenamiento.',
        'El objetivo solo mide las salidas. Para generarlas entran los vecinos MIDI ±1.',
        'La mezcla latente decodifica el promedio de medias posteriores; la mezcla reconstruida promedia las dos salidas del decoder.',
        'La mezcla mel promedia directamente los espectrogramas de entrada.',
        'La altura se mide únicamente en el punto medio predefinido. Los puntos 0,25 y 0,75 son diagnóstico de distancia al objetivo central, no se esperan idénticos al objetivo ni se elige el mejor.',
        'Inválidos cuentan como fallos. Las variantes de envolvente no son entrenamientos independientes.','',
        '| Modelo | Método central | Altura | Espectro | Ambos | Inválidos | MSE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in aggregates:
        if row['fraction']!=.5:continue
        n=row['trials']
        lines.append(f"| {row['architecture']} | {row['method']} | {row['pitch_hits']}/{n} | {row['spectral_hits']}/{n} | {row['joint_hits']}/{n} | {row['invalid']} | {row['spectral_mse']:.6f} |")
    lines+=['','La comparación mide la no linealidad del decoder con pesos fijos. Una mejora de altura aislada puede reflejar dos notas mezcladas.',
            'No prueba interpolación de notas ausentes, no certifica calidad auditiva, y no modifica la comparación principal de 12 corridas.']
    (output/'informe.md').write_text('\n'.join(lines)+'\n')
    audit_existing(root)
    assert_loss_unchanged(root)
    print('\n'.join(lines),flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    run(jobs=args.jobs)
