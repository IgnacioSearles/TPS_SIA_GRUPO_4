"""Fixed MLP/CNN audio comparison; audits and reports, no model/loss changes.

python -m src.analysis.audio_convolution --audit
python -m src.analysis.audio_convolution
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.data.art_search import assert_loss_unchanged
from src.data.factory import build_dataset
from src.evaluation.context import write_csv
from src.evaluation.pitch_interpolation import validate_pitch_interpolation
from src.experiments.grid import expand_runs
from src.models.factory import build_model
from src.options import ConfigurationError, Options

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = 'reports/ej2_audio_convolution'
ARCHITECTURES = ('mlp', 'conv')
MODES = ('complete', 'omitted')
TARGETS = [51, 57, 63]
RESERVED = [55, 61, 67]
VARIANTS = ['v00', 'v20', 'v39']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def configuration_path(root, architecture, mode):
    return Path(root)/f'configs/audio_conv/ej2_audio_conv_{architecture}_{mode}.json'


def configurations(root):
    return {(a,m):read(configuration_path(root,a,m)) for a in ARCHITECTURES for m in MODES}


def validate_configurations(configs):
    """Reject confounding changes, pitch supervision, and final-reference access."""
    if set(configs)!={(a,m) for a in ARCHITECTURES for m in MODES}:
        raise ConfigurationError('Se requieren las cuatro condiciones MLP/CNN, completo/omitido')
    normalized=[]
    for (architecture,mode),config in configs.items():
        value=copy.deepcopy(config)
        settings=value['dataset']['pitch_study']
        if (settings['targets']!=TARGETS or settings['reserved_targets']!=RESERVED
                or settings['evaluation_variants']!=VARIANTS
                or settings['exclusion_radius']!=.75
                or settings['omit_targets']!=(mode=='omitted')
                or value['seeds']!=[7,17,27]):
            raise ConfigurationError('Separación/semillas distintas del protocolo de audio CNN')
        model=value['model']
        if model.pop('architecture')!=architecture:
            raise ConfigurationError('Arquitectura incorrecta')
        if architecture=='conv':
            if model.pop('image_shape')!=[128,64,1] or model.pop('conv_channels')!=[8,16]:
                raise ConfigurationError('La CNN requiere mapas mel 128×64×1 y canales 8/16')
        settings.pop('omit_targets')
        value.pop('name'); value.pop('output')
        normalized.append(value)
    if any(value!=normalized[0] for value in normalized[1:]):
        raise ConfigurationError('Solo pueden variar arquitectura y pertenencia de objetivos')
    reference=normalized[0]
    expected_model={'type':'variational','latent_size':16,'hidden_layers':[512,128],
                    'activation':'relu','initializer':'he','loss':'mse','precision':'float32',
                    'kl':{'weight':.5,'warmup_epochs':50}}
    expected_training={'epochs':200,'batch_size':64,'log_every':10,
                       'optimizer':{'type':'adam','learning_rate':.001}}
    expected_dataset={'type':'synthetic_notes','pitch_range':[48,72],
                      'pitch_study':{'targets':TARGETS,'reserved_targets':RESERVED,
                                     'evaluation_variants':VARIANTS,'exclusion_radius':.75},
                      'spectrogram':{'duration_seconds':1.,'fft_size':1024,'hop_length':256,'mel_bands':128},
                      'variants_per_pitch':40,'microtones_per_semitone':4,'microtone_variants':10}
    if (reference['model']!=expected_model or reference['training']!=expected_training
            or reference['dataset']!=expected_dataset):
        raise ConfigurationError('Cambió el protocolo fijo o se agregó supervisión de pitch')


def training_fingerprint(dataset):
    digest=hashlib.sha256()
    digest.update(json.dumps(dataset.labels,ensure_ascii=False).encode())
    digest.update(np.ascontiguousarray(dataset.samples,dtype='<f4').tobytes())
    digest.update(json.dumps(asdict(dataset.audio_codec),sort_keys=True).encode())
    return digest.hexdigest()


def audit(root=ROOT):
    root=Path(root)
    configs=configurations(root)
    validate_configurations(configs)
    guard=assert_loss_unchanged(root)
    records={}
    counts={}
    # Build both independently so a future model-specific dataset change is detected.
    for architecture in ARCHITECTURES:
        for mode in MODES:
            config=configs[architecture,mode]
            dataset=build_dataset(Options(config['dataset'],'dataset'),root)
            validate_pitch_interpolation(dataset,Options({},'pitch_interpolation'))
            if tuple(dataset.image_shape)!=(128,64):
                raise ConfigurationError('El orden del códec debe ser frecuencia × tiempo')
            labels=set(dataset.labels)
            forbidden=RESERVED+(TARGETS if mode=='omitted' else [])
            for note in dataset.pitch_study.notes:
                if note.label in labels and any(abs(note.pitch-p)<=.75+1e-9 for p in forbidden):
                    raise ConfigurationError('Una altura excluida entró en entrenamiento')
            model=build_model(Options(config['model'],'model'),dataset.samples.shape[1],
                              np.random.default_rng(7),np.random.default_rng(8))
            counts[architecture]=sum(p.size for p in model.parameters())
            records[f'{architecture}_{mode}']={
                'samples':len(dataset),'training_sha256':training_fingerprint(dataset),
                'codec':asdict(dataset.audio_codec),'parameters':counts[architecture]}
            del model,dataset
    for mode in MODES:
        if records[f'mlp_{mode}']['training_sha256']!=records[f'conv_{mode}']['training_sha256']:
            raise ConfigurationError('Las arquitecturas no reciben exactamente los mismos datos')
    difference=counts['conv']/counts['mlp']-1
    if abs(difference)>.01:
        raise ConfigurationError('La diferencia de parámetros supera el 1 %')
    result={'protocol':'audio_convolution_v1','datasets':records,'parameters_relative_difference':difference,
            'loss_guard':guard,'targets':TARGETS,'reserved_targets_not_evaluated':RESERVED,
            'seeds':[7,17,27],'expected_runs':12,'no_pitch_supervision':True}
    output=root/OUTPUT;output.mkdir(parents=True,exist_ok=True)
    (output/'protocol.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def summarize_rows(rows):
    hits=[r['hit']=='True' for r in rows]
    spectra=[r['spectral_hit']=='True' for r in rows]
    per_variant={v:{'hits':sum(r['hit']=='True' for r in rows if r['variant']==v),
                    'total':sum(r['variant']==v for r in rows)} for v in VARIANTS}
    return {'trials':len(rows),'pitch_hits':sum(hits),'pitch_rate':float(np.mean(hits)),
            'spectral_hits':sum(spectra),'spectral_rate':float(np.mean(spectra)),
            'joint_hits':sum(a and b for a,b in zip(hits,spectra)),
            'valid_rate':float(np.mean([r['valid']=='True' for r in rows])),
            'spectral_mse':float(np.mean([float(r['spectral_mse']) for r in rows])),
            'per_variant':per_variant}


def report(root=ROOT):
    root=Path(root);configs=configurations(root);validate_configurations(configs)
    assert_loss_unchanged(root)
    output=root/OUTPUT;output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list);trajectories=[];run_rows=[];missing=[];calibration=True
    paired={};signatures={}
    for (architecture,mode),config in configs.items():
        experiment=root/config['output']
        run_config={k:v for k,v in config.items() if k not in ('name','output')}
        for run in expand_runs(run_config):
            directory=experiment/run.label
            if not (directory/'summary.json').is_file():
                missing.append(str(directory.relative_to(root)));continue
            if read(directory/'config.json')!=run.config:
                raise ConfigurationError(f'Configuración incompatible en {directory}')
            manifest=read(directory/'pitch_manifest.json')
            metrics=read(directory/'pitch_summary.json')
            if manifest.get('reserved_targets')!=RESERVED or manifest['targets']!=TARGETS:
                raise ConfigurationError('Referencias incompatibles')
            for note in manifest['notes']:
                forbidden=RESERVED+(TARGETS if mode=='omitted' else [])
                if note['split']=='training' and any(abs(note['pitch']-p)<=.75+1e-9 for p in forbidden):
                    raise ConfigurationError('Contaminación en el manifiesto de entrenamiento')
            signature=json.dumps({k:manifest[k] for k in ('notes','codec','targets','reserved_targets')},sort_keys=True)
            signatures.setdefault(mode,signature)
            if signatures[mode]!=signature:
                raise ConfigurationError('Los manifiestos no usan los mismos audios y separación')
            with (directory/'pitch_interpolation.csv').open(newline='') as stream:
                rows=list(csv.DictReader(stream))
            expected_methods={'vae_interpolation','spectrogram_mix','lower_neighbor'}
            if mode=='complete':expected_methods.add('target_reconstruction')
            central_keys=[(int(r['target_pitch']),r['variant'],r['method']) for r in rows if float(r['fraction'])==.5]
            expected_keys={(p,v,m) for p in TARGETS for v in VARIANTS for m in expected_methods}
            if len(central_keys)!=len(expected_keys) or set(central_keys)!=expected_keys:
                raise ConfigurationError('Predicciones centrales faltantes o repetidas')
            path_keys=[(int(r['target_pitch']),r['variant'],round(float(r['fraction']),6))
                       for r in rows if r['method']=='vae_interpolation']
            expected_paths={(p,v,round(f,6)) for p in TARGETS for v in VARIANTS for f in np.linspace(0,1,11)}
            if len(path_keys)!=len(expected_paths) or set(path_keys)!=expected_paths:
                raise ConfigurationError('Trayectorias incompletas o repetidas')
            calibration &= metrics['pitch_calibration_reliable']
            for row in rows:
                row={**row,'architecture':architecture,'condition':mode,'run':str(directory.relative_to(root))}
                trajectories.append(row)
                if float(row['fraction'])!=.5:continue
                groups[architecture,mode,row['method']].append(row)
                if row['method'] in ('spectrogram_mix','lower_neighbor'):
                    key=(mode,run.seed,row['target_pitch'],row['variant'],row['method'])
                    comparison={k:v for k,v in row.items() if k not in ('architecture','run')}
                    if key in paired and paired[key]!=comparison:
                        raise ConfigurationError('Las referencias sin aprendizaje difieren entre arquitecturas')
                    paired[key]=comparison
            training=read(directory/'training.json')
            run_rows.append({'architecture':architecture,'condition':mode,'seed':run.seed,
                             'samples':training['samples'],'training_seconds':training['training_seconds'],
                             'train_mse':read(directory/'summary.json')['mean_squared_error'],
                             **{k:v for k,v in metrics.items() if isinstance(v,(float,int,bool))}})
    aggregates=[];per_seed=[]
    for (architecture,mode,method),rows in groups.items():
        expected={'vae_interpolation','spectrogram_mix','lower_neighbor'}
        if mode=='complete':expected.add('target_reconstruction')
        if method not in expected:raise ConfigurationError('Método inesperado')
        keys=[(r['seed'],r['target_pitch'],r['variant']) for r in rows]
        if len(set(keys))!=len(keys):raise ConfigurationError('Predicciones duplicadas')
        aggregates.append({'architecture':architecture,'condition':mode,'method':method,**summarize_rows(rows)})
        for seed in sorted({int(r['seed']) for r in rows}):
            selected=[r for r in rows if int(r['seed'])==seed]
            per_seed.append({'architecture':architecture,'condition':mode,'method':method,
                             'seed':seed,**summarize_rows(selected)})
    favorable={}
    for architecture in ARCHITECTURES:
        selected={r['method']:r for r in aggregates if r['architecture']==architecture and r['condition']=='omitted'}
        if missing or not calibration or 'vae_interpolation' not in selected:
            favorable[architecture]=None;continue
        main=selected['vae_interpolation']
        favorable[architecture]=bool(main['trials']==27
            and all(v['total']==9 and v['hits']>=8 for v in main['per_variant'].values())
            and all(main['pitch_rate']>selected[b]['pitch_rate'] for b in ('spectrogram_mix','lower_neighbor')))
    status='incomplete' if missing else 'inconclusive_codec' if not calibration else 'complete'
    result={'status':status,'completed_runs':len(run_rows),'expected_runs':12,'missing_runs':missing,
            'codec_reliable':bool(calibration) if run_rows else None,'favorable_pitch_criterion':favorable,
            'aggregates':aggregates,'note':'Pitch criterion is not evidence of spectral fidelity or novelty.'}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if trajectories:write_csv(output/'trajectories.csv',trajectories)
    if run_rows:write_csv(output/'runs.csv',run_rows)
    if per_seed:write_csv(output/'per_seed.csv',[{k:v for k,v in r.items() if k!='per_variant'} for r in per_seed])
    if aggregates:
        write_csv(output/'comparison.csv',[{k:v for k,v in r.items() if k!='per_variant'} for r in aggregates])
        plot(aggregates,output/'comparison.png')
    lines=['# Audio: VAE denso frente a convolucional','',f'Estado: **{status}**, {len(run_rows)}/12 corridas terminadas.','',
           'Cada denominador incluye inválidos como fallos. El punto central se fijó antes de observar resultados.',
           'No se usa la trayectoria para elegir la mejor fracción. Las variantes de una nota no son entrenamientos independientes.','',
           '| Modelo | Condición | Método | Altura | Espectro más próximo al objetivo | Ambos | MSE espectral |',
           '|---|---|---|---:|---:|---:|---:|']
    for row in aggregates:
        lines.append(f"| {row['architecture']} | {row['condition']} | {row['method']} | {row['pitch_hits']}/{row['trials']} | {row['spectral_hits']}/{row['trials']} | {row['joint_hits']}/{row['trials']} | {row['spectral_mse']:.6f} |")
    lines += ['', 'Las reconstrucciones directas de objetivos solo aparecen en los controles completos; no intervienen en la interpolación.',
              'La MSE de entrenamiento no es una métrica de generalización. La cercanía espectral no certifica calidad auditiva ni novedad.',
              'Las notas finales 55, 61 y 67 no se entrenan ni se evalúan. Este informe corresponde únicamente a validación.',
              'Si faltan corridas o la calibración del códec falla, no se declara evidencia favorable ni un ganador.',
              'Un resultado musical negativo es un resultado válido. No se inicia una búsqueda de hiperparámetros automáticamente.']
    (output/'informe.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def plot(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['method'] in ('vae_interpolation','target_reconstruction')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    labels=[f"{r['architecture']} / {r['condition']}\n{r['method']}" for r in selected]
    for ax,field,title in zip(axes,['pitch_hits','joint_hits'],['Altura ≤ 50 cents','Altura y cercanía espectral']):
        values=[100*r[field]/r['trials'] for r in selected]
        bars=ax.bar(np.arange(len(selected)),values,color=['#176a98' if r['architecture']=='mlp' else '#d4602a' for r in selected])
        ax.set_xticks(np.arange(len(selected)),labels,rotation=35,ha='right',fontsize=8)
        ax.set_ylim(0,112);ax.set_ylabel('% de todas las predicciones');ax.set_title(title)
        for bar,row in zip(bars,selected):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{row[field]}/{row['trials']}",ha='center',fontsize=9)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',action='store_true',help='verificar datos, parámetros y fórmulas antes de entrenar')
    args=parser.parse_args()
    result=audit() if args.audit else report()
    print(json.dumps(result if args.audit else {k:v for k,v in result.items() if k!='aggregates'},indent=2))


if __name__=='__main__':main()
