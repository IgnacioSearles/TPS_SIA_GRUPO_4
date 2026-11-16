"""Matched denoising listening: frozen DAE models, four exact test corruptions.

Regenerates full-test noise before selecting six examples. Existing reports,
models and WAVs are read-only. Every comparison uses the explicit PGHI codec.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
import csv
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.build_nsynth_listening import cached_codec,audit_wav,sha,page as reconstruction_page
from src.analysis.nsynth_cnn_evaluation import corruption,reconstruct_batches
from src.data.audio import waveform_to_wav_bytes
from src.data.nsynth_cnn_study import build_study_data,parse_label
from src.evaluation.context import write_csv
from src.experiments.runner import load_run

OUTPUT=Path('reports/nsynth_denoising_listening')
SEEDS=(7,17,27)
LEVELS=(0.,.05,.1,.2,.3,.4,.5)
KINDS=('clean','noisy','dense','cnn')


def read(path):return json.loads(path.read_text())


def decode_one(task):
    """Canonical float64 inputs and atomic, versioned cache prevent dtype/race drift."""
    key,spec,description,cache=task
    spec=np.ascontiguousarray(spec,dtype='<f8')
    codec=cached_codec(description)
    signature=sha(spec.tobytes()+description.encode())
    path=Path(cache)/f'nsynth_denoise_f64_v1_{signature}.wav'
    if path.is_file():payload=path.read_bytes()
    else:
        waveform=codec.decode(spec.reshape(codec.image_shape))
        if not np.isfinite(waveform).all():raise ValueError('Non-finite audio')
        payload=waveform_to_wav_bytes(waveform,codec.sample_rate)
        temporary=path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_bytes(payload);os.replace(temporary,path)
    return key,payload,{'cache_signature':signature,'cache_version':'float64_atomic_v1',
                       'spectrogram_sha256':sha(spec.astype('<f4').tobytes()),'codec_input_precision':'float64',
                       'codec':json.loads(description),**audit_wav(payload)}


def prepare_seed(task):
    root,seed=task;root=Path(root);output=root/OUTPUT;output.mkdir(parents=True,exist_ok=True)
    study=build_study_data(root);test=study['test'];manifest=study['manifest']
    labels=manifest['eval_example_labels'];indices=[test.index_of(label) for label in labels]
    noisy={str(level):corruption(test.samples,seed,level) for level in LEVELS}
    inputs={str(level):value[indices] for level,value in ((level,noisy[str(level)]) for level in LEVELS)}
    arrays={'labels':np.asarray(labels),'clean':test.samples[indices]}
    for level in LEVELS:arrays[f'noisy_{level}']=inputs[str(level)]
    audit={'seed':seed,'test_count':len(test),'labels':labels,'noise_scope':'all224testnotes_before_selecting6',
           'noise_hashes':{key:sha(value.tobytes()) for key,value in noisy.items()},'models':[],'mse_checks':[]}
    for kind,condition in (('dense','dae_mlp'),('cnn','dae_conv')):
        run=root/f'results/ej2_nsynth_cnn_{condition}/base__seed={seed}'
        old=root/f'reports/nsynth_cnn_study/{condition}/seed={seed}'
        summary=read(old/'summary.json');before=sha((run/'model.npz').read_bytes())
        if before!=summary['model_sha256']:raise ValueError(f'Weights differ from evaluated model: {run}')
        if any(audit['noise_hashes'][k]!=v for k,v in summary['corrupted_inputs_sha256'].items()):raise ValueError(f'Test noise hash mismatch: {condition}/{seed}')
        model,used=load_run(run)
        if used.labels!=study['train'].labels or set(used.labels).intersection(test.labels):
            raise ValueError('Held-out split or training order mismatch')
        with (old/'per_note.csv').open(newline='') as stream:
            previous={(r['label'],float(r['noise_level']),r['method']):r for r in csv.DictReader(stream)}
        with (root/'reports/nsynth_denoising_noise_sweep/per_note.csv').open(newline='') as stream:
            previous.update({(r['label'],float(r['noise_level']),r['method']):r for r in csv.DictReader(stream) if r['condition']==condition and int(r['seed'])==seed})
        for level in LEVELS:
            value=reconstruct_batches(model,inputs[str(level)])
            if not np.isfinite(value).all():raise ValueError('Non-finite denoising output')
            arrays[f'{kind}_{level}']=value
            for method,prediction in (('model',value),('input',inputs[str(level)])):
                errors=np.mean((prediction.astype(float)-test.samples[indices])**2,axis=1)
                for label,error in zip(labels,errors):
                    original=previous[label,level,method]
                    difference=abs(float(error)-float(original['mse']))
                    if difference>1e-7 or original['valid']!='True':
                        raise ValueError(f'Per-note MSE mismatch: {condition} {seed} {label} {level} {difference}')
                    audit['mse_checks'].append({'condition':condition,'seed':seed,'label':label,'sigma':level,
                        'method':method,'mse':float(error),'original_mse':float(original['mse']),
                        'absolute_difference':difference})
        if sha((run/'model.npz').read_bytes())!=before:raise ValueError('Model weights changed')
        audit['models'].append({'condition':condition,'weights_sha256':before,
                               'summary_sha256':sha((old/'summary.json').read_bytes()),
                               'per_note_csv_sha256':sha((old/'per_note.csv').read_bytes()),
                               'weights_unchanged':True,'noise_hashes_match':True})
    np.savez_compressed(output/f'spectrograms_seed={seed}.npz',**arrays)
    (output/f'inference_audit_seed={seed}.json').write_text(json.dumps(audit,indent=2))
    return arrays,audit


def key_for(seed,level,kind,index):
    return f'clean/{index}' if kind=='clean' else f'{seed}/{level}/{kind}/{index}'


def export_priority(root=ROOT,jobs=2):
    """Make the seven fixed slide examples available before the full page."""
    root=Path(root);arrays,audit=prepare_seed((str(root),7))
    study=build_study_data(root);codec=replace(study['test'].audio_codec,phase_initialization='pghi')
    description=json.dumps(asdict(codec),sort_keys=True);output=root/OUTPUT
    cache=output/'cache';cache.mkdir(exist_ok=True)
    tasks=[('clean',arrays['clean'][0],description,str(cache))]
    tasks.extend((f'{level}/{kind}',arrays[f'{kind}_{level}'][0],description,str(cache))
                 for level in (.1,.5) for kind in ('noisy','dense','cnn'))
    files=[];label=str(arrays['labels'][0])
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for key,payload,wav_audit in pool.map(decode_one,tasks):
            levels=(.1,.5) if key=='clean' else (float(key.split('/')[0]),)
            kind='clean' if key=='clean' else key.split('/')[1]
            for level in levels:
                path=output/f'audio/seed=7/sigma={level}/{kind}/{label}.wav'
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
                files.append({'label':label,'seed':7,'sigma':level,'kind':kind,'path':str(path.relative_to(root)),**wav_audit})
    (output/'priority_audio_manifest.json').write_text(json.dumps({'files':files,'unique_audio_count':7,
                     'explicit_phase':'pghi','inference_audit':'inference_audit_seed=7.json'},indent=2))
    print('PRIORITY READY: 7 unique WAV, bass seed7, sigma0.1/0.5',flush=True)


def build(root=ROOT,jobs=2,windows_output=None):
    root=Path(root);output=root/OUTPUT;output.mkdir(parents=True,exist_ok=True)
    study=build_study_data(root);labels=study['manifest']['eval_example_labels']
    codec=replace(study['test'].audio_codec,phase_initialization='pghi')
    description=json.dumps(asdict(codec),sort_keys=True);cache=output/'cache';cache.mkdir(exist_ok=True)
    prepared=[]
    # Always revalidate source hashes; no unversioned inference cache is trusted.
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        prepared=list(pool.map(prepare_seed,[(str(root),seed) for seed in SEEDS]))
    tasks=[];audio={};audits={}
    for seed,(arrays,audit) in zip(SEEDS,prepared):
        if list(arrays['labels'])!=labels:raise ValueError('Seed example alignment mismatch')
        for level in LEVELS:
            for kind in KINDS:
                if kind=='clean' and (seed!=SEEDS[0] or level!=LEVELS[0]):continue
                values=arrays['clean'] if kind=='clean' else arrays[f'{kind}_{level}']
                for index,spec in enumerate(values):tasks.append((key_for(seed,level,kind,index),spec,description,str(cache)))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for key,payload,audit in pool.map(decode_one,tasks):audio[key]=payload;audits[key]=audit
    zero_matches=all(audio[key_for(seed,0.,'noisy',index)]==audio[key_for(seed,0.,'clean',index)]
                     for seed in SEEDS for index in range(len(labels)))
    if not zero_matches:raise ValueError('Sigma zero input differs from clean reference after decoding')
    files=[]
    for seed in SEEDS:
        for level in LEVELS:
            for kind in KINDS:
                for index,label in enumerate(labels):
                    key=key_for(seed,level,kind,index)
                    path=output/f'audio/seed={seed}/sigma={level}/{kind}/{label}.wav'
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(audio[key])
                    files.append({'seed':seed,'sigma':level,'kind':kind,'label':label,'path':str(path.relative_to(root)),**audits[key]})
    mse_checks=[r for _,audit in prepared for r in audit['mse_checks']]
    write_csv(output/'per_note_mse_audit.csv',mse_checks)
    payload={'seeds':SEEDS,'levels':LEVELS,'notes':[parse_label(label) for label in labels],
             'audio':{key:'data:audio/wav;base64,'+base64.b64encode(value).decode() for key,value in audio.items()}}
    html=page(payload);(output/'index.html').write_text(html)
    audit={'protocol':'nsynth_denoising_listening_v1','seeds':SEEDS,'sigma':LEVELS,'labels':labels,
           'codec':asdict(codec),'all_audio_same_pghi_codec':True,'test_count':224,
           'codec_input_precision':'float64','cache_version':'float64_atomic_v1',
           'sigma_zero_input_clean_byte_identical_all_18':zero_matches,
           'noise_generated_before_selecting_six':True,'noise_hashes_match_both_original_models':True,
           'inference':{'notes_per_seed_level_model':6,'models':6,'levels':len(LEVELS),'predictions':len(SEEDS)*len(LEVELS)*len(labels)*2,
                        'per_note_mse_checks_including_input':len(mse_checks),
                        'max_mse_absolute_difference':max(r['absolute_difference'] for r in mse_checks)},
           'model_audits':[r for _,result in prepared for r in result['models']],
           'embedded_audio_entries':len(audio),'unique_wav_contents':len({sha(value) for value in audio.values()}),
           'external_wav_paths':len(files),'files':files,
           'html_sha256':sha(html.encode()),'no_training':True,'old_waveforms_preserved':True}
    (output/'audio_manifest.json').write_text(json.dumps(audit,indent=2))
    (output/'FUENTES.md').write_text('# Escucha completa de denoising NSynth\n\n'
        'Seis ejemplos fijados en results/nsynth_cnn_study/data/manifest.json. Modelos Densos y convolucionales de '
        'results/ej2_nsynth_cnn_dae_{mlp,conv}/base__seed={7,17,27}. No se entrenan ni modifican.\n\n'
        'La contaminación se regenera con src.analysis.nsynth_cnn_evaluation.corruption sobre las 224 notas de prueba '
        'antes de seleccionar 6. Los 12 hashes de entrada se verifican originales contra ambos summary.json; los niveles nuevos se verifican contra reports/nsynth_denoising_noise_sweep/per_note.csv. '
        'Las 252 reconstrucciones y 252 entradas tienen MSE por nota comprobada con per_note.csv original, tolerancia 1e-7.\n\n'
        'El ruido es gaussiano sobre valores mel normalizados, con recorte [0,1], no ruido acústico añadido a la onda. '
        'Todos los audios se regeneran con PGHI y 60 iteraciones Griffin-Lim e inversión mel 200. '
        'Todas las entradas al códec usan float64. El caché incluye versión y codec, y se escribe de forma atómica. '
        'Una mejora por reconstruir fase no es una mejora de arquitectura. No se reemplazan WAV históricos.\n\n'
        'WAV: audio/seed={seed}/sigma={level}/{clean,noisy,dense,cnn}/{label}.wav. '
        'Todos mono PCM16, 16000Hz, 1s y normalizados por pico del códec. Sin igualación de sonoridad. '
        'La página incorpora todos los audios sin requerir conexión.\n')
    if windows_output:
        destination=Path(windows_output);destination.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(output/'index.html',destination/'denoising.html')
        shutil.copyfile(output/'audio_manifest.json',destination/'denoising_audio_manifest.json')
        shutil.copyfile(output/'FUENTES.md',destination/'DENOISING_FUENTES.md')
    print(json.dumps({k:audit[k] for k in ('embedded_audio_entries','unique_wav_contents','external_wav_paths','inference')},indent=2),flush=True)
    return audit


def page(payload):
    style=re.search(r'<style>(.*?)</style>',reconstruction_page({}),re.S).group(1)
    return '''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Denoising NSynth · Escucha comparada</title><style>'''+style+'''</style><main>
<h1>Ruido añadido.<br>Sonido reconstruido.</h1><p class="intro">La misma nota: limpia, con ruido, reconstruida por el denso y por la CNN.</p>
<div class="controls"><label>Semilla de entrenamiento<select id="seed"><option>7</option><option>17</option><option>27</option></select></label>
<label>Ruido gaussiano en mel, σ<select id="sigma"><option value="0.0">0 · sin ruido</option><option value="0.05">0,05</option><option value="0.1" selected>0,1 · nivel entrenado</option><option value="0.2">0,2</option><option value="0.3">0,3</option><option value="0.4">0,4</option><option value="0.5">0,5</option></select></label><button type="button" id="stop">Detener todo</button><p class="status" id="status" aria-live="polite"></p></div>
<details class="phase-note"><summary>Detalles del experimento</summary>Ruido gaussiano sobre mel, recorte [0,1]. Todos usan el mismo códec PGHI. Modelos entrenados con σ = 0,1.</details>
<div class="guide"><div><strong>Nota de prueba</strong><small>La misma en toda la fila</small></div><div><strong>Referencia limpia</strong><small>Mel original pasado por el códec</small></div><div><strong>Entrada con ruido</strong><small>Contaminación que recibe cada red</small></div><div><strong class="dense">Denso</strong><small>Reconstrucción de la entrada ruidosa</small></div><div><strong class="cnn">CNN</strong><small>Reconstrucción de la misma entrada</small></div></div>
<div id="notes"></div><footer>Compará cuánto ruido desaparece y cuánto sonido original se pierde.</footer></main>
<script id="listening-data" type="application/json">'''+json.dumps(payload,ensure_ascii=False)+'''</script><script>
'use strict';const study=JSON.parse(document.getElementById('listening-data').textContent);const kinds=['clean','noisy','dense','cnn'];const names=['Referencia limpia','Entrada con ruido','Denso','CNN'];const tones=['Do','Do♯','Re','Re♯','Mi','Fa','Fa♯','Sol','Sol♯','La','La♯','Si'];let players=[];
function stopAll(except=null){for(const a of players)if(a!==except){a.pause();a.currentTime=0}}
function render(){stopAll();players=[];const seed=document.getElementById('seed').value,level=document.getElementById('sigma').value;document.getElementById('notes').replaceChildren();study.notes.forEach((note,index)=>{const row=document.createElement('section');row.className='note';const header=document.createElement('header');const family=document.createElement('div');family.className='family';family.textContent=note.family;header.append(family);const pitch=document.createElement('div');pitch.className='pitch';pitch.textContent=tones[note.pitch%12]+(Math.floor(note.pitch/12)-1)+' · MIDI '+note.pitch+' · velocidad '+note.velocity;header.append(pitch);const metadata=document.createElement('div');metadata.className='metadata';metadata.textContent=note.instrument;header.append(metadata);row.append(header);
kinds.forEach((kind,column)=>{const cell=document.createElement('div');cell.className='player';const title=document.createElement('label');title.className='audio-label '+(kind==='dense'?'dense':kind==='cnn'?'cnn':'');title.textContent=names[column];cell.append(title);const a=document.createElement('audio');a.controls=true;a.preload='metadata';a.setAttribute('aria-label',names[column]+', '+note.label);const key=kind==='clean'?'clean/'+index:seed+'/'+level+'/'+kind+'/'+index;a.src=study.audio[key];a.addEventListener('play',()=>{stopAll(a);document.getElementById('status').textContent=note.family+' · '+names[column]});a.addEventListener('error',()=>document.getElementById('status').textContent='No se pudo reproducir '+note.label);cell.append(a);row.append(cell);players.push(a)});document.getElementById('notes').append(row)});document.getElementById('status').textContent='Semilla '+seed+' · σ '+level.replace('.',',')+' · fase PGHI'}
document.getElementById('seed').addEventListener('change',render);document.getElementById('sigma').addEventListener('change',render);document.getElementById('stop').addEventListener('click',()=>{stopAll();document.getElementById('status').textContent='Todos los audios detenidos'});render();
</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--windows-output',type=Path)
    parser.add_argument('--priority-only',action='store_true')
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    if args.priority_only:export_priority(jobs=args.jobs)
    else:build(jobs=args.jobs,windows_output=args.windows_output)
