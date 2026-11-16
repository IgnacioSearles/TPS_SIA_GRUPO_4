"""Offline matched-note listening page, regenerated from frozen mel examples.

No training, existing waveform replacement, or model decoding is performed.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.data.audio import SpectrogramCodec,waveform_to_wav_bytes
from src.data.nsynth_cnn_study import build_study_data,digest_samples,parse_label

SEEDS=(7,17,27)
PHASES=('pghi','random')
KINDS=('original','codec','dense','cnn')
OUTPUT=Path('reports/nsynth_cnn_listening')


def sha(value):return hashlib.sha256(value).hexdigest()


@lru_cache(maxsize=4)
def cached_codec(description):return SpectrogramCodec(**json.loads(description))


def audit_wav(payload):
    with wave.open(io.BytesIO(payload)) as stream:
        result={'sample_rate':stream.getframerate(),'samples':stream.getnframes(),
                'channels':stream.getnchannels(),'sample_width_bytes':stream.getsampwidth()}
    assert result=={'sample_rate':16000,'samples':16000,'channels':1,'sample_width_bytes':2}
    return {**result,'duration_seconds':1.,'wav_sha256':sha(payload),'wav_bytes':len(payload)}


def decode_one(task):
    key,spec,description,cache=task
    codec=cached_codec(description)
    signature=sha(spec.astype('<f8').tobytes()+description.encode())
    path=Path(cache)/f'nsynth_listen_{signature}.wav'
    if path.is_file():payload=path.read_bytes()
    else:
        waveform=codec.decode(spec.reshape(codec.image_shape))
        if not np.isfinite(waveform).all():raise ValueError('Non-finite audio')
        payload=waveform_to_wav_bytes(waveform,codec.sample_rate)
        path.write_bytes(payload)
    return key,payload,{'cache_signature':signature,'spectrogram_sha256':digest_samples(spec),
                       'codec':json.loads(description),**audit_wav(payload)}


def load_examples(root,condition,seed,labels):
    path=root/f'reports/nsynth_cnn_study/{condition}/seed={seed}/examples.npz'
    with np.load(path,allow_pickle=False) as saved:
        if list(saved['labels'])!=labels:raise ValueError(f'Example label/order mismatch: {path}')
        clean=saved['clean_examples'].copy();reconstructed=saved['reconstructed_examples'].copy()
    if clean.shape!=reconstructed.shape or clean.shape!=(6,8192):raise ValueError(f'Bad shape: {path}')
    if not np.isfinite(reconstructed).all():raise ValueError(f'Non-finite reconstruction: {path}')
    return clean,reconstructed,{'path':str(path.relative_to(root)),'sha256':sha(path.read_bytes())}


def build(root=ROOT,jobs=2,windows_output=None):
    root=Path(root);output=root/OUTPUT;output.mkdir(parents=True,exist_ok=True)
    cache=output/'cache';cache.mkdir(exist_ok=True)
    study=build_study_data(root)
    manifest=study['manifest'];labels=manifest['eval_example_labels']
    assert len(labels)==6
    codec=study['test'].audio_codec
    originals_path=root/'results/nsynth_cnn_study/data/example_originals.npz'
    with np.load(originals_path,allow_pickle=False) as originals:
        assert list(originals['labels'])==labels
        raw=originals['waveforms'].copy()
        assert int(originals['sample_rate'])==16000
    sources=[{'path':str(originals_path.relative_to(root)),'sha256':sha(originals_path.read_bytes())}]
    tasks=[];audio={};audits={};reference=None
    for seed in SEEDS:
        left,dense,source=load_examples(root,'vae_mlp',seed,labels);sources.append(source)
        right,cnn,source=load_examples(root,'vae_conv',seed,labels);sources.append(source)
        if not np.array_equal(left,right):raise ValueError('Dense/CNN reference mismatch')
        if reference is not None and not np.array_equal(reference,left):raise ValueError('Seed reference mismatch')
        reference=left
        for phase in PHASES:
            description=json.dumps(asdict(replace(codec,phase_initialization=phase)),sort_keys=True)
            for kind,values in (('dense',dense),('cnn',cnn)):
                for index,spec in enumerate(values):
                    key=f'{seed}/{phase}/{kind}/{index}'
                    tasks.append((key,spec,description,str(cache)))
    for index,(label,waveform) in enumerate(zip(labels,raw)):
        expected=study['test'].samples[study['test'].index_of(label)]
        if not np.array_equal(reference[index],expected):raise ValueError('Saved reference/test split mismatch')
        if sha(waveform.astype('<f8').tobytes())!=manifest['raw_example_sha256'][label]:
            raise ValueError('Original reference hash mismatch')
        if not np.allclose(codec.encode(waveform).ravel(),expected,atol=1e-7,rtol=0):
            raise ValueError('Raw original is not the encoded reference')
        key=f'original/{index}';payload=waveform_to_wav_bytes(waveform,16000)
        audio[key]=payload;audits[key]={'raw_sha256':sha(waveform.astype('<f8').tobytes()),**audit_wav(payload)}
        for phase in PHASES:
            description=json.dumps(asdict(replace(codec,phase_initialization=phase)),sort_keys=True)
            tasks.append((f'codec/{phase}/{index}',reference[index],description,str(cache)))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for key,payload,audit in pool.map(decode_one,tasks):audio[key]=payload;audits[key]=audit
    files=[]
    for seed in SEEDS:
        for phase in PHASES:
            for index,label in enumerate(labels):
                for kind in KINDS:
                    key=f'original/{index}' if kind=='original' else f'codec/{phase}/{index}' if kind=='codec' else f'{seed}/{phase}/{kind}/{index}'
                    path=output/f'audio/{phase}/seed={seed}/{kind}/{label}.wav'
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(audio[key])
                    files.append({'seed':seed,'phase':phase,'kind':kind,'label':label,
                                  'path':str(path.relative_to(root)),**audits[key]})
    notes=[parse_label(label) for label in labels]
    payload={'seeds':SEEDS,'phases':PHASES,'notes':notes,
             'audio':{key:'data:audio/wav;base64,'+base64.b64encode(value).decode() for key,value in audio.items()}}
    html=page(payload)
    (output/'index.html').write_text(html,encoding='utf-8')
    audit={'scope':'six predeclared held-out NSynth notes, original VAE warmup50 models only',
           'seeds':SEEDS,'phases':PHASES,'labels':labels,'sources':sources,
           'model_weights_not_loaded_or_changed':True,'old_waveforms_not_overwritten':True,
           'same_codec_per_architecture':True,'all_label_orders_verified':True,
           'manifest_sha256':sha((root/'results/nsynth_cnn_study/data/manifest.json').read_bytes()),
           'html_sha256':sha(html.encode()),'unique_embedded_wavs':len(audio),'files':files,
           'normalization':'codec decodes peak-normalized; raw originals retain existing full-note peak normalization, then crop; no loudness matching'}
    (output/'audio_manifest.json').write_text(json.dumps(audit,indent=2))
    (output/'FUENTES.md').write_text(
        '# Comparación de escucha NSynth\n\n'
        'Seis notas fijadas en el manifiesto antes de observar los resultados; no son los mejores casos. '
        'Instrumentos de prueba, fuera del entrenamiento de estos modelos.\n\n'
        'Originales: results/nsynth_cnn_study/data/example_originals.npz. '
        'Espectrogramas limpios y reconstrucciones: reports/nsynth_cnn_study/vae_{mlp,conv}/seed={7,17,27}/examples.npz. '
        'No se cargan pesos ni se reentrena. Se verifican etiquetas, orden, referencias y hashes antes de generar.\n\n'
        'PGHI y fase aleatoria inicializan la reconstrucción de fase, seguida en ambos casos de 60 iteraciones Griffin-Lim; inversión mel 200 iteraciones. '
        'El selector se aplica al control de códec y a ambos modelos simultáneamente. Una mejora por cambiar fase no es una mejora de la CNN. '
        'La fase cambia el audio, no los espectrogramas ni la MSE. Los WAV históricos no se modifican.\n\n'
        'WAV nuevos: audio/{pghi,random}/seed={7,17,27}/{original,codec,dense,cnn}/{label}.wav. '
        'Cada WAV es mono PCM16, 16000Hz, un segundo. Las salidas del códec se normalizan a pico1; no hay igualación de sonoridad. '
        'audio_manifest.json identifica cada recurso y el códec explícito. index.html incorpora todos los WAV y funciona sin conexión.\n\n'
        'Regenerar: OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/build_nsynth_listening.py --jobs 2\n')
    if windows_output:
        destination=Path(windows_output);destination.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(output/'index.html',destination/'escuchar.html')
        shutil.copyfile(output/'audio_manifest.json',destination/'audio_manifest.json')
        shutil.copyfile(output/'FUENTES.md',destination/'FUENTES.md')
    print(json.dumps({'page':str(output/'index.html'),'unique_wavs':len(audio),
                      'wav_paths':len(files),'audio_manifest':str(output/'audio_manifest.json')},indent=2))
    return audit


def page(payload):
    data=json.dumps(payload,ensure_ascii=False)
    return '''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Escucha comparada · NSynth</title><style>
:root{--ink:#173547;--muted:#536c7a;--paper:#fff;--wash:#eaf2f5;--line:#cad9e0;--dense:#087f8c;--cnn:#6554c0}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);font-family:Bahnschrift,"Segoe UI",Arial,sans-serif;font-size:17px;line-height:1.5}
main{max-width:1450px;margin:auto;padding:54px 52px 70px}h1{font-size:clamp(36px,4.2vw,62px);line-height:1.04;font-weight:650;letter-spacing:-1.5px;max-width:850px;margin:0 0 20px}
.intro{max-width:790px;color:var(--muted);font-size:20px;margin-bottom:32px}p{margin:10px 0}.controls{display:flex;align-items:end;gap:26px;flex-wrap:wrap;padding:22px 0;border-top:2px solid var(--ink);border-bottom:1px solid var(--line)}
label{display:grid;gap:6px;font-weight:600;font-size:15px}select,button{font:inherit;border:1px solid #8da7b4;padding:10px 16px;background:white;color:var(--ink);border-radius:4px;min-height:46px}button{cursor:pointer}button:hover{background:var(--wash)}:focus-visible{outline:3px solid #0b8797;outline-offset:4px}
.phase-note{max-width:960px;color:var(--muted);font-size:16px;margin:18px 0 30px}.guide{display:grid;grid-template-columns:1.1fr 1fr 1fr 1fr 1fr;gap:18px;align-items:start;padding:16px 0;background:var(--wash);border-bottom:1px solid var(--line)}
.guide>div:first-child{padding-left:14px}.guide strong{display:block;font-size:18px}.guide small{display:block;color:var(--muted);line-height:1.3;font-size:14px;margin-top:4px}.dense{color:var(--dense)}.cnn{color:var(--cnn)}
.note{display:grid;grid-template-columns:1.1fr 1fr 1fr 1fr 1fr;gap:18px;align-items:center;border-bottom:1px solid var(--line);padding:26px 0}.family{font-size:23px;font-weight:650;text-transform:capitalize;line-height:1.15;margin-bottom:7px}.metadata{font-size:13px;color:var(--muted);overflow-wrap:anywhere}.pitch{font-size:16px;margin-bottom:5px}.audio-label{display:none}audio{display:block;width:100%;height:42px;min-width:0}.player{min-width:0}.status{font-size:14px;min-height:22px;color:var(--muted);margin-left:auto}footer{max-width:900px;margin-top:34px;color:var(--muted);font-size:15px}footer strong{color:var(--ink)}
@media(max-width:1000px){main{padding:30px 24px}.guide{display:none}.note{grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.note>header{grid-column:1/-1}.audio-label{display:block;font-size:15px;font-weight:600;margin-bottom:6px}.status{margin-left:0}}
@media(max-width:520px){main{padding:24px 16px}.controls{gap:15px}label{width:100%}select{width:100%}.note{gap:18px 12px}.intro{font-size:18px}h1{letter-spacing:-.8px}}
</style><main><h1>La misma nota.<br>Cuatro formas de escucharla.</h1>
<p class="intro">Compará el sonido real, lo que conserva el espectrograma y lo que reconstruyen el VAE denso y el convolucional. Seis notas de instrumentos que estos modelos no vieron al entrenar.</p>
<div class="controls"><label>Semilla de entrenamiento<select id="seed" aria-label="Semilla de entrenamiento"><option>7</option><option>17</option><option>27</option></select></label>
<label>Reconstrucción de fase<select id="phase" aria-label="Reconstrucción de fase"><option value="pghi">PGHI + Griffin-Lim</option><option value="random">Aleatoria + Griffin-Lim</option></select></label><button id="stop" type="button">Detener todo</button><p class="status" id="status" aria-live="polite">Listo para escuchar</p></div>
<p class="phase-note">El cambio de fase se aplica al códec, al denso y a la CNN a la vez. Si mejora al elegir PGHI, esa mejora proviene de reconstruir el audio, no de cambiar la red. El original NSynth se mantiene igual.</p>
<div class="guide"><div><strong>Nota de prueba</strong><small>Misma nota en toda la fila</small></div><div><strong>Original NSynth</strong><small>Grabación de referencia, 1s</small></div><div><strong>Original por el códec</strong><small>Mel original convertido a audio</small></div><div><strong class="dense">VAE denso</strong><small>Reconstrucción de la misma entrada</small></div><div><strong class="cnn">VAE convolucional</strong><small>Reconstrucción de la misma entrada</small></div></div>
<div id="notes"></div><footer><p><strong>Cómo compararlos:</strong> primero escuchá original y códec. Esa diferencia ya existe antes de usar una red. Después compará denso y CNN con el control del códec.</p><p>Los seis ejemplos se fijaron antes de observar resultados. Son los mismos para las tres semillas y no representan por sí solos las 224 notas evaluadas. Ambos VAE usan calentamiento KL de 50 épocas.</p><p>Cada audio es mono, 16kHz y dura un segundo. Las salidas del códec se normalizan por pico, sin igualación de sonoridad. Se reproduce un solo audio por vez. Todos los sonidos están incluidos en esta página y funcionan sin conexión.</p></footer></main>
<script id="listening-data" type="application/json">'''+data+'''</script><script>
'use strict';const study=JSON.parse(document.getElementById('listening-data').textContent);const names=['Original NSynth','Original por el códec','VAE denso','VAE convolucional'];const kinds=['original','codec','dense','cnn'];const noteNames=['Do','Do♯','Re','Re♯','Mi','Fa','Fa♯','Sol','Sol♯','La','La♯','Si'];let players=[];
function stopAll(except=null){for(const a of players){if(a!==except){a.pause();a.currentTime=0}}}
function render(){stopAll();players=[];const seed=document.getElementById('seed').value,phase=document.getElementById('phase').value;document.getElementById('notes').replaceChildren();
study.notes.forEach((note,index)=>{const row=document.createElement('section');row.className='note';const header=document.createElement('header');const family=document.createElement('div');family.className='family';family.textContent=note.family;header.append(family);const pitch=document.createElement('div');pitch.className='pitch';pitch.textContent=noteNames[note.pitch%12]+(Math.floor(note.pitch/12)-1)+' · MIDI '+note.pitch+' · velocidad '+note.velocity;header.append(pitch);const metadata=document.createElement('div');metadata.className='metadata';metadata.textContent=note.instrument;header.append(metadata);row.append(header);
kinds.forEach((kind,column)=>{const cell=document.createElement('div');cell.className='player';const title=document.createElement('label');title.className='audio-label '+(kind==='dense'?'dense':kind==='cnn'?'cnn':'');title.textContent=names[column];cell.append(title);const a=document.createElement('audio');a.controls=true;a.preload='metadata';a.setAttribute('aria-label',names[column]+', '+note.label);const key=kind==='original'?'original/'+index:kind==='codec'?'codec/'+phase+'/'+index:seed+'/'+phase+'/'+kind+'/'+index;a.src=study.audio[key];a.addEventListener('play',()=>{stopAll(a);document.getElementById('status').textContent=note.family+' · '+names[column]});a.addEventListener('ended',()=>{document.getElementById('status').textContent='Terminó '+note.family+' · '+names[column]});a.addEventListener('error',()=>{document.getElementById('status').textContent='No se pudo reproducir '+note.label});cell.append(a);row.append(cell);players.push(a)});document.getElementById('notes').append(row)});document.getElementById('status').textContent='Semilla '+seed+' · fase '+(phase==='pghi'?'PGHI':'aleatoria')}
document.getElementById('seed').addEventListener('change',render);document.getElementById('phase').addEventListener('change',render);document.getElementById('stop').addEventListener('click',()=>{stopAll();document.getElementById('status').textContent='Todos los audios detenidos'});render();
</script></html>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--windows-output',type=Path)
    args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs must be positive')
    build(jobs=args.jobs,windows_output=args.windows_output)

