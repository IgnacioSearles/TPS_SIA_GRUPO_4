"""Shared NSynth selection and instrument-held-out split for the CNN study.

No factory/model/trainer changes. Training configs consume train_settings via
the existing dataset.subset option. Group membership is fixed before training.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np

from src.data.audio import read_waveform
from src.data.factory import build_dataset
from src.options import ConfigurationError,Options

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path('results/nsynth_cnn_study/data')
SPLIT_SEED=20261115
TEST_FRACTION=.2
SETTINGS={'type':'nsynth','pitch_range':[48,72],'max_per_family':150,'selection_seed':0,
          'spectrogram':{'duration_seconds':1.,'fft_size':1024,'hop_length':256,'mel_bands':128}}
PATTERN=re.compile(r'(?P<family>[a-z]+)_(?P<source>[a-z]+)_(?P<id>\d+)-(?P<pitch>\d+)-(?P<velocity>\d+)$')


def digest_samples(values):
    """Canonical encoded-audio hash: row-major little-endian float32."""
    return hashlib.sha256(np.ascontiguousarray(values,dtype='<f4').tobytes()).hexdigest()


def parse_label(label):
    match=PATTERN.fullmatch(label)
    if match is None:raise ConfigurationError(f'Etiqueta NSynth inválida: {label}')
    fields=match.groupdict()
    return {'label':label,'family':fields['family'],'source':fields['source'],
            'instrument':label.split('-')[0],'pitch':int(fields['pitch']),'velocity':int(fields['velocity'])}


def instrument_split(dataset,seed=SPLIT_SEED,test_fraction=TEST_FRACTION):
    """Assign complete instruments within every family, independent of row order.

    Pick round-half-up(fraction * instrument_count) test instruments, constrained
    to [1,n-1]. SHA256(seed,family,instrument) defines their fixed ranking.
    """
    if not 0<test_fraction<1:raise ConfigurationError('test_fraction debe estar entre 0 y 1')
    if len(set(dataset.labels))!=len(dataset.labels):raise ConfigurationError('Etiquetas NSynth duplicadas')
    records=[parse_label(label) for label in dataset.labels]
    instruments=defaultdict(set)
    for record in records:instruments[record['family']].add(record['instrument'])
    insufficient={family:len(groups) for family,groups in instruments.items() if len(groups)<2}
    if insufficient:
        raise ConfigurationError(f'Split por instrumento requiere al menos 2 instrumentos por familia: {insufficient}. No se cambia a split por notas.')
    test_instruments=set()
    for family,groups in sorted(instruments.items()):
        rank=lambda instrument:hashlib.sha256(f'{seed}:{family}:{instrument}'.encode()).hexdigest()
        ordered=sorted(groups,key=lambda instrument:(rank(instrument),instrument))
        count=max(1,min(len(ordered)-1,int(np.floor(test_fraction*len(ordered)+.5))))
        test_instruments.update(ordered[:count])
    for record in records:
        record['split']='test' if record['instrument'] in test_instruments else 'train'
        record['spectrogram_sha256']=digest_samples(dataset.samples[dataset.index_of(record['label'])])
    records.sort(key=lambda record:record['label'])
    train_labels=[r['label'] for r in records if r['split']=='train']
    test_labels=[r['label'] for r in records if r['split']=='test']
    assert set(train_labels).isdisjoint(test_labels)
    assert set(r['instrument'] for r in records if r['split']=='train').isdisjoint(test_instruments)
    examples={split:{} for split in ('train','test')}
    family_counts=[]
    for family in sorted(instruments):
        counts={'family':family}
        for split in ('train','test'):
            selected=[r for r in records if r['family']==family and r['split']==split]
            labels=[r['label'] for r in selected]
            positions=np.linspace(0,len(labels)-1,min(3,len(labels)),dtype=int)
            examples[split][family]=[labels[i] for i in positions]
            counts[f'{split}_notes']=len(labels)
            counts[f'{split}_instruments']=len({r['instrument'] for r in selected})
        family_counts.append(counts)
    return {'train_labels':train_labels,'test_labels':test_labels,'records':records,
            'family_counts':family_counts,'examples':examples,
            'eval_example_labels':[examples['test'][family][0] for family in sorted(instruments)[:6]]}


def original_examples(root,dataset,labels,output):
    """Read only six fixed raw references, in one streaming pass over the archive."""
    path=output/'example_originals.npz'
    codec=dataset.audio_codec
    if path.is_file():
        with np.load(path,allow_pickle=False) as saved:
            if list(saved['labels'])!=labels:raise ConfigurationError('Cambió la selección de ejemplos originales')
            waveforms=saved['waveforms'].copy()
    else:
        archive=root/'data/nsynth/nsynth-test.jsonwav.tar.gz'
        if not archive.is_file():
            raise FileNotFoundError(f'Falta {archive}; descargar con: python scripts/download_nsynth.py')
        pending=set(labels);found={}
        with tarfile.open(archive,'r|gz') as stream:
            for member in stream:
                if not member.isfile() or Path(member.name).stem not in pending:continue
                label=Path(member.name).stem
                waveform=read_waveform(io.BytesIO(stream.extractfile(member).read()),codec.sample_rate)
                fitted=np.zeros(codec.sample_count,dtype=float)
                count=min(len(fitted),len(waveform));fitted[:count]=waveform[:count]
                found[label]=fitted;pending.remove(label)
                if not pending:break
        if pending:raise ConfigurationError(f'No se encontraron originales: {sorted(pending)}')
        waveforms=np.asarray([found[label] for label in labels],dtype='<f8')
        np.savez_compressed(path,labels=np.asarray(labels),waveforms=waveforms,sample_rate=codec.sample_rate)
    if waveforms.shape!=(len(labels),codec.sample_count):raise ConfigurationError('Formato de originales incompatible')
    for label,waveform in zip(labels,waveforms):
        expected=dataset.samples[dataset.index_of(label)]
        if not np.allclose(codec.encode(waveform).ravel(),expected,atol=1e-7,rtol=0):
            raise ConfigurationError(f'Original y espectrograma guardado no coinciden: {label}')
    return {label:hashlib.sha256(waveform.astype('<f8').tobytes()).hexdigest() for label,waveform in zip(labels,waveforms)}


def reconcile_manifest(previous,current):
    """Keep historical provenance; permit only the decoder phase initialization field.

    Absence of phase_initialization denotes the legacy random initializer. Every
    other field, including all encoded hashes and split/selection settings, must
    compare exactly. Decoder iteration counts are deliberately not ignored.
    """
    old=deepcopy(previous);new=deepcopy(current)
    old_phase=old['codec'].pop('phase_initialization','random')
    new_phase=new['codec'].pop('phase_initialization','random')
    if old_phase not in ('random','pghi') or new_phase not in ('random','pghi') or old!=new:
        raise ConfigurationError('Cambió el dataset/split/códec respecto del manifiesto existente; '
                                 'solo se admite phase_initialization como cambio de decodificación')
    return previous,{'stored_phase_initialization':old_phase,'current_phase_initialization':new_phase,
                     'phase_changed':old_phase!=new_phase,
                     'legacy_manifest_preserved':True,'encoding_selection_split_identical':True,
                     'stored_codec':deepcopy(previous['codec']),'current_codec':deepcopy(current['codec'])}


def build_study_data(root=ROOT):
    """Return full/train/test Datasets, manifest, and existing-factory train options."""
    root=Path(root);output=root/OUTPUT;output.mkdir(parents=True,exist_ok=True)
    full=build_dataset(Options(deepcopy(SETTINGS),'dataset'),root)
    if tuple(full.image_shape)!=(128,64):raise ConfigurationError('NSynth CNN requiere mel 128×64')
    manifest=instrument_split(full)
    train=full.select(manifest['train_labels']);test=full.select(manifest['test_labels'])
    train_settings={**deepcopy(SETTINGS),'subset':manifest['train_labels']}
    manifest.update({'protocol':'nsynth_cnn_instrument_split_v1','selection':deepcopy(SETTINGS),
                     'split_seed':SPLIT_SEED,'test_instrument_fraction_requested':TEST_FRACTION,
                     'split_rule':'per-family SHA256 ranking, nearest20% instruments rounded half up, min1/maxN-1',
                     'split_unit':'exact instrument including family and source; all velocities and pitches grouped',
                     'codec':asdict(full.audio_codec),'encoded_hash_format':'C-order little-endian float32',
                     'full_sha256':digest_samples(full.samples),'train_sha256':digest_samples(train.samples),
                     'test_sha256':digest_samples(test.samples),'full_count':len(full),
                     'train_count':len(train),'test_count':len(test),
                     'instrument_counts':dict(sorted(Counter(r['instrument'] for r in manifest['records']).items())),
                     'no_note_level_fallback':True})
    manifest['raw_example_sha256']=original_examples(root,full,manifest['eval_example_labels'],output)
    manifest_path=output/'manifest.json'
    if manifest_path.is_file():
        # Do not rewrite the historical file, even if the new field is compatible.
        manifest,decoding_metadata=reconcile_manifest(json.loads(manifest_path.read_text()),manifest)
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2))
        _,decoding_metadata=reconcile_manifest(manifest,manifest)
    decoding_metadata['manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (output/'decoding_metadata.json').write_text(json.dumps(decoding_metadata,indent=2))
    (output/'train_settings.json').write_text(json.dumps(train_settings,indent=2))
    (output/'README.md').write_text(
        '# NSynth: partición compartida por instrumento\n\n'
        f'{len(full)} notas, {len(train)} entrenamiento, {len(test)} prueba.\n\n'
        'Se seleccionan primero las mismas notas del ejemplo NSynth (MIDI 48–72, hasta 150 por familia, semilla 0). '
        'Después se reserva aproximadamente 20% de instrumentos completos dentro de cada familia. '
        'Con solo 2 instrumentos se reserva 1. No se mezclan notas de un instrumento entre entrenamiento y prueba. '
        'La proporción de notas no tiene por qué ser80/20.\n\n'
        'train_settings.json usa dataset.subset existente y debe ser idéntico para todos los modelos. '
        'manifest.json fija ejemplos antes de observar resultados y guarda hashes por nota y conjunto. '
        'example_originals.npz conserva seis originales normalizados y recortados a 1s para comparar original/códec/reconstrucción. '
        'Todos los casos de prueba deben entrar en métricas, no solo los ejemplos audibles.\n')
    return {'full':full,'train':train,'test':test,'manifest':manifest,'train_settings':train_settings,
            'decoding_metadata':decoding_metadata}


if __name__=='__main__':
    study=build_study_data()
    print(json.dumps({k:study['manifest'][k] for k in ('full_count','train_count','test_count','family_counts','eval_example_labels')},indent=2))
