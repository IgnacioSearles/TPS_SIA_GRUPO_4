"""Ampliación de entrenamiento sin tocar las referencias del estudio."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.art_capacity import read_json,file_hash,load_probe
from src.data.art_search import configuration,assert_loss_unchanged
from src.data.factory import build_dataset
from src.options import Options,ConfigurationError


def fingerprint(samples):
    return hashlib.sha256(samples.tobytes()).hexdigest()


def prepare_expanded(root):
    root=Path(root);assert_loss_unchanged(root)
    settings=configuration(root,'cnn8k_refined25')['dataset']
    expanded=build_dataset(Options(settings,'dataset'),root)
    old=build_dataset(Options({**settings,'train_count':4096},'dataset'),root)
    if not np.array_equal(old.samples,expanded.samples[:4096]):
        raise ConfigurationError('Cambió el prefijo de entrenamiento al ampliar los datos')
    hashes={hashlib.sha256(row.tobytes()).digest() for row in expanded.samples}
    overlaps={}
    # Comparar hashes garantiza separación; no se evalúa ningún modelo en final.
    for split in ('validation','reserved_final'):
        references,_=load_probe(root/'results/art_search/data',split)
        overlaps[split]=len(hashes.intersection(hashlib.sha256(r.tobytes()).digest() for r in references))
    if any(overlaps.values()):raise ConfigurationError('La ampliación incorpora referencias reservadas')
    report={'train_count':len(expanded),'data_seed':settings['data_seed'],
            'samples_sha256':fingerprint(expanded.samples),
            'generator_sha256':file_hash(root/'src/data/art.py'),
            'original_data_manifest_sha256':file_hash(root/'results/art_search/data/manifest.json'),
            'old_4096_training_samples_preserved':True,'reference_overlaps':overlaps,
            'family_counts':dict(Counter(r['family'] for r in expanded.generative_study.train_records)),
            'old_internal_test_now_in_training':bool(np.array_equal(
                old.generative_study.test_samples,expanded.samples[4096:4160])),
            'interpretation':'Old internal test was never the search validation. External validation/final remain excluded.'}
    path=root/'results/art_search/expanded_data_manifest.json'
    if path.exists() and read_json(path)!=report:raise ConfigurationError('Cambió la ampliación registrada')
    path.write_text(json.dumps(report,indent=2))
    return report


def validate_expanded_training(root,dataset):
    manifest=read_json(root/'results/art_search/expanded_data_manifest.json')
    if (manifest['samples_sha256']!=fingerprint(dataset.samples)
            or manifest['generator_sha256']!=file_hash(root/'src/data/art.py')):
        raise ConfigurationError('Los datos ampliados no coinciden con el manifiesto')


if __name__=='__main__':
    print(json.dumps(prepare_expanded(Path(__file__).resolve().parents[2]),indent=2))
