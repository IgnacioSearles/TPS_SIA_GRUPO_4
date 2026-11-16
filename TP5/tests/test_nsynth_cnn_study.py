import numpy as np
import pytest

from src.data.dataset import Dataset
from src.data.nsynth_cnn_study import instrument_split,digest_samples,parse_label,reconcile_manifest
from src.options import ConfigurationError


def toy(instruments=5):
    labels=[f'{family}_acoustic_{instrument:03d}-{pitch:03d}-{velocity:03d}'
            for family in ('guitar','flute') for instrument in range(instruments)
            for pitch in (48,60) for velocity in (25,100)]
    return Dataset(np.arange(len(labels)*4).reshape(-1,4)/1000,labels,
                   [label.split('_')[0] for label in labels],(2,2),False)


def test_split_groups_all_pitches_velocities_and_is_reproducible():
    dataset=toy()
    split=instrument_split(dataset)
    assert split==instrument_split(dataset)
    assignment={}
    for row in split['records']:
        if row['instrument'] in assignment:assert assignment[row['instrument']]==row['split']
        assignment[row['instrument']]=row['split']
    assert all(r['train_instruments']==4 and r['test_instruments']==1 for r in split['family_counts'])
    assert set(split['train_labels']).isdisjoint(split['test_labels'])
    assert len(split['train_labels'])+len(split['test_labels'])==len(dataset)


def test_split_independent_of_dataset_row_order_and_values():
    dataset=toy()
    permuted=dataset.select(list(reversed(dataset.labels)))
    assert instrument_split(dataset)==instrument_split(permuted)
    changed=Dataset(np.zeros_like(dataset.samples),dataset.labels,dataset.groups,(2,2),False)
    assert instrument_split(dataset)['test_labels']==instrument_split(changed)['test_labels']


def test_requires_two_instruments_and_keeps_one_each_when_two():
    with pytest.raises(ConfigurationError,match='No se cambia a split por notas'):
        instrument_split(toy(1))
    assert all(r['train_instruments']==1 and r['test_instruments']==1 for r in instrument_split(toy(2))['family_counts'])


def test_encoded_hash_canonical_and_metadata_exact():
    values=np.array([.1,.2],dtype=np.float32)
    assert digest_samples(values)==digest_samples(values.astype(np.float64))
    parsed=parse_label('keyboard_electronic_007-060-100')
    assert parsed['instrument']=='keyboard_electronic_007'
    assert parsed['pitch']==60 and parsed['velocity']==100
    with pytest.raises(ConfigurationError):parse_label('invalid')


def test_legacy_manifest_phase_change_preserves_provenance():
    previous={'codec':{'fft_size':1024,'griffin_lim_iterations':60},'train_labels':['a'],'train_sha256':'hash'}
    current={**previous,'codec':{**previous['codec'],'phase_initialization':'pghi'}}
    stored,metadata=reconcile_manifest(previous,current)
    assert stored is previous and 'phase_initialization' not in stored['codec']
    assert metadata['stored_phase_initialization']=='random'
    assert metadata['current_phase_initialization']=='pghi'
    assert metadata['encoding_selection_split_identical']


@pytest.mark.parametrize('change',('encoding','iterations','split','hash'))
def test_compatibility_rejects_every_other_change(change):
    from copy import deepcopy
    previous={'codec':{'fft_size':1024,'griffin_lim_iterations':60},'train_labels':['a'],'train_sha256':'hash'}
    current=deepcopy(previous);current['codec']['phase_initialization']='pghi'
    if change=='encoding':current['codec']['fft_size']=512
    elif change=='iterations':current['codec']['griffin_lim_iterations']=30
    elif change=='split':current['train_labels']=['b']
    else:current['train_sha256']='new hash'
    with pytest.raises(ConfigurationError,match='solo se admite phase_initialization'):
        reconcile_manifest(previous,current)
