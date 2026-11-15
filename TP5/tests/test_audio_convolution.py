import copy
from pathlib import Path

import numpy as np
import pytest

from src.analysis.audio_convolution import configurations, validate_configurations, summarize_rows, report
from src.models.factory import build_model, load_parameters, save_parameters
from src.nn.spatial import Conv3, Upsample2
from src.options import ConfigurationError, Options

ROOT=Path(__file__).resolve().parents[1]


def test_fixed_comparison_and_rejection_of_training_confound():
    configs=configurations(ROOT)
    validate_configurations(configs)
    changed=copy.deepcopy(configs)
    changed['conv','omitted']['training']['optimizer']['learning_rate']=.002
    with pytest.raises(ConfigurationError,match='Solo pueden variar'):
        validate_configurations(changed)
    changed=copy.deepcopy(configs)
    changed['conv','complete']['dataset']['pitch_study']['reserved_targets']=[]
    with pytest.raises(ConfigurationError,match='Separación'):
        validate_configurations(changed)


def test_single_channel_rectangular_audio_model_trains_and_reloads(tmp_path):
    config=configurations(ROOT)['conv','omitted']['model']
    model=build_model(Options(config,'model'),128*64,np.random.default_rng(7),np.random.default_rng(8))
    x=np.random.default_rng(3).uniform(0,.4,(2,128*64)).astype('float32')
    terms,gradients=model.loss_and_gradients(x,x,1)
    assert all(np.isfinite(v) for v in terms.values())
    assert all(np.isfinite(g).all() for g in gradients)
    assert [layer.outputs.shape[1:] for layer in model._encoder.layers if isinstance(layer,Conv3)]==[(64,32,8),(32,16,16)]
    assert sum(isinstance(layer,Conv3) for layer in model._decoder.layers)==2
    assert sum(isinstance(layer,Upsample2) for layer in model._decoder.layers)==2
    assert model.encode(x).shape==(2,16)
    expected=model.reconstruct(x)
    assert expected.shape==x.shape and expected.dtype==np.float32
    assert expected.min()>=0 and expected.max()<=1
    path=tmp_path/'model.npz';save_parameters(model,path)
    loaded=build_model(Options(config,'model'),8192,np.random.default_rng(9),np.random.default_rng(10))
    load_parameters(loaded,path)
    np.testing.assert_array_equal(loaded.reconstruct(x),expected)


def test_invalid_pitch_is_counted_as_failure_and_joint_requires_both():
    rows=[dict(hit='True',spectral_hit='False',valid='True',variant='v00',spectral_mse='.1'),
          dict(hit='False',spectral_hit='True',valid='False',variant='v00',spectral_mse='.2')]
    metrics=summarize_rows(rows)
    assert metrics['trials']==2 and metrics['pitch_hits']==1 and metrics['joint_hits']==0
    assert metrics['pitch_rate']==.5 and metrics['valid_rate']==.5


@pytest.fixture
def protocol_root(tmp_path):
    import shutil
    (tmp_path/'configs/audio_conv').mkdir(parents=True)
    for source in (ROOT/'configs/audio_conv').glob('*.json'):
        shutil.copyfile(source,tmp_path/'configs/audio_conv'/source.name)
    # Reuse only the protected source files and recorded guard, never results.
    for path in ['src/nn/losses.py','src/models/autoencoder.py','src/models/variational.py','reports/ej2_search/loss_guard.json']:
        destination=tmp_path/path;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/path,destination)
    return tmp_path


def test_absent_results_remain_incomplete(protocol_root):
    tmp_path=protocol_root
    result=report(tmp_path)
    assert result['status']=='incomplete' and result['completed_runs']==0
    assert len(result['missing_runs'])==12
    assert result['favorable_pitch_criterion']=={'mlp':None,'conv':None}


def test_finished_run_metadata_and_negative_results_are_reported(protocol_root):
    import csv
    import json
    root=protocol_root
    config=configurations(root)['mlp','complete']
    directory=root/config['output']/'base__seed=7'
    directory.mkdir(parents=True)
    actual={k:v for k,v in config.items() if k not in ('name','output','seeds')}
    actual['seed']=7
    def write(name,value):(directory/name).write_text(json.dumps(value))
    write('config.json',actual)
    write('summary.json',{'mean_squared_error':.1})
    write('training.json',{'samples':1420,'training_seconds':1})
    write('pitch_manifest.json',dict(reserved_targets=[55,61,67],targets=[51,57,63],codec={},notes=[]))
    write('pitch_summary.json',dict(pitch_calibration_reliable=True))
    rows=[]
    for target in [51,57,63]:
        for variant in ['v00','v20','v39']:
            for method in ['vae_interpolation','spectrogram_mix','lower_neighbor','target_reconstruction']:
                for fraction in (np.linspace(0,1,11) if method=='vae_interpolation' else [.5]):
                    rows.append(dict(seed=7,target_pitch=target,variant=variant,method=method,
                                     fraction=fraction,hit='False',valid='False',spectral_hit='False',spectral_mse=.2))
    with (directory/'pitch_interpolation.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    result=report(root)
    assert result['status']=='incomplete' and result['completed_runs']==1
    assert all(row['joint_hits']==0 and row['pitch_hits']==0 and row['trials']==9 for row in result['aggregates'])
    # A missing midpoint must not silently shrink the denominator.
    with (directory/'pitch_interpolation.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows[1:])
    with pytest.raises(ConfigurationError,match='Trayectorias incompletas'):
        report(root)
