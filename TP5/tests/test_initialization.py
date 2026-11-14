import numpy as np
import pytest
import json

from src.models.factory import build_model, save_parameters
from src.options import Options, ConfigurationError
from src.training.initialization import initialize_from_parameters


def _assert_preserved_objective(source,target,x,offset):
    noise=np.random.default_rng(6).normal(size=(len(x),source.latent_size))
    before,old_gradients=source.objective(x,x,noise,.02)
    after,new_gradients=target.objective(x,x,noise,.02)
    assert after['loss']==pytest.approx(before['loss'],abs=1e-12)
    mapped=new_gradients[:offset]+new_gradients[offset+2:]
    for a,b in zip(old_gradients,mapped):np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
    assert np.max(np.abs(new_gradients[offset]))>1e-8


def model(seed):
    return build_model(Options({'type':'variational','hidden_layers':[4],'latent_size':2},'model'),
                       6,np.random.default_rng(seed),np.random.default_rng(2))


def test_warm_start_loads_exact_parameters_and_records_provenance(tmp_path):
    old, fresh = model(1), model(3)
    path = tmp_path/'source_7.npz'
    save_parameters(old,path)
    report = initialize_from_parameters(fresh,'source_{seed}.npz',tmp_path,7)
    for expected, actual in zip(old.parameters(),fresh.parameters()):
        np.testing.assert_array_equal(actual,expected)
    assert report['optimizer_state_restored'] is False
    assert report['rng_states_restored'] is False
    assert len(report['sha256']) == 64
    assert initialize_from_parameters(fresh,None,tmp_path,7) is None


@pytest.mark.parametrize('specification', ['',False,{},'missing_{seed}.npz'])
def test_missing_or_invalid_source_fails_before_training(tmp_path,specification):
    with pytest.raises(ConfigurationError):
        initialize_from_parameters(model(1),specification,tmp_path,7)


def test_saved_warm_started_run_can_be_evaluated_without_its_initial_file(tmp_path,monkeypatch):
    from src.experiments import runner
    monkeypatch.setattr(runner,'PROJECT_ROOT',tmp_path)
    (tmp_path/'data').mkdir()
    (tmp_path/'data/font.h').write_text('   {0x04,0x04,0x02,0x00,0x00,0x00,0x00}, // 0x60, `\n')
    spec={'type':'variational','hidden_layers':[4],'latent_size':2,'loss':'mse'}
    initial=build_model(Options(spec,'model'),35,np.random.default_rng(1),np.random.default_rng(2))
    source=tmp_path/'source_7.npz';save_parameters(initial,source)
    config={'seed':7,'model':spec,'initial_parameters':'source_{seed}.npz','evaluation':{},
            'training':{'epochs':1,'optimizer':{'type':'adam','learning_rate':.001}}}
    path=tmp_path/'warm.json';path.write_text(json.dumps(config))
    output=runner.run_experiment(path)
    directory=output/'base__seed=7'
    trained,_=runner.load_run(directory)
    expected=[p.copy() for p in trained.parameters()]
    summary=json.loads((directory/'training.json').read_text())
    assert summary['initial_parameters']['sha256']
    source.unlink()
    runner.run_experiment(path,evaluate_only=True)
    loaded,_=runner.load_run(directory)
    for a,b in zip(expected,loaded.parameters()):np.testing.assert_array_equal(a,b)


def test_identity_transfer_preserves_trained_function_and_rejects_random_layer(tmp_path):
    spec={'type':'variational','architecture':'conv','image_shape':[8,8,3],
          'conv_channels':[2,3],'hidden_layers':[4],'latent_size':2,'activation':'relu','loss':'mse'}
    def build(extra,seed):
        return build_model(Options({**spec,**extra},'model'),192,np.random.default_rng(seed),np.random.default_rng(2))
    source=build({},1)
    path=tmp_path/'trained.npz';save_parameters(source,path)
    target=build({'conv_refinement':True,'conv_refinement_init':'identity'},5)
    initialize_from_parameters(target,{'path':'trained.npz','transfer':'identity_refinement'},tmp_path,7)
    x=np.random.default_rng(3).random((3,192));z=np.random.default_rng(4).normal(size=(3,2))
    np.testing.assert_array_equal(target.encode(x),source.encode(x))
    np.testing.assert_array_equal(target.decode(z),source.decode(z))
    _assert_preserved_objective(source,target,x,len(target.parameters())-4)
    random=build({'conv_refinement':True},5)
    with pytest.raises(ConfigurationError,match='identidad'):
        initialize_from_parameters(random,{'path':'trained.npz','transfer':'identity_refinement'},tmp_path,7)


@pytest.mark.parametrize('spatial',[False,True])
def test_dense_identity_insertion_preserves_function(tmp_path,spatial):
    spec={'type':'variational','hidden_layers':[4,3],'latent_size':2,'activation':'relu','loss':'mse'}
    if spatial:spec.update(architecture='conv',image_shape=[8,8,3],conv_channels=[2,3],conv_refinement=True)
    count=192 if spatial else 6
    def build(extra,seed):
        return build_model(Options({**spec,**extra},'model'),count,np.random.default_rng(seed),np.random.default_rng(2))
    source=build({},1)
    for p in source.parameters():p+=.01
    path=tmp_path/'dense.npz';save_parameters(source,path)
    target=build({'decoder_hidden_layers':[3,4,4]},5)
    initialize_from_parameters(target,{'path':'dense.npz','transfer':'identity_dense','layer_index':2},tmp_path,7)
    x=np.random.default_rng(3).random((3,count));z=np.random.default_rng(4).normal(size=(3,2))
    np.testing.assert_array_equal(target.encode(x),source.encode(x))
    np.testing.assert_array_equal(target.decode(z),source.decode(z))
    _assert_preserved_objective(source,target,x,len(target._encoder.parameters())+4)
    invalid=build({'decoder_hidden_layers':[3,4,5]},5)
    with pytest.raises(ConfigurationError,match='dimensiones iguales'):
        initialize_from_parameters(invalid,{'path':'dense.npz','transfer':'identity_dense','layer_index':2},tmp_path,7)


def test_second_spatial_refinement_preserves_learned_first_block(tmp_path):
    spec={'type':'variational','architecture':'conv','image_shape':[8,8,3],'conv_channels':[2,3],
          'hidden_layers':[4],'latent_size':2,'activation':'relu','loss':'mse',
          'conv_refinement':True,'conv_refinement_init':'identity'}
    def build(depth,seed):
        return build_model(Options({**spec,'conv_refinement_layers':depth},'model'),192,
                           np.random.default_rng(seed),np.random.default_rng(2))
    source=build(1,1)
    for p in source.parameters():p+=.01
    path=tmp_path/'one.npz';save_parameters(source,path)
    target=build(2,5)
    initialize_from_parameters(target,{'path':'one.npz','transfer':'identity_refinement'},tmp_path,7)
    x=np.random.default_rng(3).random((3,192));z=np.random.default_rng(4).normal(size=(3,2))
    np.testing.assert_array_equal(target.encode(x),source.encode(x))
    np.testing.assert_array_equal(target.decode(z),source.decode(z))
    _assert_preserved_objective(source,target,x,len(target.parameters())-4)
