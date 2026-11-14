import json

import numpy as np
import pytest

from src.models.factory import build_model
from src.nn.optimizers import build_optimizer
from src.options import Options,ConfigurationError
from src.training.checkpoint import Checkpoint
from src.training.trainer import Trainer,TrainingPlan


class Corruption:
    def apply(self,values,rng):return values+rng.normal(0,.03,values.shape)


def setup(model_type,optimizer_name,precision,seed=1):
    rngs={key:np.random.default_rng(seed+i) for i,key in enumerate(('shuffle','corruption','sampling'))}
    model=build_model(Options({'type':model_type,'hidden_layers':[5],'latent_size':2,'loss':'mse',
                               'activation':'tanh','precision':precision,'kl':{'weight':.5,'warmup_epochs':5}}
                              if model_type=='variational' else
                              {'type':model_type,'hidden_layers':[5],'latent_size':2,'loss':'mse',
                               'activation':'tanh','precision':precision},'model'),
                      6,np.random.default_rng(seed+4),rngs['sampling'])
    optimizer=build_optimizer(Options({'type':optimizer_name,'learning_rate':.002},'optimizer'))
    plan=TrainingPlan(epochs=12,batch_size=4,log_every=3,optimizer=optimizer,corruption=Corruption())
    trainer=Trainer(model,plan,lambda _: {},rngs['shuffle'],rngs['corruption'])
    return model,optimizer,rngs,trainer


@pytest.mark.parametrize('model_type',['autoencoder','variational'])
@pytest.mark.parametrize('optimizer_name',['sgd','momentum','rmsprop','adam'])
@pytest.mark.parametrize('precision',['float32','float64'])
def test_resume_is_identical_to_uninterrupted_training(tmp_path,model_type,optimizer_name,precision):
    samples=np.random.default_rng(20).random((12,6))
    full,_,full_rngs,full_trainer=setup(model_type,optimizer_name,precision)
    expected_history=full_trainer.train(samples)
    partial,opt,rngs,trainer=setup(model_type,optimizer_name,precision)
    path=tmp_path/'checkpoint.npz'
    signature={'test':model_type,'optimizer':optimizer_name,'precision':precision}
    store=Checkpoint(path,partial,opt,rngs,signature,{'source':'kept'})
    def interrupt(epoch,history):
        if epoch==5:
            store.save(epoch,history,2.5,interval=5)
            raise InterruptedError('simulated shutdown')
    with pytest.raises(InterruptedError):trainer.train(samples,epoch_callback=interrupt)
    resumed,opt,rngs,trainer=setup(model_type,optimizer_name,precision,seed=50)
    restored=Checkpoint(path,resumed,opt,rngs,signature)
    state=restored.load()
    assert state['epoch']==5 and restored.extra=={'source':'kept'}
    actual_history=trainer.train(samples,start_epoch=state['epoch'],history=state['history'])
    assert actual_history==expected_history
    for expected,actual in zip(full.parameters(),resumed.parameters()):np.testing.assert_array_equal(expected,actual)
    for key in rngs:assert json.dumps(rngs[key].bit_generator.state)==json.dumps(full_rngs[key].bit_generator.state)


def test_incompatible_checkpoint_does_not_modify_parameters(tmp_path):
    model,opt,rngs,_=setup('variational','adam','float32')
    path=tmp_path/'checkpoint.npz'
    Checkpoint(path,model,opt,rngs,{'data':'original'}).save(0,[],0,10)
    target,opt,rngs,_=setup('variational','adam','float32',seed=50)
    before=[p.copy() for p in target.parameters()]
    with pytest.raises(ConfigurationError,match='incompatible'):
        Checkpoint(path,target,opt,rngs,{'data':'different'}).load()
    for a,b in zip(before,target.parameters()):np.testing.assert_array_equal(a,b)


def test_failed_save_preserves_previous_checkpoint(tmp_path,monkeypatch):
    model,opt,rngs,_=setup('variational','adam','float32')
    path=tmp_path/'checkpoint.npz';store=Checkpoint(path,model,opt,rngs,{})
    store.save(0,[],0,10);before=path.read_bytes()
    def fail(*args,**kwargs):raise OSError('simulated disk failure')
    monkeypatch.setattr(np,'savez_compressed',fail)
    with pytest.raises(OSError):store.save(1,[],1,10)
    assert path.read_bytes()==before


def test_runner_resumes_without_parent_weights_and_keeps_history(tmp_path,monkeypatch):
    from src.experiments import runner
    from src.models.factory import save_parameters
    monkeypatch.setattr(runner,'PROJECT_ROOT',tmp_path)
    (tmp_path/'data').mkdir()
    (tmp_path/'data/font.h').write_text('   {0x04,0x04,0x02,0x00,0x00,0x00,0x00}, // 0x60, `\n')
    config={'seed':7,'model':{'type':'variational','hidden_layers':[4],'latent_size':2,'loss':'mse',
                            'precision':'float32','kl':{'weight':.5,'warmup_epochs':3}},
            'training':{'epochs':5,'batch_size':1,'log_every':1,
                        'optimizer':{'type':'adam','learning_rate':.001}},'evaluation':{}}
    initial=runner._prepare(config).model
    source=tmp_path/'parent.npz';save_parameters(initial,source)
    config['initial_parameters']='parent.npz'
    path=tmp_path/'reference.json';path.write_text(json.dumps({**config,'output':'results/reference'}))
    expected=runner.run_experiment(path)/'base__seed=7'
    source_save=Checkpoint.save
    def stop(self,epoch,history,elapsed,interval):
        source_save(self,epoch,history,elapsed,interval)
        if epoch==2:raise InterruptedError('simulated reboot')
    monkeypatch.setattr(Checkpoint,'save',stop)
    path=tmp_path/'paused.json';path.write_text(json.dumps({**config,'output':'results/paused'}))
    with pytest.raises(InterruptedError):runner.run_experiment(path,checkpoint_every=2)
    source.unlink()
    monkeypatch.setattr(Checkpoint,'save',source_save)
    resumed=runner.run_experiment(path,checkpoint_every=2)/'base__seed=7'
    assert (expected/'history.csv').read_bytes()==(resumed/'history.csv').read_bytes()
    a,_=runner.load_run(expected);b,_=runner.load_run(resumed)
    for x,y in zip(a.parameters(),b.parameters()):np.testing.assert_array_equal(x,y)
    assert json.loads((resumed/'training.json').read_text())['initial_parameters']['sha256']
    runner.run_experiment(path,checkpoint_every=2,evaluate_only=True)
