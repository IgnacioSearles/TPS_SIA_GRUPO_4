import copy
import json
import numpy as np
import pytest

from src.analysis import nsynth_reconstruction_replication as study
from src.analysis.nsynth_cnn_protocol import make_configuration


@pytest.mark.parametrize('architecture',['mlp','conv'])
def test_replication_changes_seed_only(architecture):
    original=make_configuration({'type':'nsynth','subset':['fixed_train']},f'vae_{architecture}')
    study.validate_base(original,architecture)
    first=study.run_config(original,7);new=study.run_config(original,67)
    assert first['seed']==7 and new['seed']==67
    first.pop('seed');new.pop('seed')
    assert first==new
    for change in ('epochs','warmup','noise'):
        changed=copy.deepcopy(original)
        if change=='epochs':changed['training']['epochs']=100
        elif change=='warmup':changed['model']['kl']['warmup_epochs']=0
        else:changed['training']['noise']={'type':'gaussian','level':.1}
        with pytest.raises(ValueError):study.validate_base(changed,architecture)


def test_existing_checkpoint_is_reused_only_when_configuration_matches(tmp_path):
    config={'seed':37,'model':{'kl':{'weight':.5,'warmup_epochs':50}}}
    metadata={'epoch':75,'signature':{'config':study.sha(json.dumps(config,sort_keys=True).encode())}}
    np.savez(tmp_path/'checkpoint.npz',metadata=np.array(json.dumps(metadata)))
    state=study.progress(tmp_path,config)
    assert state=={'status':'training_pending','epoch':75}
    with pytest.raises(ValueError,match='Incompatible'):
        study.progress(tmp_path,{**config,'seed':67})


def test_nine_seed_report_does_not_invent_missing_pairs(tmp_path,monkeypatch):
    from src.data import nsynth_cnn_study
    monkeypatch.setattr(study,'assert_loss_unchanged',lambda root:None)
    monkeypatch.setattr(nsynth_cnn_study,'build_study_data',lambda root:{})
    monkeypatch.setattr(study,'evaluate_reconstruction',lambda root,a,s,data:{'architecture':a,'seed':s,'status':'queued','epoch':0})
    result=study.report(tmp_path)
    assert result['status']=='incomplete' and result['evaluated_models']==0
    assert result['expected_models']==18 and result['expected_paired_seeds']==9
    assert result['paired_seeds']==0 and len(result['runs'])==18
    assert (tmp_path/'reports/nsynth_reconstruction_replication/informe.md').exists()


def test_paired_seed_comparison_rejects_different_references():
    a={'architecture':'mlp','seed':7,'status':'evaluated','mse':.02,'signature':{'data_sha256':'same'}}
    b={**a,'architecture':'conv','mse':.01}
    assert study.paired([a])==[]
    pairs=study.paired([a,b])
    assert len(pairs)==1 and pairs[0]['difference_conv_minus_mlp']==-.01
    with pytest.raises(ValueError,match='different held-out'):
        study.paired([a,{**b,'signature':{'data_sha256':'different'}}])


def test_partial_pair_keeps_incomplete_status_and_per_note_rows(tmp_path,monkeypatch):
    from src.data import nsynth_cnn_study
    monkeypatch.setattr(study,'assert_loss_unchanged',lambda root:None)
    monkeypatch.setattr(nsynth_cnn_study,'build_study_data',lambda root:{})
    def evaluated(root,architecture,seed,data):
        if seed!=7:return {'architecture':architecture,'seed':seed,'status':'queued','epoch':0}
        folder=tmp_path/study.OUTPUT/f'vae_{architecture}/seed={seed}'
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'per_note.csv').write_text('architecture,seed,label,family,mse,valid\n'+f'{architecture},7,test,brass,0.01,True\n')
        (folder/'per_family.csv').write_text('architecture,seed,family,notes,mse,invalid\n'+f'{architecture},7,brass,1,0.01,0\n')
        return {'architecture':architecture,'seed':seed,'status':'evaluated','epoch':200,'mse':.01,
                'signature':{'data_sha256':'same'}}
    monkeypatch.setattr(study,'evaluate_reconstruction',evaluated)
    value=study.report(tmp_path)
    assert value['evaluated_models']==2 and value['paired_seeds']==1
    assert value['status']=='incomplete'
    assert (tmp_path/study.OUTPUT/'per_note.csv').exists()
