import hashlib
import json
import numpy as np
import pytest

from src.analysis import audio_conv_beta as study


def test_freeze_extracts_only_weights_and_preserves_source(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    source=tmp_path/'checkpoint.npz'; destination=tmp_path/'frozen/model.npz'
    weights=np.arange(12,dtype=np.float32).reshape(3,4)
    np.savez_compressed(source,parameter_0=weights,metadata=np.array(json.dumps({'epoch':160,'elapsed':100})),state_0=np.ones(3))
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    record=study.freeze_checkpoint(source,destination,7)
    with np.load(destination,allow_pickle=False) as saved:
        assert saved.files==['parameter_0']
        np.testing.assert_array_equal(saved['parameter_0'],weights)
        assert saved['parameter_0'].dtype==np.float32
    assert record['parent_epoch']==160
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    with pytest.raises(ValueError,match='overwrite'):
        study.freeze_checkpoint(source,destination,7)


def test_freeze_rejects_early_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    source=tmp_path/'checkpoint.npz'
    np.savez(source,parameter_0=np.ones(2),metadata=np.array(json.dumps({'epoch':20,'elapsed':5})))
    with pytest.raises(ValueError,match='epoch'):
        study.freeze_checkpoint(source,tmp_path/'model.npz',7)
    assert not (tmp_path/'model.npz').exists()


def test_report_retains_valid_incomplete_result_without_finished_runs(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'assert_loss_unchanged',lambda root:None)
    folder=tmp_path/'results/audio_conv_beta_pilot';folder.mkdir(parents=True)
    (folder/'protocol.json').write_text(json.dumps({'parents':[]}))
    result=study.report(tmp_path)
    assert result['completed_runs']==0
    assert result['status']=='incomplete'
    assert result['codec_reliable'] is None
    assert len(result['missing'])==9
    assert (tmp_path/'reports/audio_conv_beta_pilot/informe.md').exists()
