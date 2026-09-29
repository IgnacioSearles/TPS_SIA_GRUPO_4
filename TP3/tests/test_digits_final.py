import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.config import load_config
from experiments.digits import final_evaluation as final


def write_digits(path, labels):
    images = np.zeros((len(labels),784))
    images[np.arange(len(labels)),labels] = 1
    pd.DataFrame({'label':labels,'image':[json.dumps(row.tolist()) for row in images]}).to_csv(path,index=False)


def test_final_uses_all_train_rows_and_loads_test_only_after_weights_saved(tmp_path,monkeypatch):
    training = tmp_path/'digits.csv';test = tmp_path/'digits_test.csv'
    write_digits(training,np.tile([0,1],10));write_digits(test,np.array([0,1,8]))
    config = load_config(experiment_defaults=copy.deepcopy(final.FINAL_DEFAULTS))
    config['data'] = {'path':str(training),'test_path':str(test)}
    config['training']['epochs'] = 2
    output = tmp_path/'results';config['output'] = {'directory':str(output)}
    original = final.load_digit_arrays
    reads=[]
    def checked_loader(path):
        reads.append(Path(path).name)
        assert (output/'protocol.json').is_file()
        if Path(path).name == 'digits_test.csv':
            assert (output/'final_weights.npz').is_file()
            assert len(pd.read_csv(output/'history.csv')) == 2
        return original(path)
    monkeypatch.setattr(final,'load_digit_arrays',checked_loader)
    summary = final.run_final(config)
    assert reads == ['digits.csv','digits_test.csv']
    assert summary['train_samples'] == 20 and summary['test_samples'] == 3
    assert summary['test']['per_class'][8]['support'] == 1
    assert summary['diagnostic_test_classes_seen_in_training']['samples'] == 2
    assert summary['exact_test_images_also_in_training'] == 2
    assert summary['weights_reload_verified'] and summary['test_model_evaluations'] == 1
    assert 'validation_loss' not in pd.read_csv(output/'history.csv').columns
    with pytest.raises(FileExistsError): final.run_final(config)
    assert reads == ['digits.csv','digits_test.csv']


def test_final_rejects_swapped_data_before_loading(tmp_path,monkeypatch):
    config = load_config(experiment_defaults=copy.deepcopy(final.FINAL_DEFAULTS))
    config['data']['path'] = 'datasets/digits_test.csv'
    def forbidden(path):
        raise AssertionError('Must not load data with invalid roles')
    monkeypatch.setattr(final,'load_digit_arrays',forbidden)
    with pytest.raises(ValueError,match='separate'): final.run_final(config)
