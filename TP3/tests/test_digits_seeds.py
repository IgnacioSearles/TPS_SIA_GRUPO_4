import copy
import json
import numpy as np
import pytest
from experiments.config import load_config
from experiments.digits.seed_study import STUDY_DEFAULTS, aggregate, configurations, reuse


def test_paired_seed_grid_preserves_split_and_changes_training_randomness():
    config = load_config(experiment_defaults=copy.deepcopy(STUDY_DEFAULTS))
    original = copy.deepcopy(config)
    runs = configurations(config)
    assert config == original and len(runs) == 15
    assert len({r['output']['directory'] for r in runs}) == 15
    assert {r['data']['split_seed'] for r in runs} == {42}
    for seed in [42,7,21,84,123]:
        group = [r for r in runs if r['seed'] == seed]
        assert len(group) == 3
        assert {r['training']['shuffle_seed'] for r in group} == {seed}
        assert all(r['model'] == group[0]['model'] for r in group)


def test_aggregation_uses_sample_std_and_mean_not_best_seed():
    rows = []
    for value, epoch in [(.9,10),(.92,20)]:
        rows.append(dict(candidate='sgd', optimizer='sgd', momentum=0, learning_rate=.1,
                         best_epoch=epoch, validation_accuracy=value, validation_loss=1-value,
                         validation_macro_f1=value, validation_recall_5=value))
    result = aggregate(rows)[0]
    assert result['validation_accuracy_mean'] == pytest.approx(.91)
    assert result['validation_accuracy_std'] == pytest.approx(np.sqrt(.0002))
    assert result['median_best_epoch'] == 15


@pytest.mark.parametrize('seeds', [[42], [42,42], [-1,42], [1.5,42]])
def test_invalid_seeds_fail(seeds):
    config = load_config(experiment_defaults=copy.deepcopy(STUDY_DEFAULTS))
    config['study']['seeds'] = seeds
    with pytest.raises(ValueError): configurations(config)


def test_reuse_rejects_wrong_dataset_or_configuration(tmp_path):
    config = configurations(load_config(experiment_defaults=copy.deepcopy(STUDY_DEFAULTS)))[0]
    source = tmp_path/'source';source.mkdir()
    for name in ['history.csv','best_weights.npz','split_indices.npz','validation_predictions.csv']:
        (source/name).touch()
    (source/'config.json').write_text(json.dumps(config))
    (source/'summary.json').write_text(json.dumps({'dataset_sha256':'old'}))
    assert reuse(config,[source],'new') == (None,None)
    changed = copy.deepcopy(config);changed['training']['epochs'] = 100
    assert reuse(changed,[source],'old') == (None,None)
