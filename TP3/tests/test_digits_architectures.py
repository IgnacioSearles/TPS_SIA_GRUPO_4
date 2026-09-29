import copy
import numpy as np
import pytest
from experiments.config import load_config
from experiments.digits.architectures import STUDY_DEFAULTS, configurations
from nn.network import build_model
from training.trainer import batches


def test_only_architecture_and_destination_change_and_batches_are_identical():
    config = copy.deepcopy(STUDY_DEFAULTS)
    resolved = load_config(experiment_defaults=config)
    runs = configurations(resolved)
    assert config == STUDY_DEFAULTS
    orders, normalized = [], []
    for run in runs:
        build_model(run['model'], np.random.default_rng(run['seed']))
        rng = np.random.default_rng(run['training']['shuffle_seed'])
        X = np.arange(30)[:, None]
        orders.append([np.concatenate([x[:, 0] for x, _ in batches(X, X, 7, rng)]) for _ in range(3)])
        run.pop('output')
        run['model'].pop('layers')
        normalized.append(run)
    assert normalized[0] == normalized[1] == normalized[2]
    np.testing.assert_array_equal(orders[0], orders[1])
    np.testing.assert_array_equal(orders[0], orders[2])


@pytest.mark.parametrize('architectures', [[], [[784,0,10]], [[784,2,9]], [[784,10]], [[784,2.5,10]], [[784,2,10],[784,2,10]]])
def test_invalid_architectures(architectures):
    config = copy.deepcopy(STUDY_DEFAULTS)
    config['study']['architectures'] = architectures
    with pytest.raises(ValueError):
        configurations(config)
