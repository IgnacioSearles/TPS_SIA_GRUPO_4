import copy
import numpy as np
import pytest
from nn.layers import Parameter
from nn.optimizers import SGD, Momentum
from experiments.config import load_config
from experiments.digits.optimizer_study import STUDY_DEFAULTS, configurations


def test_momentum_two_steps_with_changing_gradient_and_independent_parameters():
    p = Parameter('same_name', np.array([1., -1.]))
    q = Parameter('same_name', np.array([3., 4.]))
    p.grad[:] = [.5, -.25]
    q.grad[:] = 0
    optimizer = Momentum(lr=.1, momentum=.9)
    optimizer.step([p, q])
    np.testing.assert_allclose(p.value, [.95, -.975])
    p.grad[:] = [-.25, .5]
    optimizer.step([p, q])
    np.testing.assert_allclose(p.value, [.93, -1.0025])
    np.testing.assert_array_equal(q.value, [3., 4.])


def test_zero_momentum_equals_sgd_over_several_updates():
    p = Parameter('w', np.array([1., 2.]))
    q = Parameter('w', p.value.copy())
    a, b = SGD(.1), Momentum(.1, 0)
    for gradient in [[1., -2.], [-3., 4.], [0., 0.]]:
        p.grad[:] = q.grad[:] = gradient
        a.step([p]); b.step([q])
        np.testing.assert_array_equal(p.value, q.value)


@pytest.mark.parametrize('kwargs', [{'lr':0}, {'lr':float('nan')}, {'lr':float('inf')}, {'momentum':1}, {'momentum':-.1}, {'momentum':float('nan')}])
def test_invalid_momentum_configuration(kwargs):
    with pytest.raises(ValueError):
        Momentum(**kwargs)


def test_grid_changes_only_optimizer_and_destination():
    config = load_config(experiment_defaults=STUDY_DEFAULTS)
    original = copy.deepcopy(config)
    runs = configurations(config)
    assert config == original and len(runs) == 6
    assert [r['optimizer']['lr'] for r in runs] == [.01,.1,.01,.1,.01,.1]
    assert [r['optimizer'].get('momentum', 0) for r in runs] == [0,0,.5,.5,.9,.9]
    normalized = []
    for r in runs:
        r.pop('optimizer');r.pop('output');normalized.append(r)
    assert all(r == normalized[0] for r in normalized)
