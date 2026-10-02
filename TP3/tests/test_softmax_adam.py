import numpy as np
import pytest

from nn import build
from nn.layers import Parameter
from nn.optimizers import Adam


def test_softmax_rows_are_probabilities_even_for_huge_scores():
    a = build("activation", "softmax").forward(np.array([[1000.0, 1001.0, 999.0], [-5.0, 0.0, 5.0]]))

    assert np.isfinite(a).all() and (a > 0).all()
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    assert a[0].argmax() == 1 and a[1].argmax() == 2


def test_cross_entropy_is_minus_log_of_the_true_class_probability():
    loss = build("loss", "cross_entropy")
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    y_true = np.array([[1.0, 0, 0], [0, 1.0, 0]])

    assert loss.forward(y_pred, y_true) == pytest.approx(-(np.log(0.7) + np.log(0.1)) / 2)


def test_adam_two_steps_match_hand_computation():
    p = Parameter("w", np.array([1.0, -1.0]))
    optimizer = Adam(lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)

    # Step 1: m = 0.1 g, v = 0.001 g^2; after bias correction m_hat = g and v_hat = g^2,
    # so every weight moves by lr * g / |g| = lr * sign(g), whatever the gradient's size.
    p.grad[:] = [0.5, -4.0]
    optimizer.step([p])
    np.testing.assert_allclose(p.value, [0.9, -0.9], atol=1e-7)

    # Step 2, written out from the update rule.
    p.grad[:] = [-0.5, -4.0]
    m = 0.9 * np.array([0.05, -0.4]) + 0.1 * p.grad
    v = 0.999 * np.array([0.00025, 0.016]) + 0.001 * p.grad ** 2
    expected = np.array([0.9, -0.9]) - 0.1 * (m / (1 - 0.9 ** 2)) / (np.sqrt(v / (1 - 0.999 ** 2)) + 1e-8)
    optimizer.step([p])
    np.testing.assert_allclose(p.value, expected)


def test_adam_keeps_separate_state_per_parameter_even_with_equal_names():
    p, q = Parameter("same", np.array([0.0])), Parameter("same", np.array([0.0]))
    optimizer = Adam(lr=0.1)
    p.grad[:], q.grad[:] = 1.0, 0.0

    optimizer.step([p, q])

    np.testing.assert_allclose(p.value, [-0.1], atol=1e-7)
    np.testing.assert_array_equal(q.value, [0.0])


@pytest.mark.parametrize("kwargs", [{"lr": 0}, {"lr": float("nan")}, {"beta1": 1}, {"beta2": -0.1}, {"epsilon": 0}])
def test_invalid_adam_configuration(kwargs):
    with pytest.raises(ValueError):
        Adam(**kwargs)
