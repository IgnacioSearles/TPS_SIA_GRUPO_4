import numpy as np
import pytest

from nn import build_model
from nn.activations import Identity, Step
from nn.layers import Dense
from nn.network import resolve_activations


def activation_types(model_cfg: dict) -> list[type]:
    net = build_model(model_cfg, np.random.default_rng(0))
    return [type(layer) for layer in net.layers if not isinstance(layer, Dense)]


def test_single_activation_for_hidden_layers_and_output_activation_for_last():
    types = activation_types({"layers": [2, 3, 2, 1], "activation": "step", "output_activation": "identity"})

    assert types == [Step, Step, Identity]


def test_single_activation_is_used_for_output_when_output_activation_is_missing():
    assert activation_types({"layers": [2, 3, 1], "activation": "step"}) == [Step, Step]


def test_activation_list_sets_each_layer():
    types = activation_types({"layers": [2, 3, 2, 1], "activation": ["identity", "step", {"name": "identity"}]})

    assert types == [Identity, Step, Identity]


def test_activation_list_with_wrong_length_fails():
    with pytest.raises(ValueError, match="2 entries but the model has 3 Dense layers"):
        resolve_activations({"activation": ["step", "step"]}, n_dense=3)


def test_activation_list_with_output_activation_fails():
    with pytest.raises(ValueError, match="not both"):
        resolve_activations({"activation": ["step"], "output_activation": "identity"}, n_dense=1)
