import json

import numpy as np
import pandas as pd

from experiments.config import deep_merge, load_config
from experiments.digits.l2_study import STUDY_DEFAULTS, run_study
from experiments.digits.relu_depth import DEFAULTS as RELU_DEFAULTS, gradient_profile


def tiny_digits(tmp_path) -> str:
    """30 synthetic 'images': digit d lights up pixel d."""
    labels = np.tile([0, 5, 9], 10)
    X = np.zeros((len(labels), 784))
    X[np.arange(len(labels)), labels] = 1
    path = tmp_path / "digits.csv"
    pd.DataFrame({"label": labels, "image": [json.dumps(row.tolist()) for row in X]}).to_csv(path, index=False)
    return str(path)


def test_l2_study_writes_one_row_per_weight_decay_and_strong_decay_shrinks_weights(tmp_path):
    config = load_config(experiment_defaults=STUDY_DEFAULTS)
    config["data"]["path"] = tiny_digits(tmp_path)
    config["training"] = {"epochs": 3, "batch_size": 8}
    config["study"] = {"weight_decays": [0.0, 0.5], "seeds": [42]}
    config["output"] = {"directory": str(tmp_path / "l2")}

    aggregate = run_study(config)

    assert aggregate["weight_decay"].tolist() == [0.0, 0.5]
    norms = aggregate.set_index("weight_decay")["squared_weight_norm_mean"]
    assert norms[0.5] < norms[0.0]
    assert (tmp_path / "l2" / "l2_comparison.png").is_file()


def test_gradient_profile_has_one_row_per_layer_and_sigmoid_gradients_vanish_toward_the_input(tmp_path):
    config = deep_merge(RELU_DEFAULTS, {
        "data": {"path": tiny_digits(tmp_path)},
        "gradient_profile": {"depth": 6, "width": 8, "batch_size": 20, "seeds": [42]},
    })

    profile = gradient_profile(config)

    assert len(profile) == 3 * 1 * (6 + 1)  # configurations x seeds x Dense layers
    first_layer = profile[profile["layer"] == 1].set_index("configuration")["mean_abs_gradient"]
    assert first_layer["sigmoide + Xavier"] < first_layer["tanh + Xavier"]
    assert first_layer["sigmoide + Xavier"] < first_layer["ReLU + He"]
