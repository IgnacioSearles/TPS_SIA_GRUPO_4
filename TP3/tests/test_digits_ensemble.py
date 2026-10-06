import json

import numpy as np
import pandas as pd

from experiments.digits.ensemble import class_probabilities, ensemble_accuracy, resolve_members, select_winner
from nn.network import build_model, save_weights


def test_sigmoid_outputs_become_a_distribution():
    net = build_model({"layers": [4, 3], "activation": "identity", "output_activation": "sigmoid"},
                      np.random.default_rng(0))

    probabilities = class_probabilities(net, np.random.default_rng(1).normal(size=(5, 4)))

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_averaging_lets_two_confident_networks_outvote_one():
    y = np.array([0])
    right = np.array([[0.6, 0.4]])
    wrong = np.array([[0.1, 0.9]])

    assert ensemble_accuracy([right, right, wrong], y) == 0.0  # 1.3 / 3 < 1.7 / 3
    assert ensemble_accuracy([right, right, np.array([[0.45, 0.55]])], y) == 1.0


def write_run(directory, seed):
    spec = {"layers": [4, 2], "activation": "identity", "output_activation": "sigmoid"}
    directory.mkdir(parents=True)
    save_weights(build_model(spec, np.random.default_rng(seed)), directory / "best_weights.npz")
    protocol = {"architecture": [4, 2], "config": {"model": spec, "files": ["a.csv"],
                                                    "validation_ratio": 0.2, "split_seed": 1}}
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (directory / "summary.json").write_text(json.dumps({"best": {"accuracy": 0.5}}))


def test_identical_weights_count_once(tmp_path):
    write_run(tmp_path / "study_a" / "seed_1", seed=1)
    write_run(tmp_path / "study_b" / "seed_1", seed=1)
    write_run(tmp_path / "study_b" / "seed_2", seed=2)

    members = resolve_members([str(tmp_path / "study_*" / "seed_*")])

    assert len(members) == 2


def test_winner_is_best_on_validation_and_smaller_on_a_tie():
    ensembles = pd.DataFrame({"ensemble": ["big", "small", "worse"], "networks": [12, 5, 3],
                              "validation_accuracy": [0.991, 0.991, 0.989]})

    assert select_winner(ensembles) == "small"
