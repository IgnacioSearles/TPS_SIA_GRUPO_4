import json

import pytest

from experiments.config import BASE_DEFAULTS, deep_merge, load_config


def test_deep_merge_merges_nested_dicts_and_replaces_other_values():
    base = {"optimizer": {"name": "sgd", "lr": 0.01}, "callbacks": ["progress"], "seed": 1}
    override = {"optimizer": {"lr": 0.1}, "callbacks": []}

    merged = deep_merge(base, override)

    assert merged == {"optimizer": {"name": "sgd", "lr": 0.1}, "callbacks": [], "seed": 1}
    assert base["optimizer"]["lr"] == 0.01, "deep_merge must not mutate its inputs"


def test_load_config_layers_file_over_experiment_defaults_over_base(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"optimizer": {"lr": 0.5}}))

    config = load_config(path, experiment_defaults={"optimizer": {"lr": 0.1}, "seed": 7})

    assert config["optimizer"] == {"name": "sgd", "lr": 0.5}
    assert config["seed"] == 7
    assert config["loss"] == BASE_DEFAULTS["loss"]


def test_load_config_without_file_returns_defaults():
    assert load_config() == BASE_DEFAULTS


def test_load_config_missing_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")
