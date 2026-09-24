"""Config loading: every key is optional and falls back to layered defaults.

Precedence (later wins): BASE_DEFAULTS < experiment defaults < JSON file.
Nested dicts are merged key by key; any other value (lists included) is replaced.
"""

import argparse
import json
from pathlib import Path
from typing import Any

Config = dict[str, Any]

BASE_DEFAULTS: Config = {
    "seed": 42,
    "model": {"activation": "identity", "initializer": "uniform"},
    "loss": "mse",
    "optimizer": {"name": "sgd", "lr": 0.01},
    "training": {"epochs": 100, "batch_size": 1},
    "callbacks": [{"name": "progress", "every": 10}],
}


def deep_merge(base: Config, override: Config) -> Config:
    """Return a new dict with `override` merged on top of `base`, recursing into nested dicts."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_json(path: str | Path) -> Config:
    """Read a JSON config file, failing with a clear message if it is missing or not an object."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as file:
        content = json.load(file)
    if not isinstance(content, dict):
        raise ValueError(f"Config {path} must contain a JSON object, got {type(content).__name__}")
    return content


def load_config(path: str | Path | None = None, experiment_defaults: Config | None = None) -> Config:
    """Build the final config from the defaults and the optional JSON file at `path`."""
    config = deep_merge(BASE_DEFAULTS, experiment_defaults or {})
    if path is not None:
        config = deep_merge(config, read_json(path))
    return config


def config_from_cli(experiment_defaults: Config | None = None) -> Config:
    """Load the config whose path is the (optional) first command-line argument."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", help="path to a JSON config; omitted keys use defaults")
    args = parser.parse_args()
    return load_config(args.config, experiment_defaults)
