"""Expansión de una configuración de experimento en corridas individuales.

`grid` asocia rutas con puntos (por ejemplo "training.optimizer.type") a
listas de valores; se corre el producto cartesiano de todas ellas, repetido
para cada semilla de `seeds`.
"""

from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.evaluation.context import safe_filename
from src.options import ConfigurationError

GRID_KEY = "grid"
SEEDS_KEY = "seeds"
SEED_KEY = "seed"
DEFAULT_SEED = 0
BASELINE_CONFIGURATION = "base"


@dataclass(frozen=True)
class RunSpecification:
    """Una corrida concreta: configuración sin `grid` ni `seeds`, con `seed` fija."""

    configuration: str
    seed: int
    grid_values: dict[str, Any]
    config: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.configuration}__seed={self.seed}"


def expand_runs(experiment: Mapping[str, Any]) -> list[RunSpecification]:
    base = {key: value for key, value in experiment.items() if key not in (GRID_KEY, SEEDS_KEY)}
    grid = experiment.get(GRID_KEY, {})
    seeds = _seeds(experiment)
    _validate_grid(grid)

    runs = []
    for combination in itertools.product(*grid.values()):
        grid_values = dict(zip(grid, combination))
        configuration = _configuration_name(grid_values)
        for seed in seeds:
            config = copy.deepcopy(base)
            for path, value in grid_values.items():
                _set_by_path(config, path, copy.deepcopy(value))
            config[SEED_KEY] = seed
            runs.append(RunSpecification(configuration, seed, grid_values, config))

    labels = [run.label for run in runs]
    if len(set(labels)) != len(labels):
        raise ConfigurationError("La grilla produce corridas con nombres repetidos")
    return runs


def _seeds(experiment: Mapping[str, Any]) -> list[int]:
    if SEEDS_KEY in experiment and SEED_KEY in experiment:
        raise ConfigurationError("Usar 'seed' o 'seeds', no ambos")
    seeds = experiment.get(SEEDS_KEY, [experiment.get(SEED_KEY, DEFAULT_SEED)])
    if not isinstance(seeds, list) or not seeds or any(not isinstance(s, int) for s in seeds):
        raise ConfigurationError("'seeds' debe ser una lista no vacía de enteros")
    if len(set(seeds)) != len(seeds):
        raise ConfigurationError("'seeds' contiene semillas repetidas")
    return seeds


def _validate_grid(grid: object) -> None:
    if not isinstance(grid, Mapping):
        raise ConfigurationError("'grid' debe ser un objeto {ruta: [valores]}")
    for path, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ConfigurationError(f"grid['{path}'] debe ser una lista no vacía de valores")
        if path.split(".")[0] in (GRID_KEY, SEEDS_KEY, SEED_KEY):
            raise ConfigurationError(f"grid no puede modificar '{path}'; usar 'seeds'")


def _set_by_path(config: dict[str, Any], path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    section = config
    for key in parents:
        child = section.setdefault(key, {})
        if not isinstance(child, dict):
            raise ConfigurationError(f"La ruta de grilla '{path}' atraviesa un valor que no es objeto")
        section = child
    section[leaf] = value


def _configuration_name(grid_values: Mapping[str, Any]) -> str:
    if not grid_values:
        return BASELINE_CONFIGURATION
    parts = [f"{path.split('.')[-1]}={_format_value(value)}" for path, value in grid_values.items()]
    return safe_filename("__".join(parts))


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return "-".join(_format_value(item) for item in value) or "none"
    if isinstance(value, Mapping):
        # El "type" de una sección se entiende solo: {"type": "adam", ...} -> "adam,...".
        return ",".join(
            _format_value(item) if key == "type" else f"{key}={_format_value(item)}"
            for key, item in value.items()
        )
    if isinstance(value, float):
        return f"{value:g}"
    return json.dumps(value) if value is None or isinstance(value, bool) else str(value)
