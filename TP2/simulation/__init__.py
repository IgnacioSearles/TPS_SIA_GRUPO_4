"""Capa de composición: configuración declarativa y ejecución de simulaciones."""

from .config import (
    CrossoverConfig,
    FitnessConfig,
    MutationConfig,
    PopulationConfig,
    PreviewConfig,
    ReheatConfig,
    SelectionConfig,
    SimulationConfig,
    load_simulation_config,
)
from .reporting import PreviewWriter, ProgressReporter
from .runner import SimulationOutcome, run_simulation
from .section import ConfigSection, ConfigurationError

__all__ = [
    "ConfigSection", "ConfigurationError", "CrossoverConfig", "FitnessConfig",
    "MutationConfig", "PopulationConfig", "PreviewConfig", "PreviewWriter",
    "ProgressReporter", "ReheatConfig", "SelectionConfig", "SimulationConfig",
    "SimulationOutcome", "load_simulation_config", "run_simulation",
]
