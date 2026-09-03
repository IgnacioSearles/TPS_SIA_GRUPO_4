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
    TerminationConfig,
    load_simulation_config,
)
from .reporting import PreviewWriter, ProgressReporter, RunArtifactWriter
from .runner import SimulationOutcome, run_simulation
from .experiments import expand_matrix, load_experiment_spec, run_experiment_matrix
from .section import ConfigSection, ConfigurationError

__all__ = [
    "ConfigSection", "ConfigurationError", "CrossoverConfig", "FitnessConfig",
    "MutationConfig", "PopulationConfig", "PreviewConfig", "PreviewWriter",
    "ProgressReporter", "RunArtifactWriter", "ReheatConfig", "SelectionConfig", "SimulationConfig",
    "SimulationOutcome", "TerminationConfig", "expand_matrix", "load_experiment_spec",
    "load_simulation_config", "run_experiment_matrix", "run_simulation",
]
