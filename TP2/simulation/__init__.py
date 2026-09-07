"""Capa de composición: configuración declarativa y ejecución de simulaciones."""

from .config import (
    CrossoverConfig,
    FitnessConfig,
    FitnessAdaptiveMutationOptimizationConfig,
    GifConfig,
    IslandsOptimizationConfig,
    LocalSearchOptimizationConfig,
    MutationConfig,
    OptimizationConfig,
    PopulationConfig,
    PreviewConfig,
    ProgressiveResolutionOptimizationConfig,
    ProgressiveResolutionStageConfig,
    ReheatConfig,
    SelectionConfig,
    SimulationConfig,
    TerminationConfig,
    load_simulation_config,
)
from .reporting import GifWriter, PreviewWriter, ProgressReporter, RunArtifactWriter
from .runner import SimulationOutcome, run_simulation
from .experiments import expand_matrix, load_experiment_spec, run_experiment_matrix
from .section import ConfigSection, ConfigurationError

__all__ = [
    "ConfigSection", "ConfigurationError", "CrossoverConfig", "FitnessConfig",
    "FitnessAdaptiveMutationOptimizationConfig", "GifConfig", "GifWriter",
    "IslandsOptimizationConfig", "LocalSearchOptimizationConfig",
    "MutationConfig", "OptimizationConfig", "PopulationConfig", "PreviewConfig",
    "PreviewWriter", "ProgressReporter", "ProgressiveResolutionOptimizationConfig",
    "ProgressiveResolutionStageConfig", "RunArtifactWriter", "ReheatConfig",
    "SelectionConfig", "SimulationConfig", "SimulationOutcome", "TerminationConfig",
    "expand_matrix", "load_experiment_spec",
    "load_simulation_config", "run_experiment_matrix", "run_simulation",
]
