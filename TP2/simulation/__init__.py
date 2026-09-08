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
from .analysis import GroupSummary, read_results, summarize_by
from .plotting import plot_group_comparison
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
    "GroupSummary", "expand_matrix", "load_experiment_spec",
    "load_simulation_config", "plot_group_comparison", "read_results", "run_experiment_matrix", "run_simulation",
    "summarize_by",
]
