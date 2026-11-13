"""Evaluaciones disponibles para la sección `evaluation` de una configuración."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.evaluation.context import EvaluationContext, Metrics
from src.evaluation.denoising import evaluate_denoising
from src.evaluation.pitch_interpolation import evaluate_pitch_interpolation
from src.evaluation.generation import evaluate_interpolation, evaluate_sampling
from src.evaluation.reconstruction import (
    evaluate_latent,
    evaluate_latent_grid,
    evaluate_reconstruction,
)
from src.options import Options, choose

Evaluation = Callable[[EvaluationContext, Options], Metrics]

EVALUATIONS: dict[str, Evaluation] = {
    "reconstruction": evaluate_reconstruction,
    "latent": evaluate_latent,
    "latent_grid": evaluate_latent_grid,
    "interpolation": evaluate_interpolation,
    "sampling": evaluate_sampling,
    "denoising": evaluate_denoising,
    "pitch_interpolation": evaluate_pitch_interpolation,
}
DEFAULT_EVALUATIONS: dict[str, dict[str, Any]] = {"reconstruction": {}, "latent": {}}


def validate_evaluation_names(requested: Mapping[str, Any]) -> None:
    """Falla antes de entrenar si se pidió una evaluación inexistente."""
    for name in requested:
        choose(EVALUATIONS, name, "Evaluación")


def run_evaluations(context: EvaluationContext, requested: Mapping[str, Any]) -> Metrics:
    """Ejecuta cada evaluación presente en `requested` y reúne sus métricas."""
    validate_evaluation_names(requested)
    metrics: Metrics = {}
    for name, settings in requested.items():
        metrics.update(EVALUATIONS[name](context, Options(settings, f"evaluation.{name}")))
    return metrics
