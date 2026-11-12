"""Ejecución de experimentos: entrenar, evaluar y guardar cada corrida.

Estructura de salida:
    results/<experimento>/summary.csv              una fila por corrida
    results/<experimento>/<corrida>/config.json    configuración exacta de la corrida
                                    model.npz, history.csv, learning.png,
                                    training.json (métricas del entrenamiento),
                                    summary.json y las salidas de cada evaluación
"""

from __future__ import annotations

import functools
import json
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src import plotting
from src.data.dataset import Dataset
from src.data.factory import build_dataset
from src.evaluation.context import EvaluationContext, write_csv
from src.evaluation.metrics import PIXEL_ERROR_TARGET, reconstruction_quality
from src.evaluation.registry import DEFAULT_EVALUATIONS, run_evaluations, validate_evaluation_names
from src.experiments.grid import RunSpecification, expand_runs
from src.models.autoencoder import Autoencoder
from src.models.factory import build_model, load_parameters, save_parameters
from src.options import ConfigurationError, Options
from src.training.trainer import Trainer, TrainingDiverged, TrainingPlan, build_training_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = "config.json"
MODEL_FILE = "model.npz"
SUMMARY_FILE = "summary.json"
TRAINING_SUMMARY_FILE = "training.json"
HISTORY_FILE = "history.csv"
EXPERIMENT_SUMMARY_FILE = "summary.csv"
DIVERGED_KEY = "diverged_at_epoch"
_INDEPENDENT_STREAMS = 5


@dataclass(frozen=True)
class PreparedRun:
    """Todo lo que una configuración describe, ya construido y validado."""

    dataset: Dataset
    model: Autoencoder
    plan: TrainingPlan
    evaluations: dict[str, Any]
    shuffle_rng: np.random.Generator
    corruption_rng: np.random.Generator
    evaluation_rng: np.random.Generator


def run_experiment(
    config_path: Path, force: bool = False, evaluate_only: bool = False, jobs: int = 1
) -> Path:
    """Corre todas las combinaciones de un archivo de configuración.

    Las corridas ya terminadas se saltean salvo `force`. Con `evaluate_only`
    no se entrena: se recargan los pesos guardados y se repiten las evaluaciones.
    Con `jobs` > 1 las corridas se reparten entre procesos; cada una depende
    solo de su semilla, así que el resultado es el mismo que en serie.
    Devuelve la carpeta del experimento.
    """
    experiment = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(experiment, dict):
        raise ConfigurationError(f"{config_path} debe contener un objeto JSON")
    name = experiment.pop("name", config_path.stem)
    output_directory = PROJECT_ROOT / experiment.pop("output", f"results/{name}")
    runs = expand_runs(experiment)

    work = functools.partial(
        _run_with_progress, name, len(runs), output_directory, force, evaluate_only
    )
    numbered = list(enumerate(runs, start=1))
    if jobs == 1:
        summaries = list(map(work, numbered))
    else:
        with multiprocessing.Pool(jobs) as pool:
            summaries = pool.map(work, numbered, chunksize=1)
    rows = [
        {"run": run.label, "configuration": run.configuration, "seed": run.seed}
        | {path: json.dumps(value) for path, value in run.grid_values.items()}
        | summary
        for run, summary in zip(runs, summaries)
    ]
    write_csv(output_directory / EXPERIMENT_SUMMARY_FILE, rows)
    print(f"Resultados en {output_directory}")
    return output_directory


def load_run(run_directory: Path) -> tuple[Autoencoder, Dataset]:
    """Modelo entrenado y dataset de una corrida guardada, para uso interactivo."""
    config = json.loads((run_directory / CONFIG_FILE).read_text(encoding="utf-8"))
    prepared = _prepare(config)
    load_parameters(prepared.model, run_directory / MODEL_FILE)
    return prepared.model, prepared.dataset


def _run_with_progress(
    name: str,
    total: int,
    output_directory: Path,
    force: bool,
    evaluate_only: bool,
    numbered_run: tuple[int, RunSpecification],
) -> dict[str, Any]:
    position, run = numbered_run
    print(f"[{position}/{total}] {name} / {run.label}", flush=True)
    return _obtain_summary(run, output_directory / run.label, force, evaluate_only)


def _obtain_summary(
    run: RunSpecification, run_directory: Path, force: bool, evaluate_only: bool
) -> dict[str, Any]:
    summary_path = run_directory / SUMMARY_FILE
    stored = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    if evaluate_only:
        # Una corrida que divergió no tiene modelo que evaluar.
        if stored is not None and DIVERGED_KEY in stored:
            return stored
        return _reevaluate(run, run_directory)
    if stored is not None and not force:
        print("  ya estaba terminada; se reutiliza (usar --force para repetirla)")
        return stored
    return _train_and_evaluate(run, run_directory)


def _train_and_evaluate(run: RunSpecification, run_directory: Path) -> dict[str, Any]:
    prepared = _prepare(run.config)
    trainer = Trainer(
        prepared.model,
        prepared.plan,
        lambda model: reconstruction_quality(model, prepared.dataset),
        prepared.shuffle_rng,
        prepared.corruption_rng,
    )
    started = time.perf_counter()
    try:
        history = trainer.train(prepared.dataset.samples)
    except TrainingDiverged as divergence:
        return _record_divergence(run, run_directory, divergence)
    training_seconds = time.perf_counter() - started

    run_directory.mkdir(parents=True, exist_ok=True)
    _write_json(run_directory / CONFIG_FILE, run.config)
    save_parameters(prepared.model, run_directory / MODEL_FILE)
    write_csv(run_directory / HISTORY_FILE, history)
    plotting.save_learning_curves(history, run_directory / "learning.png")

    training_summary = {
        "samples": len(prepared.dataset),
        "training_seconds": round(training_seconds, 2),
        **_final_loss_terms(history, prepared),
    }
    if prepared.dataset.is_binary:
        training_summary["first_epoch_within_target"] = _first_epoch_within_target(history)
    _write_json(run_directory / TRAINING_SUMMARY_FILE, training_summary)
    return _evaluate_and_summarize(prepared, run_directory, training_summary)


def _record_divergence(
    run: RunSpecification, run_directory: Path, divergence: TrainingDiverged
) -> dict[str, Any]:
    """Una divergencia es un resultado (la configuración falla), no un error del experimento.

    No hay modelo que guardar; queda el historial hasta la divergencia.
    """
    print(f"  {divergence}", flush=True)
    run_directory.mkdir(parents=True, exist_ok=True)
    _write_json(run_directory / CONFIG_FILE, run.config)
    if divergence.history:
        write_csv(run_directory / HISTORY_FILE, divergence.history)
    summary = {DIVERGED_KEY: divergence.epoch}
    _write_json(run_directory / SUMMARY_FILE, summary)
    return summary


def _reevaluate(run: RunSpecification, run_directory: Path) -> dict[str, Any]:
    training_summary_path = run_directory / TRAINING_SUMMARY_FILE
    if not training_summary_path.is_file():
        raise FileNotFoundError(f"No hay una corrida entrenada en {run_directory}")
    prepared = _prepare(run.config)
    load_parameters(prepared.model, run_directory / MODEL_FILE)
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    _write_json(run_directory / CONFIG_FILE, run.config)
    return _evaluate_and_summarize(prepared, run_directory, training_summary)


def _evaluate_and_summarize(
    prepared: PreparedRun, run_directory: Path, training_summary: dict[str, Any]
) -> dict[str, Any]:
    context = EvaluationContext(
        prepared.model, prepared.dataset, run_directory, prepared.evaluation_rng
    )
    summary = {
        **training_summary,
        **reconstruction_quality(prepared.model, prepared.dataset),
        **run_evaluations(context, prepared.evaluations),
    }
    _write_json(run_directory / SUMMARY_FILE, summary)
    return summary


def _prepare(config: dict[str, Any]) -> PreparedRun:
    options = Options(config, "config")
    seed = options.take("seed", 0)
    initialization, shuffle, corruption, sampling, evaluation = (
        np.random.default_rng(sequence)
        for sequence in np.random.SeedSequence(seed).spawn(_INDEPENDENT_STREAMS)
    )
    dataset = build_dataset(options.section("dataset"), PROJECT_ROOT)
    model = build_model(options.section("model"), dataset.samples.shape[1], initialization, sampling)
    plan = build_training_plan(options.section("training"))
    evaluations = options.take("evaluation", DEFAULT_EVALUATIONS)
    options.ensure_consumed()
    if not isinstance(evaluations, dict):
        raise ConfigurationError("'evaluation' debe ser un objeto {nombre: opciones}")
    validate_evaluation_names(evaluations)
    return PreparedRun(dataset, model, plan, evaluations, shuffle, corruption, evaluation)


def _final_loss_terms(history: list[dict[str, float]], prepared: PreparedRun) -> dict[str, float]:
    """Términos de la pérdida en la última época; la calidad se recalcula al evaluar."""
    quality_names = reconstruction_quality(prepared.model, prepared.dataset).keys()
    return {
        f"final_{name}": value
        for name, value in history[-1].items()
        if name != "epoch" and name not in quality_names
    }


def _first_epoch_within_target(history: list[dict[str, float]]) -> int | None:
    """Primera época registrada con todos los patrones en la meta; None si nunca se alcanzó.

    La resolución es la de `training.log_every`.
    """
    for row in history:
        if row["max_pixel_errors"] <= PIXEL_ERROR_TARGET:
            return int(row["epoch"])
    return None


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
