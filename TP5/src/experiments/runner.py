"""Ejecución de experimentos: entrenar, evaluar y guardar cada corrida.

Estructura de salida:
    results/<experimento>/summary.csv              una fila por corrida
    results/<experimento>/<corrida>/config.json    configuración exacta de la corrida
                                    model.npz, history.csv, learning.png,
                                    training.json (métricas del entrenamiento),
                                    summary.json y las salidas de cada evaluación
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from src import plotting
from src.data.dataset import Dataset
from src.data.factory import build_dataset
from src.evaluation.context import EvaluationContext, write_csv
from src.evaluation.pitch_interpolation import validate_pitch_interpolation
from src.evaluation.metrics import PIXEL_ERROR_TARGET, reconstruction_quality
from src.evaluation.registry import DEFAULT_EVALUATIONS, run_evaluations, validate_evaluation_names
from src.experiments.grid import RunSpecification, expand_runs
from src.models.autoencoder import Autoencoder
from src.models.variational import VariationalAutoencoder
from src.training.interpolation import build_interpolation_triplets, triplet_records
from src.models.factory import build_model, load_parameters, save_parameters
from src.options import ConfigurationError, Options
from src.training.trainer import Trainer, TrainingPlan, build_training_plan

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = "config.json"
MODEL_FILE = "model.npz"
SUMMARY_FILE = "summary.json"
TRAINING_SUMMARY_FILE = "training.json"
HISTORY_FILE = "history.csv"
EXPERIMENT_SUMMARY_FILE = "summary.csv"
_INDEPENDENT_STREAMS = 6


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
    seed: int
    interpolation_triplets: np.ndarray | None = None
    interpolation_rng: np.random.Generator | None = None


def run_experiment(config_path: Path, force: bool = False, evaluate_only: bool = False,
                   jobs: int = 1) -> Path:
    """Corre todas las combinaciones de un archivo de configuración.

    Las corridas ya terminadas se saltean salvo `force`. Con `evaluate_only`
    no se entrena: se recargan los pesos guardados y se repiten las evaluaciones.
    Devuelve la carpeta del experimento.
    """
    return run_experiments([config_path], force, evaluate_only, jobs)[0]


def run_experiments(config_paths: list[Path], force: bool = False, evaluate_only: bool = False,
                    jobs: int = 1) -> list[Path]:
    """Como run_experiment, pero con todas las corridas de varios archivos en un mismo pool.

    Con `jobs` > 1 cada corrida va a un proceso propio. Las corridas son
    independientes (cada una tiene su semilla y su carpeta), así que el resultado
    es el mismo que en serie; sólo cambia el orden en que terminan.
    """
    experiments = []
    for config_path in config_paths:
        experiment = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(experiment, dict):
            raise ConfigurationError(f"{config_path} debe contener un objeto JSON")
        name = experiment.pop("name", config_path.stem)
        output_directory = PROJECT_ROOT / experiment.pop("output", f"results/{name}")
        experiments.append((name, output_directory, expand_runs(experiment)))
    tasks = [(name, run, output_directory / run.label)
             for name, output_directory, runs in experiments for run in runs]
    if type(jobs) is not int or jobs < 1:
        raise ConfigurationError("jobs debe ser un entero positivo")

    summaries: dict[Path, dict[str, Any]] = {}
    if jobs == 1:
        for position, (name, run, run_directory) in enumerate(tasks, start=1):
            print(f"[{position}/{len(tasks)}] {name} / {run.label}", flush=True)
            summaries[run_directory] = _obtain_summary(run, run_directory, force, evaluate_only)
    else:
        # spawn: procesos limpios que leen OPENBLAS_NUM_THREADS al importar numpy.
        with ProcessPoolExecutor(min(jobs, len(tasks)), mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_obtain_summary, run, run_directory, force, evaluate_only):
                       (name, run, run_directory) for name, run, run_directory in tasks}
            print(f"{len(tasks)} corridas en {min(jobs, len(tasks))} procesos", flush=True)
            for done, future in enumerate(as_completed(futures), start=1):
                name, run, run_directory = futures[future]
                summaries[run_directory] = future.result()
                print(f"[{done}/{len(tasks)}] terminada {name} / {run.label}", flush=True)

    for name, output_directory, runs in experiments:
        rows = [
            {"run": run.label, "configuration": run.configuration, "seed": run.seed}
            | {path: json.dumps(value) for path, value in run.grid_values.items()}
            | summaries[output_directory / run.label]
            for run in runs
        ]
        write_csv(output_directory / EXPERIMENT_SUMMARY_FILE, rows)
        print(f"Resultados en {output_directory}")
    return [output_directory for _, output_directory, _ in experiments]


def load_run(run_directory: Path) -> tuple[Autoencoder, Dataset]:
    """Modelo entrenado y dataset de una corrida guardada, para uso interactivo."""
    config = json.loads((run_directory / CONFIG_FILE).read_text(encoding="utf-8"))
    prepared = _prepare(config)
    load_parameters(prepared.model, run_directory / MODEL_FILE)
    return prepared.model, prepared.dataset


def _obtain_summary(
    run: RunSpecification, run_directory: Path, force: bool, evaluate_only: bool
) -> dict[str, Any]:
    summary_path = run_directory / SUMMARY_FILE
    if evaluate_only:
        return _reevaluate(run, run_directory)
    if summary_path.is_file() and not force:
        print("  ya estaba terminada; se reutiliza (usar --force para repetirla)")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return _train_and_evaluate(run, run_directory)


def _train_and_evaluate(run: RunSpecification, run_directory: Path) -> dict[str, Any]:
    prepared = _prepare(run.config)
    trainer = Trainer(
        prepared.model,
        prepared.plan,
        lambda model: reconstruction_quality(model, prepared.dataset),
        prepared.shuffle_rng,
        prepared.corruption_rng,
        prepared.interpolation_triplets,
        prepared.interpolation_rng,
    )
    started = time.perf_counter()
    history = trainer.train(prepared.dataset.samples)
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
    if prepared.interpolation_triplets is not None:
        training_summary["interpolation_triplets"] = len(prepared.interpolation_triplets)
        training_summary["interpolation_pitch_supervision"] = prepared.plan.interpolation.weight > 0
        write_csv(run_directory / "interpolation_triplets.csv",
                  triplet_records(prepared.dataset, prepared.interpolation_triplets))
    _write_json(run_directory / TRAINING_SUMMARY_FILE, training_summary)
    return _evaluate_and_summarize(prepared, run_directory, training_summary)


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
        prepared.model, prepared.dataset, run_directory, prepared.evaluation_rng, prepared.seed
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
    initialization, shuffle, corruption, sampling, evaluation, interpolation_rng = (
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
    if "pitch_interpolation" in evaluations:
        validate_pitch_interpolation(dataset, Options(evaluations["pitch_interpolation"],
                                                     "evaluation.pitch_interpolation"))
    triplets = None
    if plan.interpolation is not None:
        if not isinstance(model, VariationalAutoencoder):
            raise ConfigurationError("training.interpolation requiere model.type variational")
        triplets = build_interpolation_triplets(dataset)
    return PreparedRun(dataset, model, plan, evaluations, shuffle, corruption, evaluation, seed,
                       triplets, interpolation_rng)


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
