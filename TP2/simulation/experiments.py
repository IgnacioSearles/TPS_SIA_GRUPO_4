"""Runner liviano de matrices de experimentos sobre la API de simulaciones."""

from __future__ import annotations

import csv
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from simulation.config import load_simulation_config
from simulation.runner import run_simulation


def expand_matrix(matrix: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    """Expande ``{"seed": [1, 2], "triangles": [10, 20]}`` en combinaciones."""
    if not matrix:
        return [{}]
    keys = tuple(matrix)
    values = []
    for key in keys:
        options = matrix[key]
        if not isinstance(options, list) or not options:
            raise ValueError(f"experiment matrix '{key}' must be a non-empty list")
        values.append(options)
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def run_experiment_matrix(
    config_path: str | Path,
    matrix: Mapping[str, list[Any]],
    output_directory: str | Path,
    *,
    resume: bool = True,
    workers: int = 1,
) -> Path:
    """Ejecuta combinaciones con concurrencia opcional y progreso reanudable."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary_path = root / "results.csv"
    combinations = expand_matrix(matrix)
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, overrides in enumerate(combinations):
        run_dir = root / f"run_{index:04d}"
        result_file = run_dir / "run" / "summary.json"
        if resume and result_file.exists():
            row = json.loads(result_file.read_text(encoding="utf-8"))
            row.update({"run": index, **overrides})
            rows.append(row)
        else:
            pending.append((index, overrides))

    _write_results(rows, summary_path, root / "fitness.png")
    _write_experiment_progress(root, len(rows), len(combinations), None)

    if workers == 1:
        completed = len(rows)
        for index, overrides in pending:
            rows.append(_run_one_experiment(config_path, root, index, overrides))
            completed += 1
            _write_results(rows, summary_path, root / "fitness.png")
            _write_experiment_progress(root, completed, len(combinations), index)
        return summary_path

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_one_experiment, config_path, root, index, overrides): index
            for index, overrides in pending
        }
        completed = len(rows)
        for future in as_completed(futures):
            index = futures[future]
            rows.append(future.result())
            completed += 1
            _write_results(rows, summary_path, root / "fitness.png")
            _write_experiment_progress(root, completed, len(combinations), index)
    return summary_path


def _run_one_experiment(
    config_path: str | Path,
    root: Path,
    index: int,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Ejecuta una corrida aislada; solo el proceso coordinador persiste la matriz."""
    run_dir = root / f"run_{index:04d}"
    top_level = {"output": str(run_dir / "best.png")}
    for key, value in _materialize_overrides(overrides).items():
        _set_dotted(top_level, key, value)
    config = load_simulation_config(config_path, top_level)
    outcome = run_simulation(config)
    return {
        "run": index, **overrides, "seed": outcome.seed,
        "generations": outcome.generations, "best_fitness": outcome.best_fitness,
        "elapsed_seconds": outcome.elapsed_seconds,
        "cpu_seconds": outcome.cpu_seconds,
        "termination_reason": outcome.termination_reason,
    }


def _materialize_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Resuelve parámetros que dependen de la imagen sin ampliar el producto cartesiano."""
    materialized = dict(overrides)
    image = materialized.get("image")
    if isinstance(image, Mapping):
        if "path" not in image or "generations" not in image:
            raise ValueError("an image matrix entry must contain path and generations")
        materialized["image"] = image["path"]
        materialized["population.generations"] = image["generations"]
    return materialized


def _write_results(
    rows: list[dict[str, Any]], summary_path: Path, plot_path: Path
) -> None:
    """Persiste resultados parciales para poder reanudar una matriz interrumpida."""
    rows = sorted(rows, key=lambda row: row.get("run", 0))
    fieldnames = sorted({key for row in rows for key in row})
    temporary_path = summary_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(summary_path)
    _write_fitness_plot(rows, plot_path)


def _write_experiment_progress(
    root: Path, completed: int, total: int, last_run: int | None
) -> None:
    """Escribe el estado observable del experimento después de cada corrida."""
    progress_path = root / "progress.json"
    temporary_path = progress_path.with_suffix(".json.tmp")
    payload = {
        "completed_runs": completed,
        "total_runs": total,
        "remaining_runs": total - completed,
        "last_completed_run": last_run,
        "status": "completed" if completed == total else "running",
    }
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(progress_path)


def _write_fitness_plot(rows: list[dict[str, Any]], path: Path) -> None:
    """Genera un gráfico agregado sin hacer obligatoria la importación al cargar módulos."""
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    values = [float(row["best_fitness"]) for row in rows if "best_fitness" in row]
    if not values:
        return
    figure, axis = plt.subplots()
    axis.plot(range(len(values)), values, marker="o")
    axis.set(xlabel="corrida", ylabel="mejor fitness", title="Resultados del experimento")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _set_dotted(mapping: dict[str, Any], key: str, value: Any) -> None:
    """Admite tanto claves top-level como rutas ``mutation.strategy``."""
    parts = key.split(".")
    current = mapping
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"experiment override conflicts at '{part}'")
        current = child
    current[parts[-1]] = value


def load_experiment_spec(path: str | Path) -> tuple[Path, Mapping[str, list[Any]], Path]:
    """Carga un spec JSON con ``config``, ``matrix`` y ``output_directory``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Path(raw["config"]), raw.get("matrix", {}), Path(raw.get("output_directory", "experiments"))
