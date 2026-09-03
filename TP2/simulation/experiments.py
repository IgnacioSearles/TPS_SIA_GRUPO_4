"""Runner liviano de matrices de experimentos sobre la API de simulaciones."""

from __future__ import annotations

import csv
import itertools
import json
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
) -> Path:
    """Ejecuta combinaciones, guarda cada corrida y agrega un CSV reproducible."""
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary_path = root / "results.csv"
    combinations = expand_matrix(matrix)
    initial_completed = (
        sum(
            (root / f"run_{index:04d}" / "run" / "summary.json").exists()
            for index in range(len(combinations))
        )
        if resume
        else 0
    )
    _write_experiment_progress(root, initial_completed, len(combinations), None)
    for index, overrides in enumerate(combinations):
        run_dir = root / f"run_{index:04d}"
        result_file = run_dir / "run" / "summary.json"
        if resume and result_file.exists():
            row = json.loads(result_file.read_text(encoding="utf-8"))
            row.update({"run": index, **overrides})
            rows.append(row)
            _write_results(rows, summary_path, root / "fitness.png")
            _write_experiment_progress(root, index + 1, len(combinations), index)
            continue
        output = run_dir / "best.png"
        top_level = {"output": str(output)}
        materialized_overrides = _materialize_overrides(overrides)
        for key, value in materialized_overrides.items():
            _set_dotted(top_level, key, value)
        config = load_simulation_config(config_path, top_level)
        outcome = run_simulation(config)
        row = {"run": index, **overrides, "seed": outcome.seed,
               "generations": outcome.generations, "best_fitness": outcome.best_fitness,
               "elapsed_seconds": outcome.elapsed_seconds,
               "termination_reason": outcome.termination_reason}
        rows.append(row)
        _write_results(rows, summary_path, root / "fitness.png")
        _write_experiment_progress(root, index + 1, len(combinations), index)
    return summary_path


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
