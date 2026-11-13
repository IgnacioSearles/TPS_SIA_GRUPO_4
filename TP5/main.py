"""Corre los experimentos descritos en uno o más archivos de configuración.

Uso, desde TP5:
    python main.py configs/ej1a_baseline.json
    python main.py configs/ej1a_*.json --force
    python main.py configs/ej2_vae_nsynth.json --evaluate-only
    python main.py configs/ej2_pitch_microtonal_*.json --jobs 6   # 6 corridas a la vez
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

from src.experiments.runner import run_experiments
from src.options import ConfigurationError

BLAS_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TP5: autoencoders, denoising y VAE")
    parser.add_argument("configs", nargs="+", help="archivos JSON de experimento (admite comodines)")
    parser.add_argument(
        "--force", action="store_true", help="repetir corridas que ya tienen resultados"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="no entrenar: recargar los modelos guardados y repetir las evaluaciones",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="corridas simultáneas, cada una en su proceso (todas las configs comparten el pool)",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="hilos BLAS por proceso con --jobs (por defecto: núcleos / jobs)",
    )
    arguments = parser.parse_args()
    if arguments.force and arguments.evaluate_only:
        parser.error("--force y --evaluate-only son excluyentes")
    if arguments.jobs < 1 or (arguments.threads is not None and arguments.threads < 1):
        parser.error("--jobs y --threads deben ser positivos")
    return arguments


def expand_config_paths(patterns: list[str]) -> list[Path]:
    """PowerShell no expande comodines, así que se resuelven acá."""
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            raise SystemExit(f"Ningún archivo coincide con {pattern}")
        paths.extend(Path(match) for match in matches)
    return paths


def main() -> None:
    arguments = parse_arguments()
    if arguments.jobs > 1:
        # Los procesos hijos heredan estas variables antes de importar numpy:
        # sin esto cada uno abriría un hilo por núcleo y se pisarían entre sí.
        threads = arguments.threads or max(1, (os.cpu_count() or 1) // arguments.jobs)
        for variable in BLAS_VARIABLES:
            os.environ[variable] = str(threads)
    paths = expand_config_paths(arguments.configs)
    try:
        run_experiments(paths, arguments.force, arguments.evaluate_only, arguments.jobs)
    except (ConfigurationError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
