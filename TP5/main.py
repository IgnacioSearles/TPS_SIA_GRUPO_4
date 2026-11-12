"""Corre los experimentos descritos en uno o más archivos de configuración.

Uso, desde TP5:
    python main.py configs/ej1a_baseline.json
    python main.py configs/ej1a_*.json --force
    python main.py configs/ej2_vae_nsynth.json --evaluate-only
    python main.py configs/ej1a_search_*.json --jobs 10
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from src.experiments.runner import run_experiment
from src.options import ConfigurationError


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
        "--jobs", type=int, default=1, help="procesos en paralelo (cada corrida usa uno)"
    )
    arguments = parser.parse_args()
    if arguments.jobs < 1:
        parser.error("--jobs debe ser al menos 1")
    if arguments.force and arguments.evaluate_only:
        parser.error("--force y --evaluate-only son excluyentes")
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
    for config_path in expand_config_paths(arguments.configs):
        try:
            run_experiment(
                config_path, arguments.force, arguments.evaluate_only, arguments.jobs
            )
        except (ConfigurationError, FileNotFoundError) as error:
            raise SystemExit(f"Error en {config_path}: {error}") from error


if __name__ == "__main__":
    main()
