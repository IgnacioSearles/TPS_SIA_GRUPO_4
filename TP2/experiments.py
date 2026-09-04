"""Punto de entrada para ejecutar una matriz declarada en JSON."""

from __future__ import annotations

import argparse

from simulation import load_experiment_spec, run_experiment_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecuta una matriz de experimentos del TP2.")
    parser.add_argument("spec", help="JSON con config, matrix y output_directory")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Cantidad de corridas simultáneas (default: 1).",
    )
    args = parser.parse_args()
    config, matrix, output_directory = load_experiment_spec(args.spec)
    print(run_experiment_matrix(config, matrix, output_directory, workers=args.workers))


if __name__ == "__main__":
    main()
