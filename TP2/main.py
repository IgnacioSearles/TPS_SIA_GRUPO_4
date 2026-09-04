"""Punto de entrada: corre una simulación descrita por un archivo de configuración."""

from __future__ import annotations

import argparse
from typing import Any

from simulation import ConfigurationError, load_simulation_config, run_simulation


def _parse_arguments() -> argparse.Namespace:
    """Define la interfaz de línea de comandos: un config y unos pocos overrides.

    Los parámetros del experimento viven en el archivo de configuración; acá solo
    se exponen los que suelen cambiar entre corridas de un mismo experimento.
    """
    parser = argparse.ArgumentParser(
        description="Aproximación de imágenes con triángulos mediante algoritmos genéticos.",
    )
    parser.add_argument("config", help="Archivo JSON de configuración de la simulación.")
    parser.add_argument("--image", help="Reemplaza la imagen objetivo del config.")
    parser.add_argument("--output", help="Reemplaza la ruta de salida del config.")
    parser.add_argument(
        "--seed", type=int,
        help="Reemplaza la semilla del config; sin semilla se sortea una y se informa.",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Desactiva los previews aunque el config los declare.",
    )
    parser.add_argument(
        "--gif", metavar="RUTA",
        help="Escribe un GIF de la evolución en esa ruta, aunque el config no lo declare.",
    )
    parser.add_argument(
        "--no-gif", action="store_true",
        help="Desactiva el GIF aunque el config lo declare.",
    )
    return parser.parse_args()


def _overrides_from(arguments: argparse.Namespace) -> dict[str, Any]:
    """Traduce los flags presentes a claves de configuración de nivel superior."""
    overrides: dict[str, Any] = {}
    if arguments.image is not None:
        overrides["image"] = arguments.image
    if arguments.output is not None:
        overrides["output"] = arguments.output
    if arguments.seed is not None:
        overrides["seed"] = arguments.seed
    if arguments.no_preview:
        overrides["preview"] = None
    if arguments.gif is not None:
        overrides["gif"] = {"path": arguments.gif}
    if arguments.no_gif:
        overrides["gif"] = None
    return overrides


def main() -> None:
    """Carga la configuración, la valida y ejecuta la simulación."""
    arguments = _parse_arguments()
    try:
        config = load_simulation_config(arguments.config, _overrides_from(arguments))
    except ConfigurationError as error:
        raise SystemExit(f"Configuración inválida: {error}") from error
    run_simulation(config)


if __name__ == "__main__":
    main()
