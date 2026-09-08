"""Punto de entrada para graficar el `results.csv` de una matriz de experimentos."""

from __future__ import annotations

import argparse
from pathlib import Path

from simulation.analysis import DEFAULT_CONFIDENCE, read_results, summarize_by
from simulation.plotting import plot_group_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grafica una columna de results.csv agrupada, con barras de error."
    )
    parser.add_argument("results", help="Ruta al results.csv de la matriz.")
    parser.add_argument(
        "--group", default="fitness.metric",
        help="Columna que define los grupos (default: fitness.metric).",
    )
    parser.add_argument(
        "--value", default="elapsed_seconds",
        help="Columna numérica a promediar (default: elapsed_seconds).",
    )
    parser.add_argument(
        "--output", help="PNG de salida (default: junto al results.csv).",
    )
    parser.add_argument("--title", help="Título del gráfico.")
    parser.add_argument("--axis-label", help="Etiqueta del eje de magnitud.")
    arguments = parser.parse_args()

    rows = read_results(arguments.results)
    summaries = summarize_by(rows, arguments.group, arguments.value)
    runs = min(summary.runs for summary in summaries)
    output = Path(arguments.output or Path(arguments.results).with_name(
        f"{arguments.value}_por_{arguments.group.replace('.', '_')}.png"
    ))
    path = plot_group_comparison(
        summaries,
        output,
        title=arguments.title or f"{arguments.value} por {arguments.group}",
        axis_label=(
            arguments.axis_label
            or f"{arguments.value} (media, IC {DEFAULT_CONFIDENCE:.0%}, n>={runs})"
        ),
    )
    for summary in summaries:
        print(
            f"{summary.label:12s} media={summary.mean:8.3f} "
            f"desvío={summary.standard_deviation:6.3f} "
            f"IC95=±{summary.confidence_margin():6.3f} n={summary.runs}"
        )
    print(f"Gráfico guardado en: {path}")


if __name__ == "__main__":
    main()
