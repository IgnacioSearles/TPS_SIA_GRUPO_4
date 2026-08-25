"""Generate only Greedy metrics for levels 6 and 7 of the Ivo benchmark."""

from pathlib import Path

import generate_ivo_plots as plots


def main():
    plots.LEVEL_ORDER[:] = ["level-6", "level-7"]
    plots.OUTPUT = Path("figures/benchmark_ivo")
    data = plots.load_data()
    plots.plot_greedy_expanded(data)
    plots.plot_greedy_metric(
        data,
        "time_sec",
        "Tiempo de Greedy por heurística — niveles 6 y 7",
        "Tiempo (s, mediana, escala logarítmica)",
        "08_greedy_tiempo",
    )
    plots.plot_greedy_metric(
        data,
        "frontier_nodes",
        "Nodos frontera de Greedy por heurística — niveles 6 y 7",
        "Nodos frontera (mediana, escala logarítmica)",
        "09_greedy_nodos_frontera",
    )
    print(f"Generados 3 gráficos en {plots.OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
