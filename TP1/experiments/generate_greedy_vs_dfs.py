"""Generate Greedy-versus-DFS comparisons from the Ivo benchmark."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path("results/benchmark_results_ivo.csv")
OUTPUT = Path("figures/greedy_vs_dfs")
LEVELS = ["level-1", "level-2", "level-3", "level-4", "level-6", "level-7"]
GREEDY_HEURISTIC = "push_distance"


def load_data():
    df = pd.read_csv(INPUT)
    df = df[df["pruning_mode"].eq("dead_squares")].copy()
    df["success_bool"] = df["success"].astype(str).str.lower().eq("true")
    for column in ("expanded_nodes", "frontier_nodes", "time_sec"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def labels(values):
    return [str(value).replace("level-", "Nivel ") for value in values]


def plot_success(df):
    dfs = df[df["algorithm"].eq("dfs")].groupby("level")["success_bool"].mean()
    greedy = df[
        df["algorithm"].eq("greedy") & df["heuristic"].eq(GREEDY_HEURISTIC)
    ].set_index("level")["success_bool"]
    table = pd.DataFrame(
        {"DFS": dfs, "Greedy + Push distance": greedy}
    ).reindex(LEVELS).mul(100)

    fig, ax = plt.subplots(figsize=(11, 6))
    table.plot(kind="bar", ax=ax, color=["#FF7F0E", "#9467BD"], width=0.75)
    ax.set_title("Tasa de éxito — DFS vs. Greedy + Push distance")
    ax.set_xlabel("")
    ax.set_ylabel("Configuraciones resueltas (%)")
    ax.set_xticklabels(labels(table.index), rotation=0)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Algoritmo")
    save(fig, "01_tasa_exito")


def plot_metric(df, metric, title, ylabel, filename):
    dfs = df[(df["algorithm"].eq("dfs")) & df["success_bool"]]
    greedy = df[
        (df["algorithm"].eq("greedy"))
        & df["heuristic"].eq(GREEDY_HEURISTIC)
        & df["success_bool"]
    ]
    table = pd.DataFrame(
        {
            "DFS": dfs.set_index("level")[metric],
            "Greedy + Push distance": greedy.set_index("level")[metric],
        }
    ).reindex(LEVELS)

    fig, ax = plt.subplots(figsize=(11, 6))
    table.plot(kind="bar", ax=ax, color=["#FF7F0E", "#9467BD"], width=0.75)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel.replace(" (mediana", " ("))
    ax.set_xticklabels(labels(table.index), rotation=0)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Algoritmo")
    save(fig, filename)


def main():
    df = load_data()
    plot_success(df)
    plot_metric(
        df,
        "expanded_nodes",
        "Nodos expandidos — DFS vs. Greedy + Push distance",
        "Nodos expandidos (escala logarítmica)",
        "02_nodos_expandidos",
    )
    plot_metric(
        df,
        "time_sec",
        "Tiempo de búsqueda — DFS vs. Greedy + Push distance",
        "Tiempo (s, escala logarítmica)",
        "03_tiempo",
    )
    plot_metric(
        df,
        "frontier_nodes",
        "Nodos frontera — DFS vs. Greedy + Push distance",
        "Nodos frontera (escala logarítmica)",
        "04_nodos_frontera",
    )
    print(f"Generados 4 gráficos en {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
