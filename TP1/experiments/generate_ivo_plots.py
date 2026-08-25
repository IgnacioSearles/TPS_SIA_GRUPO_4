"""Generate plots for the extended benchmark_results_ivo.csv benchmark."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


INPUT = Path("results/benchmark_results_ivo.csv")
OUTPUT = Path("figures/benchmark_ivo")
LEVEL_ORDER = ["level-1", "level-2", "level-3", "level-4", "level-6", "level-7"]
ALGORITHM_ORDER = ["bfs", "dfs", "iddfs", "astar", "greedy"]
HEURISTIC_ORDER = [
    "manhattan",
    "hungarian",
    "push_distance_nearest",
    "push_distance",
    "manhattan_player",
    "hungarian_player",
    "push_distance_player",
    "push_distance_nearest_player",
    "manhattan_player_all",
    "hungarian_player_all",
    "push_distance_player_all",
]
ALGORITHM_LABELS = {
    "bfs": "BFS",
    "dfs": "DFS",
    "iddfs": "IDDFS",
    "astar": "A*",
    "greedy": "Greedy",
}


def level_labels(index):
    return [str(value).replace("level-", "Nivel ") for value in index]


def heuristic_label(value):
    return (
        str(value)
        .replace("_player_all", " + jugador (todas)")
        .replace("_player", " + jugador (no resueltas)")
        .replace("push_distance_nearest", "Push distance nearest")
        .replace("push_distance", "Push distance")
        .replace("manhattan", "Manhattan")
        .replace("hungarian", "Hungarian")
    )


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def load_data():
    df = pd.read_csv(INPUT)
    df = df[df["pruning_mode"].eq("dead_squares")].copy()
    df["success_bool"] = df["success"].astype(str).str.lower().eq("true")
    for column in ("cost", "expanded_nodes", "frontier_nodes", "time_sec"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def plot_success_rate(df):
    rates = (
        df.groupby(["level", "algorithm"], observed=False)["success_bool"]
        .mean()
        .unstack("algorithm")
        .reindex(index=LEVEL_ORDER, columns=ALGORITHM_ORDER)
        .mul(100)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    rates.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Tasa de éxito por nivel — benchmark Ivo")
    ax.set_xlabel("")
    ax.set_ylabel("Configuraciones resueltas (%)")
    ax.set_xticklabels(level_labels(rates.index), rotation=0)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Algoritmo", ncol=3)
    save(fig, "01_tasa_exito_por_nivel")


def plot_astar_success(df):
    astar = df[df["algorithm"].eq("astar")]
    table = (
        astar.groupby(["heuristic", "level"], observed=False)["success_bool"]
        .mean()
        .unstack("level")
        .reindex(index=HEURISTIC_ORDER, columns=LEVEL_ORDER)
        .mul(100)
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_title("Tasa de éxito de A* por heurística y nivel")
    ax.set_xticks(range(len(LEVEL_ORDER)), level_labels(LEVEL_ORDER))
    ax.set_yticks(range(len(HEURISTIC_ORDER)), [heuristic_label(x) for x in HEURISTIC_ORDER])
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = table.iloc[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Éxito (%)")
    save(fig, "02_astar_exito_por_heuristica")


def plot_astar_expanded(df):
    astar = df[(df["algorithm"].eq("astar")) & df["success_bool"]]
    table = (
        astar.groupby(["heuristic", "level"], observed=False)["expanded_nodes"]
        .median()
        .unstack("level")
        .reindex(index=HEURISTIC_ORDER, columns=LEVEL_ORDER)
    )
    values = table.to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdYlGn_r",
        norm=LogNorm(vmin=max(float(positive.min()), 1), vmax=float(positive.max())),
    )
    ax.set_title("Nodos expandidos de A* en configuraciones resueltas")
    ax.set_xticks(range(len(LEVEL_ORDER)), level_labels(LEVEL_ORDER))
    ax.set_yticks(range(len(HEURISTIC_ORDER)), [heuristic_label(x) for x in HEURISTIC_ORDER])
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = table.iloc[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Nodos expandidos (mediana, escala logarítmica)")
    save(fig, "03_astar_nodos_expandidos")


def plot_algorithm_expanded(df):
    data = df[df["success_bool"]]
    table = (
        data.groupby(["level", "algorithm"], observed=False)["expanded_nodes"]
        .median()
        .unstack("algorithm")
        .reindex(index=LEVEL_ORDER, columns=ALGORITHM_ORDER)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    table.rename(columns=ALGORITHM_LABELS).plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Nodos expandidos por algoritmo — benchmark Ivo")
    ax.set_xlabel("")
    ax.set_ylabel("Nodos expandidos (mediana, escala logarítmica)")
    ax.set_xticklabels(level_labels(table.index), rotation=0)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Algoritmo", ncol=3)
    save(fig, "06_nodos_expandidos_por_algoritmo")


def plot_greedy_expanded(df):
    greedy = df[(df["algorithm"].eq("greedy")) & df["success_bool"]]
    table = (
        greedy.groupby(["heuristic", "level"], observed=False)["expanded_nodes"]
        .median()
        .unstack("level")
        .reindex(index=HEURISTIC_ORDER, columns=LEVEL_ORDER)
    )
    values = table.to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdYlGn_r",
        norm=LogNorm(vmin=max(float(positive.min()), 1), vmax=float(positive.max())),
    )
    ax.set_title("Nodos expandidos de Greedy por heurística")
    ax.set_xticks(range(len(LEVEL_ORDER)), level_labels(LEVEL_ORDER))
    ax.set_yticks(range(len(HEURISTIC_ORDER)), [heuristic_label(x) for x in HEURISTIC_ORDER])
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = table.iloc[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Nodos expandidos (mediana, escala logarítmica)")
    save(fig, "07_greedy_nodos_expandidos")


def plot_greedy_metric(df, metric, title, colorbar_label, filename):
    greedy = df[(df["algorithm"].eq("greedy")) & df["success_bool"]]
    table = (
        greedy.groupby(["heuristic", "level"], observed=False)[metric]
        .median()
        .unstack("level")
        .reindex(index=HEURISTIC_ORDER, columns=LEVEL_ORDER)
    )
    values = table.to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdYlGn_r",
        norm=LogNorm(vmin=max(float(positive.min()), 1e-6), vmax=float(positive.max())),
    )
    ax.set_title(title)
    ax.set_xticks(range(len(LEVEL_ORDER)), level_labels(LEVEL_ORDER))
    ax.set_yticks(range(len(HEURISTIC_ORDER)), [heuristic_label(x) for x in HEURISTIC_ORDER])
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            value = table.iloc[row, col]
            if np.isfinite(value):
                if metric == "time_sec":
                    label = f"{value:.2f}"
                else:
                    label = f"{value:.0f}"
                ax.text(col, row, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    save(fig, filename)


def plot_timeouts(df):
    table = (
        df.groupby(["level", "algorithm"], observed=False)["timed_out"]
        .sum()
        .unstack("algorithm")
        .reindex(index=LEVEL_ORDER, columns=ALGORITHM_ORDER)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="Reds", vmin=0)
    ax.set_title("Configuraciones que alcanzaron el timeout")
    ax.set_xticks(range(len(ALGORITHM_ORDER)), [ALGORITHM_LABELS[x] for x in ALGORITHM_ORDER])
    ax.set_yticks(range(len(LEVEL_ORDER)), level_labels(LEVEL_ORDER))
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            ax.text(col, row, f"{int(table.iloc[row, col])}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Cantidad de timeouts")
    save(fig, "04_timeouts_por_nivel")


def plot_cost(df):
    data = df[df["success_bool"] & df["algorithm"].isin(["astar", "greedy"])]
    table = (
        data.groupby(["level", "algorithm"], observed=False)["cost"]
        .median()
        .unstack("algorithm")
        .reindex(index=LEVEL_ORDER, columns=["astar", "greedy"])
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    table.plot(kind="bar", ax=ax, color=["#55A868", "#C44E52"], width=0.75)
    ax.set_title("Costo de solución — A* vs. Greedy en Ivo")
    ax.set_xlabel("")
    ax.set_ylabel("Costo (mediana de configuraciones resueltas)")
    ax.set_xticklabels(level_labels(table.index), rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Algoritmo")
    save(fig, "05_costo_astar_vs_greedy")


def main():
    df = load_data()
    plot_success_rate(df)
    plot_astar_success(df)
    plot_astar_expanded(df)
    plot_algorithm_expanded(df)
    plot_greedy_expanded(df)
    plot_greedy_metric(
        df,
        "time_sec",
        "Tiempo de Greedy por heurística",
        "Tiempo (s, mediana, escala logarítmica)",
        "08_greedy_tiempo",
    )
    plot_greedy_metric(
        df,
        "frontier_nodes",
        "Nodos frontera de Greedy por heurística",
        "Nodos frontera (mediana, escala logarítmica)",
        "09_greedy_nodos_frontera",
    )
    plot_timeouts(df)
    plot_cost(df)
    print(f"Generados 9 gráficos en {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
