"""Generate presentation-ready plots from a Sokoban benchmark CSV.

The script focuses on levels 1--4 and uses ``dead_squares`` for the main
algorithm/heuristic comparisons.  The pruning comparison uses both pruning
modes when they are present in the input file.

Example:
    python3 experiments/generate_benchmark_plots.py \
        --input-csv results/benchmark_results_levels_1_4.csv \
        --output-dir figures/benchmark_1_4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEVELS = ["level-1", "level-2", "level-3", "level-4"]
ALGORITHM_ORDER = ["bfs", "dfs", "iddfs", "astar", "greedy"]
BASE_HEURISTIC_ORDER = [
    "manhattan",
    "hungarian",
    "push_distance_nearest",
    "push_distance",
]

HEURISTIC_COLORS = {
    "manhattan": "#4C72B0",
    "hungarian": "#55A868",
    "push_distance_nearest": "#C44E52",
    "push_distance": "#8172B2",
}

ALGORITHM_COLORS = {
    "bfs": "#1F77B4",
    "dfs": "#FF7F0E",
    "iddfs": "#2CA02C",
    "astar": "#D62728",
    "greedy": "#9467BD",
}

PLAYER_VARIANT_COLORS = {
    "manhattan": "#4C72B0",
    "hungarian": "#4C72B0",
    "push_distance": "#4C72B0",
    "push_distance_nearest": "#4C72B0",
}


def plot_color(value: str) -> str | None:
    if value in PLAYER_VARIANT_COLORS:
        return PLAYER_VARIANT_COLORS[value]
    if value.endswith("_player_all"):
        return "#55A868"
    if value.endswith("_player"):
        return "#DD8452"
    return HEURISTIC_COLORS.get(value)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparative plots from a Sokoban benchmark CSV."
    )
    parser.add_argument(
        "--input-csv",
        default="results/benchmark_results_levels_1_4.csv",
        help="Input benchmark CSV (default: results/benchmark_results_levels_1_4.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/benchmark_1_4",
        help="Directory where PNG and SVG plots are written.",
    )
    return parser.parse_args()


def canonical_heuristic(value: str) -> str:
    """Collapse the explicit ``*_unsolved`` aliases used by the codebase."""
    value = str(value or "").strip().lower()
    return value.removesuffix("_unsolved")


def heuristic_label(value: str) -> str:
    labels = {
        "manhattan": "Manhattan",
        "hungarian": "Hungarian",
        "push_distance_nearest": "Push distance nearest",
        "push_distance": "Push distance",
    }
    if value.endswith("_player_all"):
        base = value.removesuffix("_player_all")
        return f"{labels.get(base, base)} + jugador (todas)"
    if value.endswith("_player"):
        base = value.removesuffix("_player")
        return f"{labels.get(base, base)} + jugador (no resueltas)"
    return labels.get(value, value)


def algorithm_label(value: str) -> str:
    return {
        "bfs": "BFS",
        "dfs": "DFS",
        "iddfs": "IDDFS",
        "astar": "A*",
        "greedy": "Greedy",
    }.get(value, value.upper())


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "level",
        "algorithm",
        "heuristic",
        "pruning_mode",
        "success",
        "timed_out",
        "cost",
        "expanded_nodes",
        "frontier_nodes",
        "time_sec",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    df = df[df["level"].isin(LEVELS)].copy()
    df["algorithm"] = df["algorithm"].astype(str).str.lower()
    df["heuristic"] = df["heuristic"].fillna("").map(canonical_heuristic)
    df["success_bool"] = df["success"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    df["timeout_bool"] = df["timed_out"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    for column in ("cost", "expanded_nodes", "time_sec"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["finished"] = df["success_bool"] & ~df["timeout_bool"]
    return df


def main_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pruning_mode"].eq("dead_squares")].copy()


def successful(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["finished"]].copy()


def available_heuristics(df: pd.DataFrame, algorithm: str = "astar") -> list[str]:
    values = set(df.loc[df["algorithm"].eq(algorithm), "heuristic"])
    ordered = []
    for value in BASE_HEURISTIC_ORDER:
        if value in values:
            ordered.append(value)
        for suffix in ("_player", "_player_all"):
            if value + suffix in values:
                ordered.append(value + suffix)
    ordered.extend(sorted(values - set(ordered) - {""}))
    return ordered


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{filename}.{extension}", bbox_inches="tight")
    plt.close(fig)


def no_data_figure(title: str, message: str, output_dir: Path, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_title(title, pad=20)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    save_figure(fig, output_dir, filename)


def grouped_bar(
    table: pd.DataFrame,
    title: str,
    ylabel: str,
    output_dir: Path,
    filename: str,
    log_scale: bool = False,
    label_func=None,
) -> None:
    levels = [level for level in LEVELS if level in table.index]
    columns = list(table.columns)
    if not levels or not columns:
        no_data_figure(title, "No hay datos terminados para este gráfico.", output_dir, filename)
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(levels))
    width = min(0.8 / max(len(columns), 1), 0.22)
    for index, column in enumerate(columns):
        values = table.reindex(levels)[column].to_numpy(dtype=float)
        offset = (index - (len(columns) - 1) / 2) * width
        color = plot_color(column)
        label = label_func(column) if label_func else column
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.set_xticks(x, [level.replace("level-", "Nivel ") for level in levels])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    save_figure(fig, output_dir, filename)


def plot_success_rate(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby(["level", "algorithm"], observed=False)["success_bool"]
        .mean()
        .mul(100)
        .unstack("algorithm")
        .reindex(index=LEVELS, columns=ALGORITHM_ORDER)
    )
    grouped_bar(
        grouped,
        "Tasa de éxito por algoritmo y nivel\nA* y Greedy: promedio sobre heurísticas disponibles",
        "Ejecuciones exitosas (%)",
        output_dir,
        "01_tasa_exito_por_algoritmo",
    )


def plot_algorithm_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> None:
    data = successful(df)
    grouped = (
        data.groupby(["level", "algorithm"], observed=False)[metric]
        .median()
        .unstack("algorithm")
        .reindex(index=LEVELS, columns=ALGORITHM_ORDER)
    )
    grouped_bar(grouped, title, ylabel, output_dir, filename, log_scale=True)


def plot_astar_greedy_metric(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Compare A* and Greedy with one linear-scale panel per level."""
    data = successful(df[df["algorithm"].isin(["astar", "greedy"])])
    table = (
        data.groupby(["level", "algorithm"], observed=False)[metric]
        .median()
        .unstack("algorithm")
        .reindex(index=LEVELS, columns=["astar", "greedy"])
    )

    if table.dropna(how="all").empty:
        no_data_figure(title, "No hay resultados terminados para este gráfico.", output_dir, filename)
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    axes = axes.flat
    for ax, level in zip(axes, LEVELS):
        values = table.loc[level]
        values.plot(
            kind="bar",
            ax=ax,
            color=[ALGORITHM_COLORS["astar"], ALGORITHM_COLORS["greedy"]],
            width=0.65,
        )
        ax.set_title(level.replace("level-", "Nivel "))
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(["A*", "Greedy"], fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def plot_uninformed_frontier(df: pd.DataFrame, output_dir: Path) -> None:
    """Compare frontier nodes for BFS, DFS and IDDFS by level."""
    data = successful(df[df["algorithm"].isin(["bfs", "dfs", "iddfs"])])
    table = (
        data.groupby(["level", "algorithm"], observed=False)["frontier_nodes"]
        .median()
        .unstack("algorithm")
        .reindex(index=LEVELS, columns=["bfs", "dfs", "iddfs"])
    )

    title = "Nodos frontera — búsquedas no informadas"
    filename = "16_nodos_frontera_no_informados"
    if table.dropna(how="all").empty:
        no_data_figure(title, "No hay resultados terminados para este gráfico.", output_dir, filename)
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, level in zip(axes.flat, LEVELS):
        table.loc[level].plot(
            kind="bar",
            ax=ax,
            color=[ALGORITHM_COLORS["bfs"], ALGORITHM_COLORS["dfs"], ALGORITHM_COLORS["iddfs"]],
            width=0.7,
        )
        ax.set_title(level.replace("level-", "Nivel "))
        ax.set_xlabel("")
        ax.set_ylabel("Nodos frontera (mediana)")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ax.patches[-3:], ["BFS", "DFS", "IDDFS"], fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def plot_informed_frontier_heatmap(df: pd.DataFrame, algorithm: str, output_dir: Path) -> None:
    """Show median frontier nodes by heuristic and level for A* or Greedy."""
    data = successful(df[df["algorithm"].eq(algorithm)])
    heuristics = available_heuristics(df, algorithm=algorithm)
    table = (
        data.groupby(["heuristic", "level"], observed=False)["frontier_nodes"]
        .median()
        .unstack("level")
        .reindex(index=heuristics, columns=LEVELS)
    )
    title = f"Nodos frontera — {algorithm_label(algorithm)} por heurística"
    filename = f"{18 if algorithm == 'astar' else 19}_nodos_frontera_{algorithm}"
    if table.dropna(how="all").empty:
        no_data_figure(title, "No hay resultados terminados para este gráfico.", output_dir, filename)
        return

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.5 * len(heuristics) + 2)))
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(LEVELS)), [level.replace("level-", "Nivel ") for level in LEVELS])
    ax.set_yticks(range(len(heuristics)), [heuristic_label(h) for h in heuristics])
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Nodos frontera (mediana)")
    save_figure(fig, output_dir, filename)


def plot_frontier_expanded_ratio(df: pd.DataFrame, output_dir: Path) -> None:
    """Compare frontier size relative to expanded nodes by algorithm."""
    data = successful(df).copy()
    data = data[data["expanded_nodes"].gt(0)]
    data["frontier_expanded_ratio"] = data["frontier_nodes"] / data["expanded_nodes"]
    grouped = (
        data.groupby(["level", "algorithm"], observed=False)["frontier_expanded_ratio"]
        .median()
        .unstack("algorithm")
        .reindex(index=LEVELS, columns=ALGORITHM_ORDER)
    )
    grouped_bar(
        grouped,
        "Relación entre nodos frontera y nodos expandidos",
        "Frontera / expandidos (mediana)",
        output_dir,
        "20_relacion_frontera_expandidos",
        label_func=algorithm_label,
    )


def plot_uninformed_level_one_cost(df: pd.DataFrame, output_dir: Path) -> None:
    """Compare solution cost and search effort for uninformed algorithms on level 1."""
    data = successful(
        df[(df["level"].eq("level-1")) & df["algorithm"].isin(["bfs", "dfs", "iddfs"])]
    )
    data = data.set_index("algorithm").reindex(["bfs", "dfs", "iddfs"])
    if data["cost"].dropna().empty:
        no_data_figure(
            "Costo de solución en Nivel 1 — búsquedas no informadas",
            "No hay soluciones terminadas para este gráfico.",
            output_dir,
            "21_costo_nivel_1_no_informados",
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    data["cost"].plot(
        kind="bar",
        ax=axes[0],
        color=[ALGORITHM_COLORS[a] for a in ["bfs", "dfs", "iddfs"]],
        width=0.7,
    )
    axes[0].set_title("Costo de la solución")
    axes[0].set_ylabel("Movimientos")
    axes[0].set_xlabel("")
    axes[0].set_xticklabels(["BFS", "DFS", "IDDFS"], rotation=0)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].scatter(
        data["expanded_nodes"],
        data["cost"],
        s=70,
        color=[ALGORITHM_COLORS[a] for a in ["bfs", "dfs", "iddfs"]],
    )
    for algorithm, row in data.dropna(subset=["expanded_nodes", "cost"]).iterrows():
        axes[1].annotate(algorithm_label(algorithm), (row["expanded_nodes"], row["cost"]), xytext=(6, 5), textcoords="offset points")
    axes[1].set_title("Costo contra nodos expandidos")
    axes[1].set_xlabel("Nodos expandidos")
    axes[1].set_ylabel("Movimientos")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Nivel 1 — costo de solución en búsquedas no informadas")
    fig.tight_layout()
    save_figure(fig, output_dir, "21_costo_nivel_1_no_informados")


def heuristic_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
    output_dir: Path,
    cmap: str = "viridis",
    center: float | None = None,
    value_format: str = ".0f",
) -> None:
    data = successful(df[df["algorithm"].eq("astar")])
    heuristics = available_heuristics(df)
    if not heuristics:
        no_data_figure(title, "El CSV no contiene resultados de A* con heurística.", output_dir, filename)
        return
    table = (
        data.groupby(["heuristic", "level"], observed=False)[metric]
        .median()
        .unstack("level")
        .reindex(index=heuristics, columns=LEVELS)
    )
    if table.dropna(how="all").empty:
        no_data_figure(title, "No hay soluciones terminadas para este gráfico.", output_dir, filename)
        return

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.42 * len(heuristics) + 2)))
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    if center is not None:
        finite = values[np.isfinite(values)]
        if finite.size:
            limit = max(abs(float(finite.min() - center)), abs(float(finite.max() - center)))
            image.set_clim(center - limit, center + limit)
    ax.set_xticks(range(len(LEVELS)), [level.replace("level-", "Nivel ") for level in LEVELS])
    ax.set_yticks(range(len(heuristics)), [heuristic_label(h) for h in heuristics])
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                text_color = "white" if luminance < 0.48 else "black"
                ax.text(
                    col,
                    row,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )
    fig.colorbar(image, ax=ax, label="Valor")
    save_figure(fig, output_dir, filename)


def plot_solution_cost(df: pd.DataFrame, output_dir: Path) -> None:
    plot_algorithm_metric(
        df,
        "cost",
        "Costo mediano de las soluciones por algoritmo\nA* y Greedy: mediana sobre heurísticas disponibles",
        "Movimientos",
        "05_costo_solucion_por_algoritmo",
        output_dir,
    )


def plot_cost_vs_nodes(df: pd.DataFrame, level: str, filename: str, output_dir: Path) -> None:
    data = successful(df[df["level"].eq(level)])
    # DFS on level 4 is an extreme outlier (1,681,570 moves and more than
    # 14 million expanded nodes), which makes the other configurations
    # unreadable in this presentation scatter plot. Keep it in the CSV and
    # all other plots; omit it only from this visualization.
    data = data[~data.set_index(["algorithm", "level"]).index.isin({("dfs", "level-4")})]
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = False
    annotation_offsets = [(6, 7), (6, -11), (-6, 7), (-6, -11), (10, 0), (-10, 0)]
    for algorithm in ALGORITHM_ORDER:
        subset = data[data["algorithm"].eq(algorithm)]
        if subset.empty:
            continue
        ax.scatter(
            subset["expanded_nodes"],
            subset["cost"],
            label=algorithm_label(algorithm),
            color=ALGORITHM_COLORS[algorithm],
            alpha=0.75,
            s=48,
        )
        for point_index, (_, row) in enumerate(subset.iterrows()):
            if algorithm in {"astar", "greedy"} and row["heuristic"]:
                label = heuristic_label(row["heuristic"])
            else:
                label = algorithm_label(algorithm)
            offset_x, offset_y = annotation_offsets[point_index % len(annotation_offsets)]
            ax.annotate(
                label,
                (row["expanded_nodes"], row["cost"]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=7,
                alpha=0.85,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.7, "linewidth": 0},
            )
        plotted = True
    if not plotted:
        no_data_figure(f"Costo contra nodos expandidos ({level})", "No hay soluciones terminadas.", output_dir, filename)
        return
    ax.set_xscale("log")
    ax.set_xlabel("Nodos expandidos (escala logarítmica)")
    ax.set_ylabel("Costo de la solución (movimientos)")
    ax.set_title(
        f"Compromiso entre costo y esfuerzo de búsqueda — {level.replace('level-', 'Nivel ')}\n"
        "Etiquetas: heurística utilizada"
    )
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, output_dir, filename)


def plot_pruning(df: pd.DataFrame, output_dir: Path) -> None:
    data = successful(df[df["algorithm"].isin(["bfs", "astar"])])
    data = data[(data["algorithm"].eq("bfs")) | (data["heuristic"].eq("hungarian"))]
    if data.empty or data["pruning_mode"].nunique() < 2:
        no_data_figure(
            "Comparación de modos de poda",
            "El CSV no contiene resultados exitosos para ambos modos de poda.",
            output_dir,
            "07_comparacion_modos_poda",
        )
        return
    table = data.groupby(["level", "algorithm", "pruning_mode"], observed=False)[["expanded_nodes", "time_sec"]].median()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, ylabel in zip(axes, ["expanded_nodes", "time_sec"], ["Nodos expandidos", "Tiempo (s)"]):
        pivot = table[metric].unstack(["algorithm", "pruning_mode"]).reindex(LEVELS)
        pivot.columns = [
            (
                f"A* + Hungarian / {pruning}"
                if algorithm == "astar"
                else f"BFS / {pruning}"
            )
            for algorithm, pruning in pivot.columns
        ]
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(ylabel)
        ax.set_xlabel("")
        ax.set_ylabel(f"{ylabel} (escala logarítmica)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Algoritmo / poda", fontsize=8)
    fig.suptitle("Efecto del modo de poda")
    save_figure(fig, output_dir, "07_comparacion_modos_poda")


def plot_player_family(df: pd.DataFrame, family: str, number: str, output_dir: Path) -> None:
    available = available_heuristics(df)
    variants = [h for h in available if h == family or h.startswith(family + "_")]
    if not variants:
        no_data_figure(
            f"Variantes de jugador: {heuristic_label(family)}",
            "El CSV no contiene esta familia de heurísticas.",
            output_dir,
            f"{number}_astar_player_variants_{family}",
        )
        return
    subset = successful(df[(df["algorithm"].eq("astar")) & df["heuristic"].isin(variants)])
    table = (
        subset.groupby(["level", "heuristic"], observed=False)["expanded_nodes"]
        .median()
        .unstack("heuristic")
        .reindex(index=LEVELS, columns=variants)
    )
    grouped_bar(
        table,
        f"A*: variantes de jugador para {heuristic_label(family)}",
        "Nodos expandidos (mediana)",
        output_dir,
        f"{number}_astar_player_variants_{family}",
        log_scale=False,
        label_func=heuristic_label,
    )


def improvement_table(df: pd.DataFrame, metric: str, algorithm: str = "astar") -> pd.DataFrame:
    data = successful(df)
    bfs = data[data["algorithm"].eq("bfs")].groupby("level")[metric].median()
    target = data[data["algorithm"].eq(algorithm)]
    target = target.groupby(["heuristic", "level"])[metric].median().unstack("level")
    target = target.reindex(columns=LEVELS)
    baseline = bfs.reindex(LEVELS)
    return target.rsub(baseline, axis="columns").div(baseline, axis="columns").mul(100)


def plot_improvement(df: pd.DataFrame, metric: str, title: str, filename: str, output_dir: Path, algorithm: str = "astar") -> None:
    table = improvement_table(df, metric, algorithm)
    heuristics = available_heuristics(df, algorithm=algorithm)
    table = table.reindex(index=heuristics, columns=LEVELS)
    if table.dropna(how="all").empty:
        no_data_figure(title, f"No hay pares BFS/{algorithm.upper()} terminados para calcular la mejora.", output_dir, filename)
        return
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.42 * len(heuristics) + 2)))
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = max(abs(float(finite.min())), abs(float(finite.max())), 1) if finite.size else 1
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(LEVELS)), [level.replace("level-", "Nivel ") for level in LEVELS])
    ax.set_yticks(range(len(heuristics)), [heuristic_label(h) for h in heuristics])
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.1f}%", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Mejora respecto de BFS (%)")
    save_figure(fig, output_dir, filename)


def plot_greedy_cost_loss(df: pd.DataFrame, output_dir: Path) -> None:
    """Show how much Greedy's solution cost exceeds BFS per level/heuristic."""
    data = successful(df)
    bfs = data[data["algorithm"].eq("bfs")].groupby("level")["cost"].median()
    greedy = data[data["algorithm"].eq("greedy")]
    greedy = greedy.groupby(["heuristic", "level"])["cost"].median().unstack("level")
    greedy = greedy.reindex(columns=LEVELS)
    baseline = bfs.reindex(LEVELS)
    table = greedy.subtract(baseline, axis="columns").div(baseline, axis="columns").mul(100)
    heuristics = available_heuristics(df, algorithm="greedy")
    table = table.reindex(index=heuristics, columns=LEVELS)
    if table.dropna(how="all").empty:
        no_data_figure(
            "Pérdida relativa de costo de Greedy respecto de BFS",
            "No hay pares BFS/Greedy terminados para calcular la pérdida.",
            output_dir,
            "13_perdida_relativa_costo_greedy_vs_bfs",
        )
        return

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.42 * len(heuristics) + 2)))
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    maximum = max(float(finite.max()), 1) if finite.size else 1
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=maximum)
    ax.set_xticks(range(len(LEVELS)), [level.replace("level-", "Nivel ") for level in LEVELS])
    ax.set_yticks(range(len(heuristics)), [heuristic_label(h) for h in heuristics])
    ax.set_title("Pérdida relativa de costo de Greedy respecto de BFS")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.1f}%", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Costo adicional respecto de BFS (%)")
    save_figure(fig, output_dir, "13_perdida_relativa_costo_greedy_vs_bfs")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe el CSV de entrada: {input_path}. "
            "Usá --input-csv para indicar la ruta correcta."
        )

    df = main_rows(load_data(str(input_path)))
    if df.empty:
        raise ValueError("El CSV no contiene filas de niveles 1–4 con dead_squares.")

    plot_success_rate(df, output_dir)
    plot_algorithm_metric(df, "expanded_nodes", "Nodos expandidos por algoritmo\nA* y Greedy: mediana sobre heurísticas disponibles", "Nodos expandidos (mediana, escala logarítmica)", "02_nodos_expandidos_por_algoritmo", output_dir)
    plot_astar_greedy_metric(df, "expanded_nodes", "Nodos expandidos — A* vs. Greedy", "Nodos expandidos (mediana)", "14_nodos_no_informados", output_dir)
    plot_astar_greedy_metric(df, "time_sec", "Tiempo de cómputo — A* vs. Greedy", "Tiempo (s, mediana)", "15_tiempo_no_informados", output_dir)
    plot_uninformed_frontier(df, output_dir)
    plot_informed_frontier_heatmap(df, "astar", output_dir)
    plot_informed_frontier_heatmap(df, "greedy", output_dir)
    plot_frontier_expanded_ratio(df, output_dir)
    plot_uninformed_level_one_cost(df, output_dir)
    heuristic_heatmap(df, "expanded_nodes", "A*: nodos expandidos por heurística", "03_astar_heuristicas_nodos_expandidos", output_dir, cmap="viridis")
    heuristic_heatmap(df, "time_sec", "A*: tiempo por heurística", "04_astar_heuristicas_tiempo", output_dir, cmap="plasma", value_format=".3f")
    plot_solution_cost(df, output_dir)
    plot_cost_vs_nodes(df, "level-2", "06a_costo_vs_nodos_nivel_2", output_dir)
    plot_cost_vs_nodes(df, "level-4", "06b_costo_vs_nodos_nivel_4", output_dir)
    plot_pruning(load_data(str(input_path)), output_dir)
    plot_player_family(df, "manhattan", "08", output_dir)
    plot_player_family(df, "hungarian", "09", output_dir)
    plot_player_family(df, "push_distance", "10", output_dir)
    plot_improvement(df, "time_sec", "Mejora porcentual de tiempo de A* respecto de BFS", "11_mejora_porcentual_tiempo_astar_vs_bfs", output_dir, algorithm="astar")
    plot_improvement(df, "expanded_nodes", "Mejora porcentual de nodos de A* respecto de BFS", "12_mejora_porcentual_nodos_astar_vs_bfs", output_dir, algorithm="astar")
    plot_improvement(df, "expanded_nodes", "Mejora porcentual de nodos de Greedy respecto de BFS", "22_mejora_porcentual_nodos_greedy_vs_bfs", output_dir, algorithm="greedy")
    plot_greedy_cost_loss(df, output_dir)

    print(f"Generados 23 gráficos en: {output_dir.resolve()}")
    missing_player = not any("_player" in value for value in df["heuristic"])
    if missing_player:
        print("Aviso: el CSV no contiene variantes _player; esos gráficos se generaron como 'sin datos'.")


if __name__ == "__main__":
    main()
