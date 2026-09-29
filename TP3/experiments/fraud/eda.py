"""Exploratory data analysis figures for Exercise 1 (fraud dataset).

Usage:
    python -m experiments.fraud.eda
    python -m experiments.fraud.eda --csv path/to/fraud_dataset.csv --output reports/fraud_eda
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from experiments.fraud.data import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    FRAUD_LABEL,
    TEACHER_TARGET,
    load_labeled_fraud_dataset,
)
from analysis.plotting import save_figure

ALL_FEATURES = FEATURE_COLUMNS + EXCLUDED_COLUMNS
# Heavily right-skewed columns are easier to read on a logarithmic axis.
LOG_SCALE_COLUMNS = {"amount_usd"}
CLASS_NAMES = {0: "legítima", 1: "fraude"}
CLASS_COLORS = {0: "C0", 1: "C1"}
HISTOGRAM_BINS = 40
TREND_QUANTILES = 20


def feature_label(column: str) -> str:
    return f"{column} (excluida)" if column in EXCLUDED_COLUMNS else column


def histogram_bins(values: pd.Series, column: str) -> np.ndarray:
    """Shared bin edges for a column; log-spaced for skewed columns."""
    if column in LOG_SCALE_COLUMNS:
        return np.geomspace(values.min(), values.max(), HISTOGRAM_BINS + 1)
    if pd.api.types.is_integer_dtype(values) and values.nunique() <= HISTOGRAM_BINS:
        # One bin per integer value; equal-width bins would leave empty gaps between them.
        return np.arange(values.min() - 0.5, values.max() + 1.5)
    return np.histogram_bin_edges(values, bins=HISTOGRAM_BINS)


def feature_grid(n_features: int, columns: int = 3) -> tuple[plt.Figure, list[plt.Axes]]:
    """A grid with one panel per feature; unused panels are removed."""
    rows = math.ceil(n_features / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.4 * rows), squeeze=False,
                                layout="constrained")
    for axis in axes.flat[n_features:]:
        axis.remove()
    return figure, list(axes.flat[:n_features])


def plot_target_distribution(data: pd.DataFrame, path: Path) -> None:
    """BigModel's probability, stacked by the real label, with the classes' overlap marked."""
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, HISTOGRAM_BINS + 1)
    groups = [data.loc[data[FRAUD_LABEL] == label, TEACHER_TARGET] for label in CLASS_NAMES]
    labels = [
        f"{CLASS_NAMES[label]} ({FRAUD_LABEL}={label}): {len(group)} ({len(group) / len(data):.1%})"
        for label, group in zip(CLASS_NAMES, groups)
    ]
    axis.hist(groups, bins=bins, stacked=True, color=list(CLASS_COLORS.values()), label=labels)

    # The gap between the classes shows which BigModel threshold produced the label.
    max_legit, min_fraud = groups[0].max(), groups[1].min()
    boundary = (max_legit + min_fraud) / 2
    axis.axvline(boundary, color="black", linestyle="--", linewidth=1)
    axis.annotate(
        f"máx. legítima {max_legit:.4f}\nmín. fraude {min_fraud:.4f}",
        (boundary, 0.80), xycoords=("data", "axes fraction"), xytext=(-6, 0), textcoords="offset points",
        ha="right", va="top", fontsize=8,
    )

    axis.set_xlim(0, 1)
    axis.set_xlabel(TEACHER_TARGET)
    axis.set_ylabel("transacciones")
    axis.set_title("Probabilidad de BigModel según la etiqueta real")
    axis.legend(loc="upper center")
    save_figure(figure, path)


def plot_correlation_matrices(data: pd.DataFrame, path: Path) -> None:
    """Pearson (linear) next to Spearman (monotonic, rank-based) for every column."""
    columns = ALL_FEATURES + [TEACHER_TARGET, FRAUD_LABEL]
    figure, axes = plt.subplots(1, 2, figsize=(17, 7.5), layout="constrained")
    for axis, method, title in zip(axes, ("pearson", "spearman"), ("Pearson (lineal)", "Spearman (monótona, por rangos)")):
        image = draw_correlation_heatmap(axis, data[columns].corr(method=method))
        axis.set_title(title)
    figure.colorbar(image, ax=axes, shrink=0.8, label="correlación")
    figure.suptitle("Correlaciones entre todas las columnas")
    save_figure(figure, path)


def draw_correlation_heatmap(axis: plt.Axes, correlation: pd.DataFrame):
    """Lower-triangle heatmap annotated with each coefficient."""
    values = correlation.to_numpy()
    lower = np.where(np.triu(np.ones_like(values, dtype=bool), k=1), np.nan, values)
    image = axis.imshow(lower, vmin=-1, vmax=1, cmap="coolwarm")
    for row, column in zip(*np.tril_indices_from(values)):
        coefficient = values[row, column]
        axis.text(column, row, f"{coefficient:.2f}", ha="center", va="center", fontsize=7,
                  color="white" if abs(coefficient) > 0.6 else "black")
    names = correlation.columns
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    return image


def plot_target_correlations(data: pd.DataFrame, path: Path) -> None:
    """Each feature's Pearson and Spearman correlation with BigModel's probability."""
    pearson = data[ALL_FEATURES].corrwith(data[TEACHER_TARGET], method="pearson")
    spearman = data[ALL_FEATURES].corrwith(data[TEACHER_TARGET], method="spearman")
    order = pearson.abs().sort_values(ascending=False).index
    positions = np.arange(len(order))
    bar_height = 0.38

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(positions - bar_height / 2, pearson[order], height=bar_height, color="C0", label="Pearson (lineal)")
    axis.barh(positions + bar_height / 2, spearman[order], height=bar_height, color="C1", label="Spearman (monótona)")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(positions, [feature_label(column) for column in order])
    for tick, column in zip(axis.get_yticklabels(), order):
        tick.set_color("gray" if column in EXCLUDED_COLUMNS else "black")
    axis.invert_yaxis()
    axis.set_xlim(-1, 1)
    axis.set_xlabel(f"correlación con {TEACHER_TARGET}")
    axis.set_title("Correlación de cada feature con la probabilidad objetivo")
    axis.legend(loc="lower right")
    save_figure(figure, path)


def plot_feature_vs_target(data: pd.DataFrame, path: Path, columns: list[str] = ALL_FEATURES) -> None:
    """Scatter of each feature in `columns` against the target, plus the mean target per feature quantile."""
    figure, axes = feature_grid(len(columns))
    for axis, column in zip(axes, columns):
        axis.scatter(data[column], data[TEACHER_TARGET], s=3, alpha=0.08, color="C0",
                     linewidths=0, rasterized=True, label="transacción")
        quantile = pd.qcut(data[column], q=TREND_QUANTILES, duplicates="drop")
        trend = data.groupby(quantile, observed=True)[[column, TEACHER_TARGET]].mean()
        axis.plot(trend[column], trend[TEACHER_TARGET], color="C1", marker="o", markersize=4,
                  label="media por cuantil")
        if column in LOG_SCALE_COLUMNS:
            axis.set_xscale("log")
        axis.set_ylim(0, 1)
        axis.set_title(feature_label(column), fontsize=10)
        axis.set_ylabel("probabilidad BigModel")
    figure.suptitle(f"Relación de cada feature con {TEACHER_TARGET}")
    legend = figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    # The scatter's tiny, faint points would make its legend marker invisible.
    legend.legend_handles[0].set_alpha(1)
    legend.legend_handles[0].set_sizes([20])
    save_figure(figure, path)


def plot_feature_distributions_by_class(data: pd.DataFrame, path: Path) -> None:
    """Density histogram of every feature, one outline per real label."""
    figure, axes = feature_grid(len(ALL_FEATURES))
    for axis, column in zip(axes, ALL_FEATURES):
        bins = histogram_bins(data[column], column)
        for label, name in CLASS_NAMES.items():
            axis.hist(data.loc[data[FRAUD_LABEL] == label, column], bins=bins, density=True,
                      histtype="step", linewidth=2, color=CLASS_COLORS[label], label=name)
        if column in LOG_SCALE_COLUMNS:
            axis.set_xscale("log")
        axis.set_title(feature_label(column), fontsize=10)
        axis.set_ylabel("densidad")
    figure.suptitle("Distribución de cada feature por etiqueta real")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2, title=FRAUD_LABEL)
    save_figure(figure, path)


def plot_feature_scales(data: pd.DataFrame, path: Path) -> None:
    """Z-scored box plots: shows skew and outliers on a common scale."""
    standardized = (data[ALL_FEATURES] - data[ALL_FEATURES].mean()) / data[ALL_FEATURES].std()
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.boxplot(
        [standardized[column] for column in ALL_FEATURES],
        orientation="horizontal",
        tick_labels=[feature_label(column) for column in ALL_FEATURES],
        flierprops={"markersize": 3, "alpha": 0.3},
    )
    axis.invert_yaxis()
    axis.axvline(0, color="gray", linestyle=":", linewidth=1)
    axis.set_xlabel("z-score  (x − media) / desvío")
    axis.set_title("Escala estandarizada y valores atípicos")
    save_figure(figure, path)


def write_eda(data: pd.DataFrame, output: Path) -> None:
    """Write the summary table and every EDA figure into `output`."""
    output.mkdir(parents=True, exist_ok=True)
    summary = data.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    summary.to_csv(output / "data_summary.csv")

    plot_target_distribution(data, output / "target_distribution.png")
    plot_correlation_matrices(data, output / "correlation_matrices.png")
    plot_target_correlations(data, output / "target_correlations.png")
    plot_feature_vs_target(data, output / "feature_vs_target.png")
    plot_feature_vs_target(data, output / "model_features_vs_target.png", columns=FEATURE_COLUMNS)
    plot_feature_distributions_by_class(data, output / "feature_distributions_by_class.png")
    plot_feature_scales(data, output / "feature_scales.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/fraud_dataset.csv")
    parser.add_argument("--output", default="reports/fraud_eda")
    args = parser.parse_args()
    write_eda(load_labeled_fraud_dataset(args.csv), Path(args.output))
    print(f"EDA figures written to {args.output}")


if __name__ == "__main__":
    main()
