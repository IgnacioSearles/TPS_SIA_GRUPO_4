"""Summary figures for the ejercicio 2 and 3 slides, read from the saved study CSVs.

Every number comes from a file another experiment wrote; nothing is trained here.
The studies' own figures plot every curve of every seed, which is right for an
appendix but unreadable on a slide. These show one mean (± std across seeds) per
condition instead.

Run from TP3:
    python -m experiments.digits.presentation_figures
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT = Path("reports/presentation")
TARGET = 98.0
GREY = "#b0b0b0"


def _comma(value, decimals=2):
    return f"{value:.{decimals}f}".replace(".", ",")


def _bars(ax, labels, means, stds, highlight=None, decimals=2, xlim=None):
    """Horizontal bars with the value written at the tip; `highlight` indexes are blue."""
    colors = ["C0" if highlight is None or i in highlight else GREY for i in range(len(labels))]
    positions = np.arange(len(labels))
    ax.barh(positions, means, xerr=stds, color=colors, capsize=3, error_kw={"elinewidth": 1})
    for y, (mean, std) in enumerate(zip(means, stds)):
        ax.annotate(f"{_comma(mean, decimals)} %", (mean + (std if np.isfinite(std) else 0), y),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=10)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    if xlim:
        ax.set_xlim(*xlim)
    ax.spines[["top", "right"]].set_visible(False)


def optimizers():
    seeds = pd.read_csv("results/digits_seeds/runs.csv")
    adam = pd.read_csv("results/digits_adam/runs.csv")
    table = pd.concat([seeds, adam], ignore_index=True)
    names = {"sgd_lr0.1_m0": "SGD  η=0,1",
             "momentum_lr0.1_m0.5": "Momentum 0,5  η=0,1",
             "momentum_lr0.1_m0.9": "Momentum 0,9  η=0,1",
             "adam_lr0.0003_m0": "Adam  η=0,0003",
             "adam_lr0.001_m0": "Adam  η=0,001",
             "adam_lr0.003_m0": "Adam  η=0,003"}
    grouped = table.groupby("candidate", sort=False).agg(
        acc=("validation_accuracy", "mean"), std=("validation_accuracy", "std"),
        epoch=("best_epoch", "mean"), epoch_std=("best_epoch", "std"),
        seeds=("seed", "nunique")).reindex(list(names))
    grouped = grouped.dropna(subset=["acc"])
    labels = [names[c] for c in grouped.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    _bars(axes[0], labels, 100 * grouped.acc.values, 100 * grouped["std"].values,
          highlight=set(range(len(labels))), xlim=(94, 97.6))
    axes[0].set(xlabel=f"Accuracy de validación (%), media de {int(grouped.seeds.min())} semillas")
    axes[1].barh(np.arange(len(labels)), grouped.epoch.values,
                 xerr=grouped.epoch_std.fillna(0).values, capsize=3,
                 color=["C0" if "Momentum 0,9" in l else GREY for l in labels])
    for y, value, deviation in zip(np.arange(len(labels)), grouped.epoch.values,
                                   grouped.epoch_std.fillna(0).values):
        axes[1].annotate(f"{value:.0f}", (value + deviation, y), xytext=(6, 0),
                         textcoords="offset points", va="center", fontsize=10)
    axes[1].set_yticks(np.arange(len(labels)), [""] * len(labels))
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Época del mejor modelo (media ± DE)", xlim=(0, 600))
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ej2_optimizadores.png", dpi=170)
    plt.close(fig)
    return grouped


def activations():
    table = pd.read_csv("results/digits_activations/runs.csv")
    order = ["tanh", "sigmoid", "relu"]
    grouped = table.groupby("activation").agg(
        acc=("validation_accuracy", "mean"), std=("validation_accuracy", "std"),
        lo=("validation_accuracy", "min"), hi=("validation_accuracy", "max"),
        seeds=("seed", "nunique")).reindex(order).dropna(subset=["acc"])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for y, activation in enumerate(grouped.index):
        values = 100 * table[table.activation == activation].validation_accuracy.values
        ax.scatter(values, np.full(len(values), y), s=60, alpha=0.8,
                   color="C0" if activation == "tanh" else GREY, zorder=3)
        mean = 100 * grouped.acc[activation]
        deviation = 100 * grouped.loc[activation, "std"]
        ax.errorbar(mean, y, xerr=deviation, fmt="D", color="#222", capsize=4,
                    markersize=5, zorder=4)
        ax.annotate(f"media {_comma(mean)} %", (101.5, y),
                    va="center", fontsize=10, clip_on=False)
    ax.set_yticks(range(len(grouped)), [{"tanh": "tanh", "sigmoid": "sigmoide", "relu": "ReLU"}[a]
                                         for a in grouped.index])
    ax.invert_yaxis()
    # Reserve a label column so the means remain readable beside wide error bars.
    ax.set(xlim=(69, 114), xlabel=f"Accuracy de validación (%); punto por semilla y ◆ media ± DE ({int(grouped.seeds.min())} semillas)")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ej2_activaciones.png", dpi=170)
    plt.close(fig)
    return grouped


def architectures():
    width = pd.read_csv("results/digits_width_sweep/architecture_summary.csv")
    depth = pd.read_csv("results/digits_depth_sweep/depth_summary.csv")
    width["neurons"] = width.architecture.str.split("-").str[1].astype(int)
    width = width.sort_values("neurons")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    width["position"] = np.arange(len(width))
    axes[0].errorbar(width.position, 100 * width.validation_accuracy_mean,
                     yerr=100 * width.validation_accuracy_std, marker="o", capsize=4)
    axes[0].set(xticks=width.position.tolist(), xticklabels=width.neurons.tolist(), xlim=(-0.4, len(width) - 0.6),
                xlabel="Neuronas por capa (2 capas ocultas)", ylabel="Accuracy de validación (%)",
                title="Ancho")
    axes[1].errorbar(depth.hidden_layers, 100 * depth.validation_accuracy_mean,
                     yerr=100 * depth.validation_accuracy_std, marker="o", capsize=4)
    axes[1].set(xticks=depth.hidden_layers.tolist(), xlabel="Capas ocultas (64 neuronas cada una)",
                title="Profundidad")
    for ax, frame, x in [(axes[0], width, "position"), (axes[1], depth, "hidden_layers")]:
        for _, row in frame.iterrows():
            ax.annotate(f"{row.parameters / 1000:.0f} k", (row[x], 100 * row.validation_accuracy_mean),
                        xytext=(9, -3), textcoords="offset points", ha="left", fontsize=8,
                        color="#666")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.99, 0.01, "Etiqueta gris: parámetros. 3 semillas por punto.", ha="right", fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(OUTPUT / "ej2_arquitecturas.png", dpi=170)
    plt.close(fig)


def data_vs_techniques():
    frame = pd.read_csv("reports/digits_e3_datos_vs_tecnicas/comparison.csv")
    runs = pd.read_csv("reports/digits_e3_datos_vs_tecnicas/runs.csv")
    names = {"Solo digits.csv, sin técnicas": ("digits.csv", "sin técnicas"),
             "Solo digits.csv, balanceo y augmentación": ("digits.csv", "balanceo + augmentación"),
             "Combinado, sin técnicas": ("digits + more_digits", "sin técnicas"),
             "Combinado, balanceo y augmentación": ("digits + more_digits", "balanceo + augmentación")}
    frame["data"] = frame.step.map(lambda s: names[s][0])
    frame["technique"] = frame.step.map(lambda s: names[s][1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [3, 2]})
    groups = ["digits.csv", "digits + more_digits"]
    techniques = ["sin técnicas", "balanceo + augmentación"]
    x = np.arange(len(groups))
    for offset, technique, color in [(-0.2, techniques[0], GREY), (0.2, techniques[1], "C0")]:
        rows = frame[frame.technique == technique].set_index("data").loc[groups]
        means = 100 * rows.validation_accuracy_mean.values
        axes[0].bar(x + offset, means, width=0.38, yerr=100 * rows.validation_accuracy_std.values,
                    capsize=3, color=color, label=technique)
        for xi, mean in zip(x + offset, means):
            axes[0].annotate(f"{_comma(mean)} %", (xi, mean), xytext=(0, 5), textcoords="offset points",
                             ha="center", fontsize=10)
        recall8 = 100 * rows.recall_8_mean.values
        recall8_std = np.array([
            100 * runs.loc[runs.step == step, "recall_8"].std(ddof=1)
            for step in rows.step
        ])
        axes[1].bar(x + offset, recall8, width=0.38, yerr=recall8_std, capsize=3,
                    color=color)
        for xi, value, deviation in zip(x + offset, recall8, recall8_std):
            axes[1].annotate(f"{value:.0f} %", (xi, value + deviation), xytext=(0, 5), textcoords="offset points",
                             ha="center", fontsize=10)
    axes[0].axhline(TARGET, color="C3", linestyle="--", linewidth=1)
    axes[0].annotate("objetivo 98 %", (0.5, TARGET), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=9, color="C3")
    axes[0].set(xticks=x, xticklabels=groups, ylim=(88, 102), ylabel="Accuracy de validación (%)",
                title="Accuracy (validación con los 10 dígitos)")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].set(xticks=x, xticklabels=groups, ylim=(0, 105), ylabel="Recall del 8 (%)", title="Recall del 8")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.99, 0.01, "Barras de error: ±1 DE entre 3 semillas.", ha="right", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT / "ej3_datos_vs_tecnicas.png", dpi=170)
    plt.close(fig)
    return frame


def factorial():
    frame = pd.read_csv("reports/digits_e3_factorial/comparison.csv")
    runs = pd.read_csv("reports/digits_e3_factorial/runs.csv")
    labels = ["Ninguna", "Solo balanceo", "Solo augmentación", "Ambas"]
    metrics = [("validation_accuracy_mean", "Accuracy"), ("recall_5_mean", "Recall del 5"),
               ("recall_8_mean", "Recall del 8")]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(metrics))
    width = 0.2
    colors = [GREY, "#8fb3d1", "#4f8fc0", "C0"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        row = frame.iloc[i]
        values = [100 * row[key] for key, _ in metrics]
        deviations = [100 * runs.loc[runs.step == row.step, key.removesuffix("_mean")].std(ddof=1)
                      for key, _ in metrics]
        bars = ax.bar(x + (i - 1.5) * width, values, width=width * 0.95,
                      yerr=deviations, capsize=3, color=color, label=label)
        for bar, value, deviation in zip(bars, values, deviations):
            ax.annotate(_comma(value, 1), (bar.get_x() + bar.get_width() / 2, value + deviation), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.5)
    ax.set(xticks=x, xticklabels=[name for _, name in metrics], ylim=(80, 100.5), ylabel="% en validación")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.99, 0.01, "Barras de error: ±1 DE entre 3 semillas.", ha="right", fontsize=8, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT / "ej3_factorial.png", dpi=170)
    plt.close(fig)


def softmax_comparison():
    """Compare sigmoid/MSE with softmax/CE on both the earlier and winning architecture."""
    small = pd.read_csv("reports/digits_training_ablation/runs.csv")
    wide = pd.read_csv("reports/digits_e3_softmax_best_arch/runs.csv")
    names = ["Sigmoide + MSE", "Softmax + CE", "Softmax + CE + Adam"]
    steps = [
        "Referencia: sigmoide + MSE + momentum",
        "+ softmax y entropía cruzada",
        "+ Adam",
    ]
    records = []
    for architecture, frame in [("784-64-32-10", small), ("784-128-128-10", wide)]:
        for name, step in zip(names, steps):
            subset = frame.loc[frame.step == step]
            if subset.empty:
                raise ValueError(f"No runs for {architecture}: {step}")
            records.append({
                "label": f"{architecture}\n{name}",
                "accuracy_mean": subset.validation_accuracy.mean(),
                "accuracy_std": subset.validation_accuracy.std(ddof=1),
                "f1_mean": subset.validation_macro_f1.mean(),
                "f1_std": subset.validation_macro_f1.std(ddof=1),
            })

    summary = pd.DataFrame(records)
    positions = np.arange(len(summary))
    colors = ["#a8a8a8", "#4f8fc0", "#255c85"] * 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, mean_key, std_key, scale, xlabel, xlim in [
        (axes[0], "accuracy_mean", "accuracy_std", 100, "Accuracy de validación (%)", (97.85, 98.95)),
        (axes[1], "f1_mean", "f1_std", 1, "F1 macro de validación", (0.970, 0.990)),
    ]:
        means = scale * summary[mean_key].to_numpy()
        deviations = scale * summary[std_key].fillna(0).to_numpy()
        for y, (mean, deviation, color) in enumerate(zip(means, deviations, colors)):
            ax.errorbar(mean, y, xerr=deviation, fmt="o", color=color,
                        capsize=4, markersize=6, linewidth=1.4)
            label = f"{mean:.2f}" if scale == 100 else f"{mean:.4f}"
            ax.annotate(label, (mean + deviation, y), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#444")
        ax.set(xlim=xlim, xlabel=xlabel)
        ax.grid(axis="x", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(2.5, color="#777", linewidth=0.8, alpha=0.5)
    axes[0].set_yticks(positions, summary.label)
    axes[0].invert_yaxis()
    fig.suptitle("Softmax + entropía cruzada: comparación en dos arquitecturas")
    fig.text(0.99, 0.01, "Media ±1 DE muestral; 3 semillas por condición. Solo validación.",
             ha="right", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUTPUT / "ej3_softmax_comparison.png", dpi=180)
    plt.close(fig)
    return summary


def expanded_architectures():
    frame = pd.read_csv("reports/digits_e3_architectures/architecture_summary.csv")
    frame = frame.sort_values(["hidden_layers", "parameters"])
    labels = [a.replace("784-", "").replace("-10", "") for a in frame.architecture]
    best = int(np.argmax(frame.validation_accuracy_mean.values))
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _bars(ax, labels, 100 * frame.validation_accuracy_mean.values, 100 * frame.validation_accuracy_std.values,
          highlight={best}, xlim=(96.5, 99.2))
    ax.axvline(TARGET, color="C3", linestyle="--", linewidth=1)
    ax.set(xlabel="Accuracy de validación (%), media de 3 semillas", ylabel="Capas ocultas")
    fig.tight_layout()
    fig.savefig(OUTPUT / "ej3_arquitecturas.png", dpi=170)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(optimizers())
    print(activations())
    architectures()
    print(data_vs_techniques()[["step", "validation_accuracy_mean", "validation_accuracy_std", "recall_8_mean"]])
    factorial()
    expanded_architectures()
    print(softmax_comparison())
    print(f"Figuras en {OUTPUT}")


if __name__ == "__main__":
    main()
