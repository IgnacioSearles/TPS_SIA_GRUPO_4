"""Figuras para inspeccionar patrones, representaciones y aprendizaje."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def patterns(rows, names, path):
    count = len(rows[0])
    fig, axes = plt.subplots(len(rows), count, squeeze=False,
                             figsize=(max(4, count * 0.7), len(rows) * 1.8))
    for r, (values, name) in enumerate(zip(rows, names)):
        for c, value in enumerate(values):
            axes[r, c].imshow(value.reshape(7, 5), cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(str(c), fontsize=8)
        axes[r, 0].set_ylabel(name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def latent(points, labels, path):
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1])
    for label, point in zip(labels, points):
        ax.annotate(label, point, xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="z1", ylabel="z2", title="Espacio latente")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def learning_curve(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_bce"] for row in history])
    axes[0].set(ylabel="BCE de entrenamiento", xlabel="Época")
    axes[1].plot(epochs, [row["clean_max_pixel_errors"] for row in history])
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[1].set(ylabel="Máximo error por patrón limpio", xlabel="Época")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
