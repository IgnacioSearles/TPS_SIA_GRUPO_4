"""EDA for the learning split only: python -m experiments.analyze_digits."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.digits_data import load_digits


def image_features(X: np.ndarray) -> pd.DataFrame:
    """Describe each image with interpretable intensity and shape measurements."""
    images = X.reshape(-1, 28, 28)
    active = images > 0.1
    active_pixels = active.sum(axis=(1, 2))
    strong_pixels = (images > 0.5).sum(axis=(1, 2))
    row_has_ink = active.any(axis=2)
    col_has_ink = active.any(axis=1)
    blank = active_pixels == 0
    bbox_height = np.where(blank, 0, 28 - row_has_ink[:, ::-1].argmax(axis=1) - row_has_ink.argmax(axis=1))
    bbox_width = np.where(blank, 0, 28 - col_has_ink[:, ::-1].argmax(axis=1) - col_has_ink.argmax(axis=1))
    mass = images.sum(axis=(1, 2))
    center_x = (images.sum(axis=1) @ np.arange(28)) / np.maximum(mass, 1e-12)
    center_y = (images.sum(axis=2) @ np.arange(28)) / np.maximum(mass, 1e-12)
    x = np.arange(28)[None, None, :]
    y = np.arange(28)[None, :, None]
    spread_x = np.sqrt(np.sum(images * (x - center_x[:, None, None]) ** 2, axis=(1, 2)) / np.maximum(mass, 1e-12))
    spread_y = np.sqrt(np.sum(images * (y - center_y[:, None, None]) ** 2, axis=(1, 2)) / np.maximum(mass, 1e-12))
    bbox_area = bbox_width * bbox_height
    # Each fraction has the same denominator, so position differences remain comparable.
    features = pd.DataFrame({
        "pixel_mean": X.mean(axis=1),
        "pixel_std": X.std(axis=1),
        "total_intensity": mass,
        "foreground_mean": np.sum(images * active, axis=(1, 2)) / np.maximum(active_pixels, 1),
        "active_pixels": active_pixels,
        "active_fraction": active_pixels / 784,
        "strong_pixels": strong_pixels,
        "strong_fraction": strong_pixels / 784,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "bbox_aspect_ratio": bbox_width / np.maximum(bbox_height, 1),
        "density_within_bbox": active_pixels / np.maximum(bbox_area, 1),
        "center_x": center_x,
        "center_y": center_y,
        "spread_x": spread_x,
        "spread_y": spread_y,
        "mass_left_fraction": images[:, :, :14].sum(axis=(1, 2)) / np.maximum(mass, 1e-12),
        "mass_top_fraction": images[:, :14, :].sum(axis=(1, 2)) / np.maximum(mass, 1e-12),
        "horizontal_symmetry_error": np.abs(images - images[:, :, ::-1]).mean(axis=(1, 2)),
        "vertical_symmetry_error": np.abs(images - images[:, ::-1, :]).mean(axis=(1, 2)),
        "horizontal_edge_variation": np.abs(np.diff(images, axis=2)).mean(axis=(1, 2)),
        "vertical_edge_variation": np.abs(np.diff(images, axis=1)).mean(axis=(1, 2)),
    })
    return features


def write_eda(X: np.ndarray, y: np.ndarray, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(y, minlength=10)
    features = image_features(X)
    features.insert(0, "label", y)
    features.to_csv(output / "image_features.csv", index=False)
    grouped = features.groupby("label").agg(["mean", "median", "std", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    grouped.columns = [f"{feature}_{stat if isinstance(stat, str) else 'quantile'}" for feature, stat in grouped.columns]
    # Give the two custom quantiles stable names in the exported column headers.
    grouped.columns = [name.replace("_<lambda_0>", "_q1").replace("_<lambda_1>", "_q3") for name in grouped.columns]
    grouped.to_csv(output / "class_features.csv")
    class_stats = []
    for digit in range(10):
        selected = y == digit
        if not selected.any():
            class_stats.append(None)
            continue
        subset = features.loc[selected]
        class_stats.append({
            "pixel_mean": float(subset["pixel_mean"].mean()),
            "active_pixels_median": float(subset["active_pixels"].median()),
            "active_fraction_median": float(subset["active_fraction"].median()),
            "active_fraction_q1": float(subset["active_fraction"].quantile(0.25)),
            "active_fraction_q3": float(subset["active_fraction"].quantile(0.75)),
            "bbox_width_median": float(subset["bbox_width"].median()),
            "bbox_height_median": float(subset["bbox_height"].median()),
            "center_x_median": float(subset["center_x"].median()),
            "center_y_median": float(subset["center_y"].median()),
        })
    summary = {
        "samples": len(y), "image_shape": [28, 28], "class_counts": counts.tolist(),
        "pixel_min": float(X.min()), "pixel_max": float(X.max()),
        "active_threshold": 0.1,
        "blank_images": int((features["active_pixels"] == 0).sum()),
        "duplicate_images": int(len(X) - len(np.unique(X, axis=0))),
        "class_stats": class_stats,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# EDA de dígitos (datos de aprendizaje)", "",
        f"Muestras: {len(y)}. Cada imagen tiene 28 × 28 píxeles en [0, 1].", "",
        "Se considera activo un píxel con valor > 0,1. La caja del trazo es el rectángulo mínimo que contiene los píxeles activos. El centro se calcula ponderando las coordenadas por el valor de cada píxel; las coordenadas van de 0 a 27.", "",
        "| Dígito | Muestras | Proporción | Intensidad media | Píxeles activos (mediana) | Área activa (mediana, Q1–Q3) | Caja ancho × alto (medianas) | Centro x, y (medianas) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for digit, stats in enumerate(class_stats):
        if stats is None:
            lines.append(f"| {digit} | 0 | 0% | — | — | — | — | — |")
            continue
        lines.append(
            f"| {digit} | {counts[digit]} | {counts[digit] / len(y):.2%} | {stats['pixel_mean']:.4f} | "
            f"{stats['active_pixels_median']:.0f} | {stats['active_fraction_median']:.1%} "
            f"({stats['active_fraction_q1']:.1%}–{stats['active_fraction_q3']:.1%}) | "
            f"{stats['bbox_width_median']:.0f} × {stats['bbox_height_median']:.0f} | "
            f"{stats['center_x_median']:.1f}, {stats['center_y_median']:.1f} |"
        )
    lines += [
        "", "La intensidad media es el promedio de los 784 valores de píxel de una imagen, promediado por clase. Incluye el fondo; por eso se complementa con la intensidad del trazo, el área activa y su geometría.",
        "", "## Medidas completas", "",
        "`image_features.csv` contiene 23 medidas por imagen y `class_features.csv` resume cada medida por dígito con media, mediana, desvío y cuartiles. Incluyen intensidad total y del trazo; píxeles > 0,1 y > 0,5; caja, densidad y proporción; centro y dispersión horizontal/vertical; fracción de masa en la mitad izquierda/superior; simetría y variación entre píxeles vecinos.",
        "", "| Familia | Columnas | Interpretación |", "| --- | --- | --- |",
        "| Intensidad | `pixel_mean`, `pixel_std`, `total_intensity`, `foreground_mean` | Brillo global, variación, suma de píxeles y brillo de los activos |",
        "| Área activa | `active_pixels`, `active_fraction`, `strong_pixels`, `strong_fraction` | Cantidad y proporción de píxeles > 0,1 y > 0,5 |",
        "| Forma | `bbox_width`, `bbox_height`, `bbox_area`, `bbox_aspect_ratio`, `density_within_bbox` | Tamaño, proporción y ocupación de la caja del trazo |",
        "| Ubicación | `center_x`, `center_y`, `spread_x`, `spread_y`, `mass_left_fraction`, `mass_top_fraction` | Centro, dispersión y distribución de intensidad |",
        "| Regularidad | `horizontal_symmetry_error`, `vertical_symmetry_error`, `horizontal_edge_variation`, `vertical_edge_variation` | Diferencia al reflejar la imagen y entre píxeles vecinos |",
        "", "Las medidas son descriptivas para el EDA; el MLP recibe directamente los 784 píxeles. La matriz de correlaciones ayuda a ver cuáles medidas aportan información similar.",
        "", f"Imágenes sin píxeles activos: {summary['blank_images']}. Imágenes duplicadas: {summary['duplicate_images']}.",
        "", "Los gráficos muestran distribución, ejemplos, promedios por clase y dispersión del área activa. El test se reserva para la evaluación final.",
    ]
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        lines += ["", f"**Limitación del aprendizaje:** faltan ejemplos de las clases {missing.tolist()}; no se puede aprender a reconocerlas con este archivo."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(10), counts)
    ax.set(xticks=range(10), xlabel="Dígito", ylabel="Muestras", title="Distribución de clases")
    fig.tight_layout(); fig.savefig(output / "class_distribution.png", dpi=150); plt.close(fig)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(10, 8, figsize=(10, 12))
    for digit in range(10):
        available = np.flatnonzero(y == digit)
        selected = rng.choice(available, size=8, replace=len(available) < 8) if len(available) else []
        for ax in axes[digit]:
            ax.axis("off")
        for col, index in enumerate(selected):
            axes[digit, col].imshow(X[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[digit, 0].set_ylabel(str(digit), rotation=0, labelpad=15)
    fig.tight_layout(); fig.savefig(output / "examples.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for digit, ax in enumerate(axes.flat):
        if counts[digit]:
            ax.imshow(X[y == digit].mean(axis=0).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(digit)); ax.axis("off")
    fig.tight_layout(); fig.savefig(output / "mean_images.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot([features.loc[y == digit, "active_fraction"] for digit in range(10) if counts[digit]],
               tick_labels=[str(digit) for digit in range(10) if counts[digit]], showfliers=False)
    ax.set(xlabel="Dígito", ylabel="Fracción de píxeles activos", title="Área activa por clase")
    fig.tight_layout(); fig.savefig(output / "active_area_by_class.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(X.ravel()[::20], bins=50)
    ax.set(xlabel="Valor de píxel", ylabel="Frecuencia (submuestra)", title="Distribución de intensidad")
    fig.tight_layout(); fig.savefig(output / "pixel_distribution.png", dpi=150); plt.close(fig)

    correlation = features.drop(columns="label").corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(correlation)), correlation.index, fontsize=7)
    fig.colorbar(image, ax=ax, label="Correlación de Pearson")
    fig.tight_layout(); fig.savefig(output / "feature_correlations.png", dpi=150); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/digits.csv")
    parser.add_argument("--output", default="reports/digits_eda")
    args = parser.parse_args()
    X, y = load_digits(args.csv)
    write_eda(X, y, Path(args.output))
    print(f"EDA: {args.output}")


if __name__ == "__main__":
    main()
