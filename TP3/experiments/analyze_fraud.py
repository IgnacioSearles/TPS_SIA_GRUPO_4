"""Generate the exploratory-data-analysis report for Exercise 1.

Usage:
    python -m experiments.analyze_fraud
    python -m experiments.analyze_fraud --csv path/to/fraud_dataset.csv --output reports/fraud_eda
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.fraud_data import (
    FEATURE_COLUMNS,
    EXCLUDED_COLUMNS,
    TEACHER_TARGET,
    dataset_profile,
    load_fraud_dataset,
)


def write_report(data: pd.DataFrame, output: Path) -> None:
    """Write a compact Markdown report plus distribution/correlation figures."""
    output.mkdir(parents=True, exist_ok=True)
    profile = dataset_profile(data)
    summary: pd.DataFrame = profile["summary"]  # type: ignore[assignment]
    all_correlations: pd.Series = profile["correlation_with_teacher"]  # type: ignore[assignment]

    (output / "data_summary.csv").write_text(summary.to_csv(), encoding="utf-8")
    correlation_table = ["| Variable | Correlación |", "| --- | ---: |"]
    correlation_table.extend(f"| {name} | {value:.6f} |" for name, value in all_correlations[FEATURE_COLUMNS + EXCLUDED_COLUMNS].items())
    range_table = ["| Columna | Mínimo | Mediana | P99 | Máximo |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in FEATURE_COLUMNS + EXCLUDED_COLUMNS + [TEACHER_TARGET]:
        row = summary.loc[name]
        range_table.append(
            f"| {name} | {row['min']:.3f} | {row['50%']:.3f} | {row['99%']:.3f} | {row['max']:.3f} |"
        )
    lines = [
        "# Exploratory data analysis — fraud dataset",
        "",
        "Fuente de definiciones: [documentación del dataset](../../datasets/fraud_dataset_documentation.pdf).",
        "",
        f"- Filas: {profile['rows']}",
        f"- Columnas analizadas: {profile['columns']}",
        f"- Valores faltantes: {sum(profile['missing_values'].values())}",  # type: ignore[union-attr]
        f"- Filas duplicadas: {profile['duplicates']}",
        f"- Probabilidad objetivo: media {data[TEACHER_TARGET].mean():.4f}, mínimo {data[TEACHER_TARGET].min():.4f}, máximo {data[TEACHER_TARGET].max():.4f}",
        "",
        "## Columnas documentadas",
        "",
        "| Columna | Significado / unidad | Uso |",
        "| --- | --- | --- |",
        "| `timestamp` | Instante de la compra, segundos Unix | Descartada |",
        "| `amount_usd` | Monto total de la compra, USD | Entrada |",
        "| `quantity_purchased` | Cantidad comprada de un mismo artículo | Entrada |",
        "| `session_duration_seconds` | Duración de la sesión, segundos | Entrada |",
        "| `days_since_last_purchase` | Días desde la última compra | Entrada |",
        "| `account_age_days` | Antigüedad de la cuenta, días | Entrada |",
        "| `device_screen_resolution` | Cantidad de píxeles de pantalla (ancho × alto) | Descartada |",
        "| `time_since_last_login_s` | Segundos desde el último ingreso | Descartada |",
        "| `items_viewed_before_purchase` | Artículos vistos antes de comprar | Entrada |",
        "| `big_model_fraud_probability` | Probabilidad del modelo de referencia, entre 0 y 1 | Objetivo |",
        "",
        "## Rangos observados",
        "",
        *range_table,
        "",
        "## Decisión de preprocesamiento",
        "",
        f"Se usan como features: {', '.join(FEATURE_COLUMNS)}.",
        f"Se excluyen: {', '.join(EXCLUDED_COLUMNS)}, por su baja correlación lineal observada con la probabilidad objetivo en el análisis inicial.",
        f"`{TEACHER_TARGET}` es el objetivo continuo en [0, 1].",
        "Se elige estandarización Z-score (media 0, desvío 1): centra las entradas y pone sus escalas en un rango comparable para el descenso por gradiente. También puede verse afectada por valores extremos. El scaler se ajusta solo con las filas de entrenamiento de cada fold y se reutiliza en su validación.",
        "Se empleará K-Fold con mezcla reproducible: toda muestra valida una vez y el escalador se ajusta de nuevo en cada fold.",
        "Se conservan histogramas y correlaciones de todas las variables originales junto a vistas de las variables elegidas.",
        "",
        "## Correlación lineal con la probabilidad objetivo",
        "",
        *correlation_table,
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save_distributions(columns_to_plot: list[str], filename: str, title: str) -> None:
        columns = 3
        rows = math.ceil(len(columns_to_plot) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows), squeeze=False)
        for axis, column in zip(axes.flat, columns_to_plot):
            axis.hist(data[column], bins=35, color="#4472c4", edgecolor="white")
            axis.set_title(column)
        for axis in axes.flat[len(columns_to_plot):]:
            axis.remove()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)

    save_distributions(FEATURE_COLUMNS + EXCLUDED_COLUMNS, "all_feature_distributions.png", "All original features")
    save_distributions(FEATURE_COLUMNS, "model_feature_distributions.png", "Features selected for TinyModel")

    def save_correlations(columns_to_plot: list[str], filename: str) -> None:
        correlation = data[columns_to_plot + [TEACHER_TARGET]].corr(numeric_only=True)
        fig, axis = plt.subplots(figsize=(10, 8))
        image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=75, ha="right")
        axis.set_yticks(range(len(correlation.index)), correlation.index)
        fig.colorbar(image, ax=axis, label="Pearson correlation")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)

    save_correlations(FEATURE_COLUMNS + EXCLUDED_COLUMNS, "correlations.png")
    save_correlations(FEATURE_COLUMNS, "model_correlations.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="datasets/fraud_dataset.csv")
    parser.add_argument("--output", default="reports/fraud_eda")
    args = parser.parse_args()
    write_report(load_fraud_dataset(args.csv), Path(args.output))
    print(f"EDA report written to {args.output}")


if __name__ == "__main__":
    main()
