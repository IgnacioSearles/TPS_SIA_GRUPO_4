"""Ejercicio 3: one final measurement on `digits_test.csv` for the selected architecture.

The architecture was chosen on validation by `architecture_e3`. This script only
reads the saved weights of every seed of that architecture and scores them on the
test file — nothing here is used to choose or tune anything. Reporting all seeds,
not only the best one, shows how much of the result depends on the initialization.

Run from TP3:
    python -m experiments.digits.final_test_e3 [config.json]

The config may instead name one row of a `training_ablation` study:
    {"training_ablation_study": "reports/digits_e3_softmax_tuned",
     "step": "+ softmax y entropía cruzada", "output": {"directory": "..."}}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nn  # noqa: F401 - register model components
from datasets.digit_dataset_loader import load_digit_arrays
from experiments.digits.baseline import digit_metrics
from nn.network import build_model, load_weights

DEFAULTS = {
    "architecture_study": "reports/digits_e3_final/entrenamiento",
    "test_file": "datasets/digits_test.csv",
    "output": {"directory": "reports/digits_e3_final"},
}


def _percent(value, decimals=2):
    return f"{100 * value:.{decimals}f} %".replace(".", ",")


def _saved_runs(config):
    """Model config and (seed, validation accuracy, weights path) of every saved seed.

    Reads either an `architecture_e3` study (`architecture_study`) or one row of a
    `training_ablation` study (`training_ablation_study` + `step`).
    """
    if "training_ablation_study" in config:
        study_dir = Path(config["training_ablation_study"])
        runs = pd.read_csv(study_dir / "runs.csv")
        runs = runs[runs.step == config["step"]]
        if runs.empty:
            raise ValueError(f"No runs for step {config['step']!r} in {study_dir}")
        seeds = []
        for row in runs.itertuples():
            run_dir = next(path.parent for path in study_dir.glob(f"runs/row_*/seed_{row.seed}/protocol.json")
                           if json.loads(path.read_text())["step"] == config["step"])
            model = json.loads((run_dir / "protocol.json").read_text())["training_config"]["model"]
            seeds.append((int(row.seed), row.validation_accuracy, run_dir / "best_weights.npz"))
        return model, seeds, study_dir
    study_dir = Path(config["architecture_study"])
    study = json.loads((study_dir / "study_config.json").read_text())
    runs = pd.read_csv(study_dir / "runs.csv")
    if runs.architecture.nunique() != 1:
        raise ValueError("Expected a study with exactly one architecture")
    architecture = [int(n) for n in runs.architecture.iloc[0].split("-")]
    model = {**study["model"], "layers": architecture}
    return model, [(int(row.seed), row.validation_accuracy, study_dir / row.run_directory / "best_weights.npz")
                   for row in runs.itertuples()], study_dir


def run(config):
    config = {**DEFAULTS, **config}
    model, seeds, study_dir = _saved_runs(config)
    architecture = model["layers"]
    X_test, y_test = load_digit_arrays(config["test_file"])

    per_seed, outputs = [], {}
    for seed, validation_accuracy, weights in seeds:
        net = build_model(model, np.random.default_rng(0))
        load_weights(net, weights)
        scores = net.forward(X_test)
        metrics = digit_metrics(y_test, scores)
        outputs[seed] = (scores, metrics)
        per_seed.append({"seed": seed, "validation_accuracy": validation_accuracy,
                         "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"],
                         "test_errors": int(round((1 - metrics["accuracy"]) * len(y_test))),
                         **{f"recall_{d}": metrics["per_class"][d]["recall"] for d in range(10)}})
    table = pd.DataFrame(per_seed)

    # The seed shown in detail is the one validation would pick, never the best on test.
    shown_seed = int(table.sort_values("validation_accuracy", ascending=False).seed.iloc[0])
    scores, metrics = outputs[shown_seed]
    predicted = scores.argmax(axis=1)

    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "test_por_semilla.csv", index=False)
    pd.DataFrame(metrics["per_class"]).to_csv(output / "test_por_digito.csv", index=False)
    matrix = np.array(metrics["confusion_matrix"])
    off_diagonal = [(matrix[a, p], a, p) for a in range(10) for p in range(10) if a != p and matrix[a, p]]
    top_confusions = sorted(off_diagonal, reverse=True)[:6]

    summary = {
        "architecture": "-".join(map(str, architecture)),
        "output_activation": model["output_activation"],
        "test_samples": int(len(y_test)),
        "test_accuracy_mean": float(table.test_accuracy.mean()),
        "test_accuracy_std": float(table.test_accuracy.std()),
        "test_accuracy_min": float(table.test_accuracy.min()),
        "test_accuracy_max": float(table.test_accuracy.max()),
        "test_macro_f1_mean": float(table.test_macro_f1.mean()),
        "shown_seed": shown_seed, "shown_seed_rule": "highest validation accuracy",
        "shown_seed_test_accuracy": metrics["accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "top_confusions": [{"real": int(a), "predicho": int(p), "cantidad": int(n)} for n, a, p in top_confusions],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _plot_confusion(matrix, shown_seed, output)
    _plot_errors(X_test, y_test, predicted, scores, output)
    _write_report(summary, table, metrics, study_dir, output, config.get("note"))
    print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "confusion_matrix"}, indent=2, ensure_ascii=False))
    return summary


def _plot_confusion(matrix, seed, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(errors, cmap="Reds")
    for actual in range(10):
        for pred in range(10):
            value = matrix[actual, pred]
            if value:
                ax.text(pred, actual, str(value), ha="center", va="center", fontsize=8,
                        color="white" if actual != pred and errors[actual, pred] > errors.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Dígito predicho", ylabel="Dígito real",
           title=f"Test — semilla {seed} (color = solo errores)")
    fig.colorbar(im, ax=ax, label="Errores")
    fig.tight_layout()
    fig.savefig(output / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)


def _plot_errors(X, y, predicted, scores, output, count=24):
    """The misclassified test images the network was most sure about."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wrong = np.flatnonzero(predicted != y)
    confidence = scores[wrong, predicted[wrong]]
    chosen = wrong[np.argsort(-confidence)][:count]
    cols = 8
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.7))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, index in zip(np.ravel(axes), chosen):
        ax.imshow(X[index].reshape(28, 28), cmap="gray_r")
        ax.set_title(f"real {y[index]} → {predicted[index]}", fontsize=8)
    fig.suptitle("Errores de test con mayor confianza de la red", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "test_errores.png", dpi=160)
    plt.close(fig)


def _write_report(summary, table, metrics, study_dir, output, note=None):
    lines = [
        "# Ejercicio 3: evaluación final en test", "",
        f"Arquitectura {summary['architecture']} (salida {summary['output_activation']}) elegida por "
        f"validación en `{study_dir}`, con balanceo y augmentación. Se evalúan las {len(table)} semillas sobre las "
        f"{summary['test_samples']} imágenes de `digits_test.csv`. El test no se usó para elegir nada.", "",
        *([note, ""] if note else []),
        "| Semilla | Accuracy validación | Accuracy test | F1 macro test | Errores |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table.itertuples():
        lines.append(f"| {row.seed} | {_percent(row.validation_accuracy)} | {_percent(row.test_accuracy)} | "
                     f"{row.test_macro_f1:.4f} | {row.test_errors} |".replace("0.", "0,"))
    std_points = f"{100 * summary['test_accuracy_std']:.2f}".replace(".", ",")
    lines += [
        "", f"**Media: {_percent(summary['test_accuracy_mean'])} ± {std_points} pp** "
        f"(mín. {_percent(summary['test_accuracy_min'])}, "
        f"máx. {_percent(summary['test_accuracy_max'])}).", "",
        f"## Por dígito (semilla {summary['shown_seed']}, la mejor en validación)", "",
        "| Dígito | Soporte | Recall | Precisión |", "| ---: | ---: | ---: | ---: |",
    ]
    for entry in metrics["per_class"]:
        lines.append(f"| {entry['label']} | {entry['support']} | {_percent(entry['recall'])} | "
                     f"{_percent(entry['precision'])} |")
    lines += ["", "## Confusiones más frecuentes", ""]
    lines += [f"- {c['real']} leído como {c['predicho']}: {c['cantidad']}" for c in summary["top_confusions"]]
    lines += ["", "Figuras: `test_confusion_matrix.png`, `test_errores.png`.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()) if args.config else {})
