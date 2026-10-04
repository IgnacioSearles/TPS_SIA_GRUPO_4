"""Ejercicio 3, second ablation: same data, change how the network learns.

`ablation.py` held the network fixed and changed the data. This study does the
opposite: every row trains on the winning data of that table (both files merged,
balanced and augmented) and changes one thing about the training:

1. Reference: 10 sigmoid outputs + MSE + momentum, exactly the previous winner.
2. + softmax output and cross-entropy loss (one combined change: cross-entropy
   is only meaningful on outputs that form a probability distribution).
3. + Adam instead of momentum.

A new loss or optimizer changes the size of the gradient steps, so reusing the
old learning rate would compare a tuned row against an untuned one. Rows that
change the loss or optimizer therefore pick their learning rate first, with a
short run on the first seed, scored on the same validation set. Then every row
trains with all seeds at full length.

`digits_test.csv` is never opened unless `--final-test` is passed. Note that the
test set was already consulted once, by `ablation.py --final-test`; a second
look should be reported as such.

Run from TP3:
    python -m experiments.digits.training_ablation [config.json]
    python -m experiments.digits.training_ablation --search-only 1 --learning-rates 0.001 0.003

`--evaluate-test` scores the saved winner (`best_weights.npz`) on digits_test.csv
without retraining, and refuses to do it a second time.
`--report-only` rewrites the report from the saved summary.json, without training.
`--search-only ROW` repeats just that row's learning-rate search (on the given
rates, or the row's grid) and writes `lr_search_row<ROW>.csv`; the seeded short
runs are deterministic, so its numbers line up with the full study's search.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.config import deep_merge, read_json
from experiments.digits.ablation import (
    _number,
    _percent,
    build_variants,
    final_test,
    merge_config,
    resolve_steps,
    train_variant,
)
from datasets.digit_dataset_loader import load_digit_arrays
from experiments.digits.baseline import digit_metrics
from experiments.digits.data import merge_digit_files, split_train_validation
from nn.network import build_model, load_weights, save_weights

SOFTMAX_CROSS_ENTROPY = {"model": {"output_activation": "softmax"}, "loss": "cross_entropy"}

# Each row lists its full difference from the reference, so rows are cumulative.
# `learning_rates` is the grid searched for that row; None keeps the reference's.
ROWS: list[dict[str, Any]] = [
    {"name": "Referencia: sigmoide + MSE + momentum", "changes": {}, "learning_rates": None},
    {"name": "+ softmax y entropía cruzada", "changes": SOFTMAX_CROSS_ENTROPY,
     "learning_rates": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]},
    {"name": "+ Adam", "changes": deep_merge(SOFTMAX_CROSS_ENTROPY, {"optimizer": {"name": "adam"}}),
     "learning_rates": [0.0003, 0.001, 0.003, 0.01]},
]

# Where the data ablation saved its table; its last row is this study's reference.
DATA_ABLATION_SUMMARY = Path("reports/digits_ablation/summary.json")

DEFAULTS: dict[str, Any] = {
    "lr_search_epochs": 40,
    "output": {"directory": "reports/digits_training_ablation"},
}


def row_config(config: dict[str, Any], changes: dict[str, Any], lr: float | None = None) -> dict[str, Any]:
    """The full training config of one row, optionally at a given learning rate.

    A row that names an optimizer replaces the reference's outright, so momentum's
    `momentum` argument never leaks into Adam.
    """
    row = deep_merge(config, changes)
    # deep_merge shares nested dicts it did not override; copy before editing the lr.
    row["optimizer"] = dict(changes.get("optimizer", config["optimizer"]))
    row["optimizer"].setdefault("lr", config["optimizer"]["lr"])
    if lr is not None:
        row["optimizer"]["lr"] = lr
    return row


def choose_learning_rate(variant, config: dict[str, Any], changes: dict[str, Any],
                         learning_rates: list[float]) -> tuple[float, list[dict[str, Any]]]:
    """Short run per candidate on the first seed; the best validation accuracy wins."""
    search = deep_merge(config, {"training": {"epochs": config["lr_search_epochs"]}})
    trials = []
    for lr in learning_rates:
        result = train_variant(variant, row_config(search, changes, lr), config["seeds"][0])
        trials.append({"lr": lr, "validation_accuracy": result["best"]["accuracy"],
                       "best_epoch": result["best"]["epoch"]})
    best = max(trials, key=lambda trial: trial["validation_accuracy"])
    return best["lr"], trials


def top_label_calibration(labels: np.ndarray, outputs: np.ndarray, n_bins: int = 10):
    """Top-score reliability diagnostics; sigmoid scores are a proxy, not probabilities."""
    confidence = outputs.max(axis=1)
    predicted = outputs.argmax(axis=1)
    correct = (predicted == labels).astype(float)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = index / n_bins, (index + 1) / n_bins
        mask = ((confidence >= lower) & (confidence < upper)
                if index < n_bins - 1 else (confidence >= lower) & (confidence <= upper))
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += count / len(labels) * abs(mean_confidence - accuracy)
        else:
            mean_confidence = accuracy = None
        bins.append({"bin": index, "lower": lower, "upper": upper, "count": count,
                     "mean_confidence": mean_confidence, "accuracy": accuracy})
    errors = ~correct.astype(bool)
    return {
        "top_label_ece": float(ece),
        "mean_top_score_on_errors": float(confidence[errors].mean()) if errors.any() else None,
        "error_fraction_with_top_score_ge_0_99": float(np.mean(confidence[errors] >= 0.99)) if errors.any() else None,
        "bins": bins,
    }


def winning_variant(config: dict[str, Any]):
    """Training data of the data ablation's last row: both files, balanced, augmented."""
    data = merge_digit_files(config["files"])
    train_rows, validation_rows = split_train_validation(
        data.y, config["validation_ratio"], config["split_seed"])
    # The last data step: both files, balanced, augmented.
    winning_step = resolve_steps(config["files"], config["baseline_files"])[-1]
    return build_variants(data, train_rows, validation_rows, [winning_step],
                          np.random.default_rng(config["split_seed"]))[0]


def search_only(config: dict[str, Any], row_index: int, learning_rates: list[float] | None) -> Path:
    """Rerun one row's learning-rate search, leaving the full study's files alone."""
    config = merge_config(deep_merge(DEFAULTS, config))
    spec = ROWS[row_index]
    learning_rates = learning_rates or spec["learning_rates"]
    if not learning_rates:
        raise ValueError(f"Row {row_index} ({spec['name']}) has no learning-rate search")
    _, trials = choose_learning_rate(winning_variant(config), config, spec["changes"], learning_rates)
    for trial in trials:
        print(f"{spec['name']}  lr {trial['lr']}: {100 * trial['validation_accuracy']:.2f} % "
              f"(best epoch {trial['best_epoch']})")
    import pandas as pd
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"lr_search_row{row_index}.csv"
    pd.DataFrame([{"step": spec["name"], **trial} for trial in trials]).to_csv(path, index=False)
    return path


def run_study(config: dict[str, Any]) -> dict[str, Any]:
    config = merge_config(deep_merge(DEFAULTS, config))
    variant = winning_variant(config)
    validation_rows = variant.validation_rows
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    selected_indices = config.get("study", {}).get("rows", list(range(len(ROWS))))
    specs = [(index, ROWS[index]) for index in selected_indices]

    rows, runs, lr_search, models, histories, calibration_bins = [], [], [], {}, {}, []
    for row_index, spec in specs:
        fixed_lr = config.get("study", {}).get("fixed_learning_rates", {}).get(spec["name"])
        if spec["learning_rates"] and fixed_lr is None:
            grid = config.get("study", {}).get("learning_rate_grids", {}).get(
                spec["name"], spec["learning_rates"])
            lr, trials = choose_learning_rate(variant, config, spec["changes"], grid)
            lr_search.extend({"step": spec["name"], **trial} for trial in trials)
            print(f"{spec['name']}: lr {lr} " + ", ".join(
                f"{t['lr']}: {100 * t['validation_accuracy']:.2f} %" for t in trials))
        elif spec["learning_rates"]:
            lr = float(fixed_lr)
            print(f"{spec['name']}: lr {lr} (reutilizada del estudio previo)", flush=True)
        else:
            lr = None
        training_config = row_config(config, spec["changes"], lr)

        per_seed = []
        for seed in config["seeds"]:
            run_dir = output / "runs" / f"row_{row_index:02d}" / f"seed_{seed}"
            protocol = {"row_index": row_index, "step": spec["name"], "seed": seed,
                        "training_config": training_config, "validation_samples": len(validation_rows)}
            protocol_path, summary_path = run_dir / "protocol.json", run_dir / "summary.json"
            weights_path, history_path = run_dir / "best_weights.npz", run_dir / "history.csv"
            if all(path.exists() for path in (protocol_path, summary_path, weights_path, history_path)):
                if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
                    raise ValueError(f"Saved run protocol differs in {run_dir}; use a new output directory")
                saved = json.loads(summary_path.read_text(encoding="utf-8"))
                net = build_model(training_config["model"], np.random.default_rng(seed))
                load_weights(net, weights_path)
                per_seed.append({"net": net, "best": saved["best"], "epochs_ran": saved["epochs_ran"],
                                 "train_accuracy": saved["train_accuracy"],
                                 "history": pd.read_csv(history_path).to_dict("records")})
                print(f"{spec['name']} seed={seed}: reutiliza corrida guardada", flush=True)
            else:
                result = train_variant(variant, training_config, seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                save_weights(result["net"], weights_path)
                pd.DataFrame(result["history"]).to_csv(history_path, index=False)
                protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
                summary_path.write_text(json.dumps({key: result[key] for key in
                    ("best", "epochs_ran", "train_accuracy")}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
                per_seed.append(result)
        accuracies = [result["best"]["accuracy"] for result in per_seed]
        models[spec["name"]] = per_seed[int(np.argmax(accuracies))]
        histories[spec["name"]] = models[spec["name"]]["history"]
        calibration = []
        for seed, result in zip(config["seeds"], per_seed):
            diagnostics = top_label_calibration(variant.y_validation,
                                                result["net"].forward(variant.X_validation))
            calibration.append(diagnostics)
            calibration_bins.extend({"step": spec["name"], "seed": seed, **entry}
                                    for entry in diagnostics["bins"])
            runs.append({"step": spec["name"], "seed": seed, "best_epoch": result["best"]["epoch"],
                         "train_accuracy": result["train_accuracy"],
                         "validation_accuracy": result["best"]["accuracy"],
                         "validation_macro_f1": result["best"]["macro_f1"],
                         "top_label_ece": diagnostics["top_label_ece"],
                         "mean_top_score_on_errors": diagnostics["mean_top_score_on_errors"],
                         "error_fraction_top_score_ge_0_99": diagnostics["error_fraction_with_top_score_ge_0_99"]})
        rows.append({
            "step": spec["name"],
            "output_activation": training_config["model"]["output_activation"],
            "loss": training_config["loss"],
            "optimizer": training_config["optimizer"],
            "seeds": len(config["seeds"]),
            "validation_accuracy_mean": float(np.mean(accuracies)),
            "validation_accuracy_std": float(np.std(accuracies)),
            "macro_f1_mean": float(np.mean([r["best"]["macro_f1"] for r in per_seed])),
            "top_label_ece_mean": float(np.mean([item["top_label_ece"] for item in calibration])),
            "top_label_ece_std": float(np.std([item["top_label_ece"] for item in calibration])),
            "mean_top_score_on_errors": float(np.mean([item["mean_top_score_on_errors"] for item in calibration
                                                       if item["mean_top_score_on_errors"] is not None])),
            "error_fraction_top_score_ge_0_99": float(np.mean([
                item["error_fraction_with_top_score_ge_0_99"] for item in calibration
                if item["error_fraction_with_top_score_ge_0_99"] is not None])),
            "train_accuracy_mean": float(np.mean([r["train_accuracy"] for r in per_seed])),
            "median_best_epoch": float(np.median([r["best"]["epoch"] for r in per_seed])),
        })
        print(f"{spec['name']:<40} {100 * np.mean(accuracies):6.2f} % ± {100 * np.std(accuracies):.2f}")

    winner = max(rows, key=lambda row: row["validation_accuracy_mean"])
    return {"config": config, "rows": rows, "runs": runs, "lr_search": lr_search,
            "winner": winner["step"], "validation_samples": int(len(validation_rows)),
            "histories": histories, "calibration_bins": calibration_bins, "_models": models}


def merge_lr_searches(lr_search: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    """The study's search plus any `--search-only` reruns saved next to it, one entry per (row, lr).

    The short runs are seeded, so a rate tried in both places gives the same number.
    """
    import pandas as pd

    trials = list(lr_search)
    for path in sorted(output.glob("lr_search_row*.csv")):
        trials += pd.read_csv(path).to_dict("records")
    unique = {(trial["step"], float(trial["lr"])): trial for trial in reversed(trials)}
    order = {spec["name"]: index for index, spec in enumerate(ROWS)}
    return sorted(unique.values(), key=lambda trial: (order.get(trial["step"], len(order)), trial["lr"]))


def reference_note(reference: dict[str, Any], config: dict[str, Any],
                   summary_path: Path = DATA_ABLATION_SUMMARY) -> list[str]:
    """A paragraph explaining why the reference row differs from the data ablation's winner, if it does."""
    if not summary_path.exists():
        return []
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    if saved.get("config", {}).get("model", {}).get("layers") != config["model"]["layers"]:
        return []
    previous = saved["rows"][-1]
    gap = 100 * abs(reference["validation_accuracy_mean"] - previous["validation_accuracy_mean"])
    if gap < 0.005:
        return []
    return [
        f"La referencia da {_percent(reference['validation_accuracy_mean'])} y no el "
        f"{_percent(previous['validation_accuracy_mean'])} de `reports/digits_ablation`: la "
        f"configuración es la misma, pero el balanceo de clases duplica imágenes al azar y aquí "
        f"se sortean distinto (el script anterior balanceaba dos filas seguidas con el mismo "
        f"generador). La diferencia, {_number(gap)} puntos, es menor que el desvío entre semillas "
        f"y las {len(ROWS)} filas de esta tabla comparten el mismo conjunto de entrenamiento.", "",
    ]


def evaluate_saved_winner(output: Path) -> dict[str, Any]:
    """Score the saved winner on the test set, once.

    `best_weights.npz` holds the winning row's best seed, chosen on validation. A
    result already in summary.json is returned as is, so the test cannot be
    consulted again by accident.
    """
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if "test" in saved:
        return saved["test"]
    config = saved["config"]
    spec = next(spec for spec in ROWS if spec["name"] == saved["winner"])
    row = next(row for row in saved["rows"] if row["step"] == saved["winner"])
    model_config = row_config(config, spec["changes"])["model"]
    net = load_weights(build_model(model_config, np.random.default_rng(0)), output / "best_weights.npz")

    X_test, y_test = load_digit_arrays(config["test_file"])
    metrics = digit_metrics(y_test, net.forward(X_test))
    test = {"step": saved["winner"], "optimizer": row["optimizer"], "samples": int(len(y_test)),
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "per_class": metrics["per_class"], "confusion_matrix": metrics["confusion_matrix"]}
    write_report(saved, output, test)
    return test


def _plot_calibration(calibration_bins: list[dict[str, Any]], output: Path) -> None:
    """Plot pooled validation reliability curves, with each seed contributing equally by rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(calibration_bins)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="calibración ideal")
    for step, group in frame.groupby("step", sort=False):
        present = group[group["count"] > 0].copy()
        present["confidence_weighted"] = present["mean_confidence"] * present["count"]
        present["accuracy_weighted"] = present["accuracy"] * present["count"]
        pooled = present.groupby("bin").agg(count=("count", "sum"),
                                             confidence=("confidence_weighted", "sum"),
                                             accuracy=("accuracy_weighted", "sum"))
        ax.plot(pooled.confidence / pooled["count"], pooled.accuracy / pooled["count"],
                marker="o", label=step)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Score máximo promedio",
           ylabel="Frecuencia de acierto", title="Confiabilidad en validación")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "validation_reliability.png", dpi=160)
    plt.close(fig)


def write_report(result: dict[str, Any], output: str | Path, test: dict[str, Any] | None = None) -> Path:
    """Write tables, summary and report. Without `histories` (a rewrite from summary.json),
    the existing validation-curve figure is kept."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    result = {**result, "lr_search": merge_lr_searches(result["lr_search"], output)}
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(result["runs"]).to_csv(output / "runs.csv", index=False)
    pd.DataFrame(result["lr_search"]).to_csv(output / "lr_search.csv", index=False)
    if result.get("calibration_bins"):
        pd.DataFrame(result["calibration_bins"]).to_csv(output / "calibration_bins.csv", index=False)
        _plot_calibration(result["calibration_bins"], output)
    serializable = {key: value for key, value in result.items() if key not in ("_models", "histories")}
    if test is not None:
        serializable["test"] = test
    (output / "summary.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    if result.get("histories"):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for name, history in result["histories"].items():
            ax.plot([record["epoch"] for record in history],
                    [100 * record["validation_accuracy"] for record in history], label=name)
        ax.set(xlabel="Época", ylabel="Accuracy de validación (%)", ylim=(90, 100),
               title="Curvas de validación de la mejor semilla de cada fila")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / "validation_curves.png", dpi=150)
        plt.close(fig)

    config = result["config"]
    study_options = config.get("study", {})
    fixed_rates = study_options.get("fixed_learning_rates", {})
    if fixed_rates:
        rate_note = "Las tasas de las variantes softmax/Adam se reutilizan del estudio previo " \
                    "con arquitectura [784,64,32,10]; no se vuelven a optimizar aquí."
    else:
        rate_note = (f"Las variantes que cambian la pérdida o el optimizador seleccionan su tasa "
                     f"con una corrida corta ({config['lr_search_epochs']} épocas) sobre validación.")
    lines = [
        "# Ejercicio 3: qué aporta cambiar cómo aprende la red", "",
        f"Mismos datos en todas las filas (ambos archivos, balanceados y con imágenes "
        f"desplazadas y rotadas) y **el mismo conjunto de validación** "
        f"({result['validation_samples']} imágenes). Cada fila agrega un cambio a la anterior y "
        f"se entrena con {len(config['seeds'])} semillas y {config['training']['epochs']} épocas.", "",
        rate_note, "",
        "| Paso | Salida | Pérdida | Optimizador | η | Accuracy validación | F1 macro | ECE top-score* | Score medio en errores |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['output_activation']} | {row['loss']} | {row['optimizer']['name']} | "
            f"{row['optimizer']['lr']} | {_percent(row['validation_accuracy_mean'])} ± "
            f"{_number(100 * row['validation_accuracy_std'])} | {_number(row['macro_f1_mean'], 4)} | "
            f"{_number(row.get('top_label_ece_mean'), 4)} ± {_number(row.get('top_label_ece_std'), 4)} | "
            f"{_number(row.get('mean_top_score_on_errors'), 4)} |")
    lines += ["", "*ECE compara el score máximo con la frecuencia de acierto en 10 bins. Para softmax es "
              "una medida de calibración top-label; para salidas sigmoides independientes, el score máximo "
              "es solo un proxy diagnóstico, no una probabilidad categórica.", "",
              "`mean_top_score_on_errors` resume cuán alto es el score asignado a la clase predicha cuando "
              "el modelo se equivoca. Ninguna de estas métricas se calcula sobre el test.", "",
              *reference_note(rows[0], result["config"])]
    lines += ["## Búsqueda de tasa de aprendizaje", "", "| Paso | η | Accuracy de validación |",
              "| --- | ---: | ---: |"]
    lines += [f"| {t['step']} | {t['lr']} | {_percent(t['validation_accuracy'])} |" for t in result["lr_search"]]

    if test is not None:
        lines += [
            "", "## Evaluación sobre el test", "",
            f"Modelo de «{test['step']}» (la mejor semilla según validación) sobre "
            f"{test['samples']} imágenes de `digits_test.csv`. Es la **segunda** consulta del test "
            f"en el ejercicio 3: la primera fue la de `ablation.py`. Todas las decisiones de esta "
            f"tabla se tomaron sobre validación.", "",
            f"- **Accuracy de test: {_percent(test['accuracy'])}**",
            f"- F1 macro: {_number(test['macro_f1'], 4)}",
        ]
        if DATA_ABLATION_SUMMARY.exists():
            first = json.loads(DATA_ABLATION_SUMMARY.read_text(encoding="utf-8")).get("test")
            if first:
                lines.append(f"- Primera consulta («{first['step']}»): {_percent(first['accuracy'])}")
        lines += ["", "| Dígito | Soporte | Recall | F1 |", "| ---: | ---: | ---: | ---: |"]
        lines += [f"| {entry['label']} | {entry['support']} | {_percent(entry['recall'])} | "
                  f"{_number(entry['f1'], 4)} |" for entry in test["per_class"]]
    lines += ["", "## Reproducir", "", "```bash", "python -m experiments.digits.training_ablation", "```", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="JSON config; omitted keys use the defaults")
    parser.add_argument("--final-test", action="store_true",
                        help="score the winner on digits_test.csv (its second use in ejercicio 3)")
    parser.add_argument("--output")
    parser.add_argument("--search-only", type=int, metavar="ROW",
                        help="only rerun the learning-rate search of ROWS[ROW]")
    parser.add_argument("--learning-rates", type=float, nargs="+",
                        help="with --search-only: the rates to try instead of the row's grid")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="score the saved winner on digits_test.csv, once, without training")
    parser.add_argument("--report-only", action="store_true",
                        help="rewrite the report from the saved summary.json, without training")
    args = parser.parse_args()

    if args.evaluate_test:
        output = Path(args.output or merge_config(deep_merge(DEFAULTS, read_json(args.config) if args.config else {}))["output"]["directory"])
        test = evaluate_saved_winner(output)
        print(f"Test ({test['step']}): {100 * test['accuracy']:.2f} %  F1 macro {test['macro_f1']:.4f}")
        return

    if args.report_only:
        output = Path(args.output or merge_config(deep_merge(DEFAULTS, read_json(args.config) if args.config else {}))["output"]["directory"])
        saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        print(f"Report: {write_report(saved, output, saved.get('test'))}")
        return

    if args.search_only is not None:
        print(f"Search: {search_only(read_json(args.config) if args.config else {}, args.search_only, args.learning_rates)}")
        return

    result = run_study(read_json(args.config) if args.config else {})
    test = final_test(result) if args.final_test else None
    if test is not None:
        print(f"\nTest ({test['step']}): {100 * test['accuracy']:.2f} %")

    output = args.output or result["config"]["output"]["directory"]
    write_report(result, output, test)
    save_weights(result["_models"][result["winner"]]["net"], Path(output) / "best_weights.npz")
    print(f"\nReport: {output}")


if __name__ == "__main__":
    main()
