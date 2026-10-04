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

    rows, runs, lr_search, models, histories = [], [], [], {}, {}
    for spec in ROWS:
        if spec["learning_rates"]:
            lr, trials = choose_learning_rate(variant, config, spec["changes"], spec["learning_rates"])
            lr_search.extend({"step": spec["name"], **trial} for trial in trials)
            print(f"{spec['name']}: lr {lr} " + ", ".join(
                f"{t['lr']}: {100 * t['validation_accuracy']:.2f} %" for t in trials))
        else:
            lr = None
        training_config = row_config(config, spec["changes"], lr)

        per_seed = [train_variant(variant, training_config, seed) for seed in config["seeds"]]
        accuracies = [result["best"]["accuracy"] for result in per_seed]
        models[spec["name"]] = per_seed[int(np.argmax(accuracies))]
        histories[spec["name"]] = models[spec["name"]]["history"]
        for seed, result in zip(config["seeds"], per_seed):
            runs.append({"step": spec["name"], "seed": seed, "best_epoch": result["best"]["epoch"],
                         "train_accuracy": result["train_accuracy"],
                         "validation_accuracy": result["best"]["accuracy"],
                         "validation_macro_f1": result["best"]["macro_f1"]})
        rows.append({
            "step": spec["name"],
            "output_activation": training_config["model"]["output_activation"],
            "loss": training_config["loss"],
            "optimizer": training_config["optimizer"],
            "seeds": len(config["seeds"]),
            "validation_accuracy_mean": float(np.mean(accuracies)),
            "validation_accuracy_std": float(np.std(accuracies)),
            "macro_f1_mean": float(np.mean([r["best"]["macro_f1"] for r in per_seed])),
            "train_accuracy_mean": float(np.mean([r["train_accuracy"] for r in per_seed])),
            "median_best_epoch": float(np.median([r["best"]["epoch"] for r in per_seed])),
        })
        print(f"{spec['name']:<40} {100 * np.mean(accuracies):6.2f} % ± {100 * np.std(accuracies):.2f}")

    winner = max(rows, key=lambda row: row["validation_accuracy_mean"])
    return {"config": config, "rows": rows, "runs": runs, "lr_search": lr_search,
            "winner": winner["step"], "validation_samples": int(len(validation_rows)),
            "histories": histories, "_models": models}


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


def reference_note(reference: dict[str, Any], summary_path: Path = DATA_ABLATION_SUMMARY) -> list[str]:
    """A paragraph explaining why the reference row differs from the data ablation's winner, if it does."""
    if not summary_path.exists():
        return []
    previous = json.loads(summary_path.read_text(encoding="utf-8"))["rows"][-1]
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
    lines = [
        "# Ejercicio 3: qué aporta cambiar cómo aprende la red", "",
        f"Mismos datos en todas las filas (ambos archivos, balanceados y con imágenes "
        f"desplazadas y rotadas) y **el mismo conjunto de validación** "
        f"({result['validation_samples']} imágenes). Cada fila agrega un cambio a la anterior y "
        f"se entrena con {len(config['seeds'])} semillas y {config['training']['epochs']} épocas.", "",
        f"Las filas que cambian la pérdida o el optimizador eligen primero su tasa de aprendizaje "
        f"con una corrida corta ({config['lr_search_epochs']} épocas, semilla {config['seeds'][0]}) "
        f"sobre la misma validación.", "",
        "| Paso | Salida | Pérdida | Optimizador | η | Accuracy de validación | F1 macro |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['output_activation']} | {row['loss']} | {row['optimizer']['name']} | "
            f"{row['optimizer']['lr']} | {_percent(row['validation_accuracy_mean'])} ± "
            f"{_number(100 * row['validation_accuracy_std'])} | {_number(row['macro_f1_mean'], 4)} |")
    lines += ["", *reference_note(rows[0])]
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
