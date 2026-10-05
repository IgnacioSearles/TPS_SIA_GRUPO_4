"""Ejercicio 3: add one change at a time and measure what each one is worth.

Answers question (b) of the assignment — which techniques improved the result —
by changing exactly one thing per row and scoring every row on **the same
validation set**. Question (c), what changed besides our own techniques, is
answered by the first two rows: the jump from `digits.csv` to the merged files is
the arrival of a class that was previously absent, not a better model.

The protocol, in order, and why it is that order:

1. Merge both training files, keeping each distinct image once.
2. Split off validation from that merge, stratified by digit. **One** validation
   set serves every row, so the rows are comparable.
3. Each row trains only on the rows credited to the files it is allowed to use,
   optionally rebalanced and optionally augmented — training side only.
4. Augmentation is redrawn every epoch, so the model rarely sees the same pixels
   twice.

`digits_test.csv` is never opened here. Run `final_test` once, at the end, on the
winning row.

Run from TP3:
    python -m experiments.digits.ablation [config.json]
    python -m experiments.digits.ablation [config.json] --final-test
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import nn  # noqa: F401  (registers activations, losses, optimizers, initializers)
import training  # noqa: F401  (registers callbacks)
from data.augmentation import augment
from datasets.digit_dataset_loader import load_digit_arrays
from experiments.config import deep_merge, read_json
from experiments.digits.baseline import digit_metrics
from experiments.digits.data import (
    DEFAULT_TRAINING_FILES,
    TEST_FILE,
    DigitData,
    balanced_indices,
    merge_digit_files,
    one_hot,
    split_train_validation,
)
from nn.network import build_model, load_weights, save_weights
from nn.registry import build
from training.trainer import train

BASE_FILE, EXTRA_FILE = DEFAULT_TRAINING_FILES

# One change per row. `files` is which training files that row may learn from.
STEPS: list[dict[str, Any]] = [
    {"name": "Ejercicio 2 (solo digits.csv)", "files": [BASE_FILE], "balance": False, "augment": False},
    {"name": "+ more_digits.csv (sin repetidos)", "files": [BASE_FILE, EXTRA_FILE], "balance": False, "augment": False},
    {"name": "+ balanceo de clases", "files": [BASE_FILE, EXTRA_FILE], "balance": True, "augment": False},
    {"name": "+ imágenes desplazadas y rotadas", "files": [BASE_FILE, EXTRA_FILE], "balance": True, "augment": True},
]

DEFAULTS: dict[str, Any] = {
    "files": list(DEFAULT_TRAINING_FILES),
    "baseline_files": [BASE_FILE],
    "test_file": TEST_FILE,
    "validation_ratio": 0.2,
    "split_seed": 42,
    "seeds": [42, 7, 21],
    # Ejercicio 2's winning configuration, held fixed so only the data changes.
    "model": {"layers": [784, 64, 32, 10], "activation": "tanh",
              "output_activation": "sigmoid", "initializer": "xavier"},
    "loss": "mse",
    "optimizer": {"name": "momentum", "lr": 0.1, "momentum": 0.9},
    "training": {"epochs": 150, "batch_size": 128},
    "augmentation": {"max_rotation": 10.0, "max_scale": 0.1, "max_shift": 2.0, "sigma": 0.0},
    "output": {"directory": "reports/digits_ablation"},
}


def resolve_steps(files: Sequence[str], baseline_files: Sequence[str]) -> list[dict[str, Any]]:
    """`STEPS` with real paths filled in: the first row sees only the baseline files."""
    steps = [dict(step) for step in STEPS]
    steps[0]["files"] = list(baseline_files)
    for step in steps[1:]:
        step["files"] = list(files)
    return steps


def merge_config(config: dict[str, Any]) -> dict[str, Any]:
    """Layer `config` over the defaults, replacing the optimizer instead of merging it.

    Nested dicts merge key by key, which would leave the default `momentum` in place
    when a caller asks for plain SGD and hand it to a constructor that has no such
    argument. An optimizer named explicitly replaces the default outright.
    """
    merged = deep_merge(DEFAULTS, config)
    if isinstance(config.get("optimizer"), dict) and "name" in config["optimizer"]:
        merged["optimizer"] = dict(config["optimizer"])
    return merged


@dataclass
class Variant:
    """One row of the table: its training data, and the validation set every row shares."""

    name: str
    files: list[str]
    balance: bool
    augment: bool
    X_train: np.ndarray
    y_train: np.ndarray
    validation_rows: np.ndarray
    X_validation: np.ndarray = field(repr=False, default=None)
    y_validation: np.ndarray = field(repr=False, default=None)


def training_rows_for(data: DigitData, train_rows: np.ndarray, files: Sequence[str]) -> np.ndarray:
    """The training rows whose image was introduced by one of `files`.

    An image present in both files is credited to the earlier one, which is where
    it genuinely appears, so restricting to `digits.csv` keeps everything that file
    contains and nothing it does not.
    """
    allowed = np.isin(data.sources[train_rows], list(files))
    if not allowed.any():
        raise ValueError(f"No training rows came from {list(files)}")
    return train_rows[allowed]


def build_variants(
    data: DigitData,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    steps: Sequence[dict[str, Any]],
    rng: np.random.Generator,
) -> list[Variant]:
    """Prepare each row's training data. Validation is untouched and identical for all."""
    variants = []
    balanced_rows_by_files: dict[tuple[str, ...], np.ndarray] = {}
    for step in steps:
        rows = training_rows_for(data, train_rows, step["files"])
        X_train, y_train = data.X[rows], data.y[rows]
        if step["balance"]:
            key = tuple(step["files"])
            if key not in balanced_rows_by_files:
                balanced_rows_by_files[key] = balanced_indices(y_train, rng)
            indices = balanced_rows_by_files[key]
            X_train, y_train = X_train[indices], y_train[indices]
        variants.append(Variant(
            name=step["name"], files=list(step["files"]),
            balance=step["balance"], augment=step["augment"],
            X_train=X_train, y_train=y_train,
            validation_rows=validation_rows,
            X_validation=data.X[validation_rows], y_validation=data.y[validation_rows],
        ))
    return variants


def train_variant(variant: Variant, config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Train one variant for one seed, keeping the weights of its best validation epoch.

    Training runs one epoch at a time so an augmented variant can redraw its images
    every epoch. The optimizer instance is reused across epochs, so momentum's state
    survives, exactly as it would inside a single long call.
    """
    rng = np.random.default_rng(seed)
    net = build_model(config["model"], rng)
    optimizer = build("optimizer", config["optimizer"])
    loss = build("loss", config["loss"])
    augmentation = config["augmentation"]

    Y_train = one_hot(variant.y_train)
    history: list[dict[str, float]] = []
    best = {"accuracy": -1.0, "loss": np.inf, "epoch": 0}
    best_weights: list[np.ndarray] = []
    early_stopping = config["training"].get("early_stopping", {})
    patience = early_stopping.get("patience")
    min_delta = early_stopping.get("min_delta", 0.0)
    best_stop_accuracy, stale_epochs = -1.0, 0
    progress_every = int(config["training"].get("progress_every", 25))
    record_train_accuracy = config["training"].get("record_train_accuracy", True)

    for epoch in range(1, config["training"]["epochs"] + 1):
        X_epoch = augment(variant.X_train, rng, **augmentation) if variant.augment else variant.X_train
        train(net, loss, optimizer, X_epoch, Y_train, epochs=1,
              batch_size=config["training"]["batch_size"], rng=rng, callbacks=[])

        validation = digit_metrics(variant.y_validation, net.forward(variant.X_validation))
        train_accuracy = (float(np.mean(net.forward(variant.X_train).argmax(axis=1) == variant.y_train))
                          if record_train_accuracy else None)
        history.append({"epoch": epoch, "train_accuracy": train_accuracy,
                        "validation_accuracy": validation["accuracy"],
                        "validation_loss": validation["loss"],
                        "validation_macro_f1": validation["macro_f1"]})
        if epoch == 1 or epoch % progress_every == 0:
            print(f"{variant.name} seed={seed} epoch={epoch}/{config['training']['epochs']} "
                  f"val_accuracy={validation['accuracy']:.4%} "
                  f"val_macro_f1={validation['macro_f1']:.4f}", flush=True)

        # Select on validation accuracy, breaking ties by lower loss, as ejercicio 2 did.
        if (validation["accuracy"], -validation["loss"]) > (best["accuracy"], -best["loss"]):
            best = {"accuracy": validation["accuracy"], "loss": validation["loss"],
                    "epoch": epoch, "macro_f1": validation["macro_f1"],
                    "per_class": validation["per_class"]}
            best_weights = [param.value.copy() for param in net.params()]

        if patience is not None:
            if validation["accuracy"] > best_stop_accuracy + min_delta:
                best_stop_accuracy, stale_epochs = validation["accuracy"], 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

    for param, value in zip(net.params(), best_weights):
        param.value[...] = value
    best_train_accuracy = history[best["epoch"] - 1]["train_accuracy"]
    if best_train_accuracy is None:
        best_train_accuracy = float(np.mean(net.forward(variant.X_train).argmax(axis=1) == variant.y_train))
    return {"net": net, "history": history, "best": best, "epochs_ran": len(history),
            "train_accuracy": best_train_accuracy}


def _recall(per_class: list[dict[str, Any]], digit: int) -> float | None:
    return per_class[digit]["recall"]


def run_ablation(config: dict[str, Any]) -> dict[str, Any]:
    """Run every step across every seed and summarise each as one row."""
    config = merge_config(config)
    data = merge_digit_files(config["files"])
    train_rows, validation_rows = split_train_validation(
        data.y, config["validation_ratio"], config["split_seed"])

    steps = config.get("study", {}).get("steps") or resolve_steps(
        config["files"], config["baseline_files"])
    variants = build_variants(data, train_rows, validation_rows, steps,
                              np.random.default_rng(config["split_seed"]))

    output = Path(config["output"]["directory"])
    run_root = output / "runs"
    rows, runs, best_models = [], [], {}
    for step_index, variant in enumerate(variants):
        per_seed = []
        for seed_index, seed in enumerate(config["seeds"]):
            run_dir = run_root / f"step_{step_index:02d}" / f"seed_{seed}"
            protocol = {"config": config, "step": {"name": variant.name, "files": variant.files,
                        "balance": variant.balance, "augment": variant.augment}, "seed": seed,
                        "training_rows": len(variant.y_train)}
            protocol_path, summary_path = run_dir / "protocol.json", run_dir / "run_summary.json"
            history_path, weights_path = run_dir / "history.csv", run_dir / "best_weights.npz"
            if summary_path.exists() and protocol_path.exists() and history_path.exists() and weights_path.exists():
                if json.loads(protocol_path.read_text()) != protocol:
                    raise ValueError(f"Saved run protocol differs in {run_dir}")
                saved = json.loads(summary_path.read_text())
                net = build_model(config["model"], np.random.default_rng(seed))
                load_weights(net, weights_path)
                result = {"net": net, "history": pd.read_csv(history_path).to_dict("records"),
                          "best": saved["best"], "epochs_ran": saved["epochs_ran"],
                          "train_accuracy": saved["train_accuracy"]}
            else:
                result = train_variant(variant, config, seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                save_weights(result["net"], weights_path)
                pd.DataFrame(result["history"]).to_csv(history_path, index=False)
                protocol_path.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
                summary = {key: result[key] for key in ("best", "epochs_ran", "train_accuracy")}
                summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            per_seed.append(result)
            print(f"{variant.name} — semilla {seed}: val={result['best']['accuracy']:.4%}, "
                  f"época={result['best']['epoch']}/{result['epochs_ran']}", flush=True)
        accuracies = [result["best"]["accuracy"] for result in per_seed]
        best_models[variant.name] = per_seed[int(np.argmax(accuracies))]
        for seed, result in zip(config["seeds"], per_seed):
            runs.append({"step": variant.name, "seed": seed,
                         "best_epoch": result["best"]["epoch"],
                         "epochs_ran": result["epochs_ran"],
                         "train_accuracy": result["train_accuracy"],
                         "validation_accuracy": result["best"]["accuracy"],
                         "validation_macro_f1": result["best"]["macro_f1"],
                         "recall_5": _recall(result["best"]["per_class"], 5),
                         "recall_8": _recall(result["best"]["per_class"], 8)})

        def mean_of(key: str) -> float | None:
            values = [run[key] for run in runs[-len(config["seeds"]):] if run[key] is not None]
            return float(np.mean(values)) if values else None

        rows.append({
            "step": variant.name,
            "training_files": [Path(path).name for path in variant.files],
            "training_rows": int(len(variant.y_train)),
            "balance": variant.balance, "augment": variant.augment,
            "seeds": len(config["seeds"]),
            "validation_accuracy_mean": float(np.mean(accuracies)),
            "validation_accuracy_std": float(np.std(accuracies)),
            "train_accuracy_mean": float(np.mean([r["train_accuracy"] for r in per_seed])),
            "macro_f1_mean": mean_of("validation_macro_f1"),
            "recall_5_mean": mean_of("recall_5"),
            "recall_8_mean": mean_of("recall_8"),
            "median_best_epoch": float(np.median([r["best"]["epoch"] for r in per_seed])),
        })

    target = config.get("study", {}).get("target_accuracy")
    eligible = [row for row in rows if target is not None and row["validation_accuracy_mean"] >= target]
    if eligible:
        winner = max(eligible, key=lambda row: (row["macro_f1_mean"], row["validation_accuracy_mean"]))
        selection = f"among mean validation accuracy >= {target:.4f}, highest mean macro F1"
    else:
        winner = max(rows, key=lambda row: (row["validation_accuracy_mean"], row["macro_f1_mean"]))
        selection = "highest mean validation accuracy; macro F1 tie-break"
    return {"config": config, "rows": rows, "runs": runs, "winner": winner["step"],
            "selection": selection,
            "validation_samples": int(len(validation_rows)),
            "histories": {variant.name: best_models[variant.name]["history"] for variant in variants},
            "_models": best_models}


def final_test(result: dict[str, Any]) -> dict[str, Any]:
    """Score the winning variant on `digits_test.csv`. Run this once, after choosing."""
    config = result["config"]
    X_test, y_test = load_digit_arrays(config["test_file"])
    net = result["_models"][result["winner"]]["net"]
    metrics = digit_metrics(y_test, net.forward(X_test))
    return {"step": result["winner"], "samples": int(len(y_test)),
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "per_class": metrics["per_class"], "confusion_matrix": metrics["confusion_matrix"]}


def _number(value: float, decimals: int = 2) -> str:
    """Spanish decimal comma. Applied per number, never to a whole sentence."""
    return f"{value:.{decimals}f}".replace(".", ",")


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{_number(100 * value)} %"


def write_report(result: dict[str, Any], output: str | Path, test: dict[str, Any] | None = None) -> Path:
    """Write the comparison table, the per-seed runs, a figure and the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]

    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(result["runs"]).to_csv(output / "runs.csv", index=False)
    serializable = {key: value for key, value in result.items() if key not in ("_models", "histories")}
    if test is not None:
        serializable["test"] = test
    (output / "summary.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    labels = [row["step"] for row in rows]
    means = [100 * row["validation_accuracy_mean"] for row in rows]
    errors = [100 * row["validation_accuracy_std"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(range(len(rows)), means, xerr=errors, capsize=4)
    for index, (mean, row) in enumerate(zip(means, rows)):
        ax.annotate(f"{mean:.2f} %".replace(".", ","), (mean, index), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set(xlabel=f"Accuracy de validación (%), media de {rows[0]['seeds']} semillas",
           title="Cada cambio y lo que aporta")
    fig.tight_layout()
    fig.savefig(output / "ablation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, history in result["histories"].items():
        ax.plot([record["epoch"] for record in history],
                [100 * record["validation_accuracy"] for record in history], label=name)
    ax.set(xlabel="Época", ylabel="Accuracy de validación (%)",
           title="Curvas de validación de la mejor semilla de cada paso")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png", dpi=150)
    plt.close(fig)

    baseline = rows[0]
    winner = next(row for row in rows if row["step"] == result["winner"])
    lines = [
        "# Ejercicio 3: qué aporta cada cambio", "",
        f"Cada fila cambia **una sola cosa** respecto de la anterior y se mide sobre "
        f"**el mismo conjunto de validación** ({result['validation_samples']} imágenes, "
        f"con los diez dígitos). Cada fila se entrena con "
        f"{len(result['config']['seeds'])} semillas; se informa la media y el desvío. "
        f"`digits_test.csv` no interviene en esta tabla.", "",
        "| Paso | Filas de entrenamiento | Accuracy de validación | Recall del 8 | Recall del 5 | F1 macro |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['step']} | {row['training_rows']} | "
            f"{_percent(row['validation_accuracy_mean'])} ± {_number(100 * row['validation_accuracy_std'])} | "
            f"{_percent(row['recall_8_mean'])} | {_percent(row['recall_5_mean'])} | "
            f"{_number(row['macro_f1_mean'], 4)} |"
        )

    gain = 100 * (winner["validation_accuracy_mean"] - baseline["validation_accuracy_mean"])
    first_gain = 100 * (rows[1]["validation_accuracy_mean"] - baseline["validation_accuracy_mean"])
    lines += [
        "", "## Lectura", "",
        f"**(a) Mejor resultado.** El mejor paso es «{winner['step']}», con "
        f"{_percent(winner['validation_accuracy_mean'])} de accuracy de validación, "
        f"{_number(gain)} puntos por encima del punto de partida.", "",
        f"**(b) Qué aportó cada técnica.** La tabla es la respuesta: cada fila agrega "
        f"un solo cambio. De esos {_number(gain)} puntos, {_number(first_gain)} vienen de la "
        f"primera fila, es decir de los datos.", "",
        f"**(c) Qué cambió además de nuestras técnicas.** El salto de la primera a la "
        f"segunda fila no es un modelo mejor: es la aparición del dígito 8, que "
        f"`digits.csv` no contenía. El recall del 8 pasa de "
        f"{_percent(baseline['recall_8_mean'])} a {_percent(rows[1]['recall_8_mean'])} "
        f"sin tocar la red. Ver `reports/digits_datasets/report.md`.", "",
        "La accuracy de entrenamiento no sirve para elegir entre estas filas: aumentar "
        "los datos y desplazar las imágenes **empeoran** el error de entrenamiento a "
        "propósito. Por eso todas las filas se comparan por validación.", "",
    ]

    if test is not None:
        lines += [
            "## Evaluación final sobre el test", "",
            f"Una única medición, con el modelo de «{test['step']}», sobre "
            f"{test['samples']} imágenes de `digits_test.csv` nunca usadas:", "",
            f"- **Accuracy de test: {_percent(test['accuracy'])}**",
            f"- F1 macro: {_number(test['macro_f1'], 4)}",
            f"- Recall del 8: {_percent(_recall(test['per_class'], 8))}",
            f"- Recall del 5: {_percent(_recall(test['per_class'], 5))}", "",
            "| Dígito | Soporte | Recall | F1 |", "| ---: | ---: | ---: | ---: |",
        ]
        for entry in test["per_class"]:
            lines.append(
                f"| {entry['label']} | {entry['support']} | {_percent(entry['recall'])} | "
                f"{_number(entry['f1'], 4)} |"
            )
        lines.append("")

    lines += ["## Reproducir", "", "```bash",
              "python -m experiments.digits.ablation --final-test", "```", ""]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="JSON config; omitted keys use the defaults")
    parser.add_argument("--final-test", action="store_true",
                        help="after ranking, score the winner once on digits_test.csv")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = run_ablation(read_json(args.config) if args.config else {})
    config = result["config"]

    for row in result["rows"]:
        print(f"{row['step']:<38} {100 * row['validation_accuracy_mean']:6.2f} % "
              f"± {100 * row['validation_accuracy_std']:.2f}  (8: {_percent(row['recall_8_mean'])})")

    test = final_test(result) if args.final_test else None
    if test is not None:
        print(f"\nTest ({test['step']}): {100 * test['accuracy']:.2f} %")

    output = args.output or config["output"]["directory"]
    write_report(result, output, test)
    save_weights(result["_models"][result["winner"]]["net"], Path(output) / "best_weights.npz")
    print(f"\nReport: {output}")


if __name__ == "__main__":
    main()
