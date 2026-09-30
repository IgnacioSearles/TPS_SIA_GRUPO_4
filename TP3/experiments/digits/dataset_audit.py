"""Audit of the digit datasets: class coverage, overlap between files and the accuracy ceiling.

Answers, from the data alone and before any model exists, what each file can and
cannot teach: which digits are absent, how many images the files share, and the
highest test accuracy still reachable when a digit never appears in training.

Usage:
    python -m experiments.digits.dataset_audit
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.digit_dataset_loader import load_digit_arrays

N_DIGITS = 10
DEFAULT_DATASETS: dict[str, str] = {
    "digits.csv": "datasets/digits.csv",
    "more_digits.csv": "datasets/more_digits.csv",
    "digits_test.csv": "datasets/digits_test.csv",
}
DEFAULT_TRAIN = "digits.csv"
DEFAULT_TEST = "digits_test.csv"


def class_counts(labels: np.ndarray) -> np.ndarray:
    """Samples per digit, always length 10, so an absent class shows up as a zero."""
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if labels.size and (labels.min() < 0 or labels.max() >= N_DIGITS):
        raise ValueError(f"Labels must be digits 0-9, got range {labels.min()}-{labels.max()}")
    return np.bincount(labels, minlength=N_DIGITS)


def missing_classes(counts: np.ndarray) -> list[int]:
    """Digits with no samples at all. A model cannot learn these, however good it is."""
    return np.flatnonzero(np.asarray(counts) == 0).tolist()


def imbalance_ratio(counts: np.ndarray) -> float:
    """Most frequent present class over least frequent present class. Absent classes are ignored."""
    present = np.asarray(counts)[np.asarray(counts) > 0]
    if not len(present):
        raise ValueError("Every class is empty, there is nothing to compare")
    return float(present.max() / present.min())


def accuracy_ceiling(train_counts: np.ndarray, test_counts: np.ndarray) -> float:
    """Highest test accuracy still reachable: test rows of a class absent from training are lost.

    This is an upper bound that ignores every other source of error, so a real
    model always scores below it.
    """
    test_counts = np.asarray(test_counts, dtype=float)
    total = test_counts.sum()
    if total == 0:
        raise ValueError("The test set is empty")
    unlearnable = test_counts[np.asarray(train_counts) == 0].sum()
    return float((total - unlearnable) / total)


def image_keys(X: np.ndarray) -> np.ndarray:
    """One digest per image row, so identical images compare equal within and across files.

    Rows are cast to float32 first — the precision the loader produces — so a
    digest never depends on how a caller happened to build the array.
    """
    rows = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
    if rows.ndim != 2:
        raise ValueError(f"Expected a 2-D array of flattened images, got shape {rows.shape}")
    return np.array([hashlib.blake2b(row.tobytes(), digest_size=16).hexdigest() for row in rows])


def duplicate_images(keys: Iterable[str]) -> int:
    """How many rows repeat an image already present in the same file."""
    keys = list(keys)
    return len(keys) - len(set(keys))


def shared_images(keys_a: Iterable[str], keys_b: Iterable[str]) -> int:
    """How many distinct images appear in both files."""
    return len(set(keys_a) & set(keys_b))


def audit_datasets(
    paths: dict[str, str | Path],
    train_name: str,
    test_name: str,
) -> dict[str, Any]:
    """Describe every file, their pairwise overlap, and the union of everything but the test set."""
    for required in (train_name, test_name):
        if required not in paths:
            raise ValueError(f"'{required}' is not one of the given datasets: {sorted(paths)}")

    loaded = {}
    for name, path in paths.items():
        X, labels = load_digit_arrays(path)
        loaded[name] = (labels, image_keys(X))

    datasets: dict[str, Any] = {}
    for name, (labels, keys) in loaded.items():
        counts = class_counts(labels)
        datasets[name] = {
            "path": str(paths[name]),
            "samples": int(len(labels)),
            "unique_images": len(set(keys)),
            "duplicate_images": duplicate_images(keys),
            "class_counts": counts.tolist(),
            "missing_classes": missing_classes(counts),
            "imbalance_ratio": imbalance_ratio(counts),
        }

    names = list(loaded)
    overlap = {
        (first, second): shared_images(loaded[first][1], loaded[second][1])
        for index, first in enumerate(names)
        for second in names[index + 1:]
    }

    train_counts = class_counts(loaded[train_name][0])
    test_counts = class_counts(loaded[test_name][0])
    unlearnable = missing_classes(train_counts)

    # Everything except the test set is material you are allowed to train on.
    training_names = [name for name in names if name != test_name]
    union_keys: set[str] = set()
    union_counts = np.zeros(N_DIGITS, dtype=int)
    seen: set[str] = set()
    for name in training_names:
        labels, keys = loaded[name]
        union_keys |= set(keys)
        for label, key in zip(labels, keys):
            if key not in seen:
                seen.add(key)
                union_counts[label] += 1

    return {
        "datasets": datasets,
        "overlap": overlap,
        "train_name": train_name,
        "test_name": test_name,
        "accuracy_ceiling": {
            "train": train_name,
            "test": test_name,
            "value": accuracy_ceiling(train_counts, test_counts),
            "unlearnable_classes": unlearnable,
            "unlearnable_rows": int(test_counts[train_counts == 0].sum()),
            "test_rows": int(test_counts.sum()),
        },
        "union": {
            "names": training_names,
            "unique_images": len(union_keys),
            "rows_before_deduplication": sum(datasets[name]["samples"] for name in training_names),
            "class_counts": union_counts.tolist(),
            "missing_classes": missing_classes(union_counts),
            "imbalance_ratio": imbalance_ratio(union_counts),
            "accuracy_ceiling": accuracy_ceiling(union_counts, test_counts),
            "test_overlap": sum(overlap.get((name, test_name), overlap.get((test_name, name), 0)) for name in training_names),
        },
    }


def _comma(value: float, decimals: int = 1) -> str:
    """Spanish decimal comma, to match the other reports and the deck."""
    return f"{value:.{decimals}f}".replace(".", ",")


def _plot_class_distribution(result: dict[str, Any], output: Path) -> None:
    names = list(result["datasets"])
    positions = np.arange(N_DIGITS)
    width = 0.8 / len(names)
    absent = result["accuracy_ceiling"]["unlearnable_classes"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for index, name in enumerate(names):
        counts = np.array(result["datasets"][name]["class_counts"])
        offset = (index - (len(names) - 1) / 2) * width
        ax.bar(positions + offset, counts, width, label=name)
        # An absent class draws no bar at all, so label the gap or it reads as a plotting glitch.
        for digit in np.flatnonzero(counts == 0):
            ax.annotate("0", (digit + offset, 0), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=10, fontweight="bold")
            ax.annotate("sin muestras", (digit + offset, 0), textcoords="offset points",
                        xytext=(0, 22), ha="center", va="bottom", fontsize=8,
                        rotation=90, color="0.3")

    # The deck's rule: the title states the finding, not the topic.
    train_name = result["train_name"]
    if absent:
        digits = ", ".join(str(digit) for digit in absent)
        subject = f"del dígito {digits}" if len(absent) == 1 else f"de los dígitos {digits}"
        title = f"{train_name} no tiene ninguna muestra {subject}"
    else:
        title = "Muestras por dígito en cada archivo"

    ax.set(xticks=positions, xlabel="Dígito", ylabel="Muestras", title=title)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(loc="upper right", framealpha=1.0)
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=150)
    plt.close(fig)


def _plot_class_share(result: dict[str, Any], output: Path) -> None:
    names = list(result["datasets"])
    positions = np.arange(N_DIGITS)
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for index, name in enumerate(names):
        counts = np.array(result["datasets"][name]["class_counts"], dtype=float)
        share = 100 * counts / counts.sum()
        ax.bar(positions + (index - (len(names) - 1) / 2) * width, share, width, label=name)

    ax.axhline(10, linestyle="--", linewidth=1, color="0.4")
    # Headroom first, so the reference label and the legend never land on a bar.
    ax.set_ylim(0, max(14.0, ax.get_ylim()[1]) * 1.25)
    ax.annotate("10 % = reparto parejo", (-0.45, 10), textcoords="offset points", xytext=(0, 4),
                ha="left", va="bottom", fontsize=8, color="0.4",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    # Name the under-represented digits from the data, so the title cannot go stale.
    union = np.array(result["union"]["class_counts"], dtype=float)
    scarce = np.flatnonzero(100 * union / union.sum() < 6.0).tolist()
    if scarce:
        digits = " y ".join(str(digit) for digit in scarce)
        title = f"Los dígitos {digits} están sub-representados en el entrenamiento, no en el test"
    else:
        title = "Cuánto pesa cada dígito dentro de su archivo"
    ax.set(xticks=positions, xlabel="Dígito", ylabel="Porcentaje del archivo", title=title)
    ax.legend(loc="upper right", ncols=len(names), framealpha=1.0)
    fig.tight_layout()
    fig.savefig(output / "class_share.png", dpi=150)
    plt.close(fig)


def _report_lines(result: dict[str, Any]) -> list[str]:
    names = list(result["datasets"])
    train_name, test_name = result["train_name"], result["test_name"]
    ceiling = result["accuracy_ceiling"]
    union = result["union"]
    train = result["datasets"][train_name]

    lines = [
        "# Auditoría de los conjuntos de dígitos", "",
        "Qué puede y qué no puede enseñar cada archivo. Todo sale de contar las filas de "
        "los CSV; no interviene ningún modelo.", "",
        "## Muestras por dígito", "",
        "| Dígito | " + " | ".join(names) + " |",
        "| ---: | " + " | ".join(["---:"] * len(names)) + " |",
    ]
    for digit in range(N_DIGITS):
        cells = " | ".join(str(result["datasets"][name]["class_counts"][digit]) for name in names)
        lines.append(f"| {digit} | {cells} |")
    lines.append("| **Total** | " + " | ".join(f"**{result['datasets'][name]['samples']}**" for name in names) + " |")

    lines += ["", "## Clases ausentes y techo de accuracy", ""]
    if ceiling["unlearnable_classes"]:
        absent = ceiling["unlearnable_classes"]
        digits = ", ".join(str(digit) for digit in absent)
        # Spanish agreement: one absent digit reads "del dígito 8", several "de los dígitos 5 y 8".
        subject = f"del dígito {digits}" if len(absent) == 1 else f"de los dígitos {digits}"
        pronoun = "ese dígito aparece" if len(absent) == 1 else "esos dígitos aparecen"
        lines += [
            f"`{train_name}` no tiene **ninguna muestra {subject}**. Un modelo "
            f"entrenado solo con ese archivo nunca puede acertarlos: la unidad de salida "
            f"correspondiente no vio jamás un ejemplo positivo.", "",
            f"En `{test_name}` {pronoun} en {ceiling['unlearnable_rows']} de "
            f"{ceiling['test_rows']} filas, así que la accuracy de test queda acotada por "
            f"**{_comma(100 * ceiling['value'], 1)} %** antes de contar cualquier otro error. "
            f"Es una cota superior, no una predicción: un modelo real queda por debajo.", "",
            f"El 98 % que pide el enunciado del ejercicio 3 es **inalcanzable** con "
            f"`{train_name}` solo, por cómo están armados los datos y no por la calidad del modelo.", "",
        ]
    else:
        lines += [f"`{train_name}` tiene muestras de los diez dígitos; no hay clases inalcanzables.", ""]

    lines += ["## Superposición entre archivos", "",
              "| Par de archivos | Imágenes en ambos |", "| --- | ---: |"]
    for (first, second), count in result["overlap"].items():
        lines.append(f"| {first} ∩ {second} | {count} |")
    leak = union["test_overlap"]
    lines += [
        "",
        f"El test no comparte ninguna imagen con los archivos de entrenamiento ({leak})."
        if leak == 0 else
        f"**Atención:** el test comparte {leak} imágenes con los archivos de entrenamiento. "
        "Eso invalida la evaluación y hay que resolverlo antes de medir nada.",
        "",
    ]

    lines += ["## Unir los archivos de entrenamiento", ""]
    duplicates = union["rows_before_deduplication"] - union["unique_images"]
    lines += [
        f"Concatenar {' y '.join(f'`{name}`' for name in union['names'])} da "
        f"{union['rows_before_deduplication']} filas, pero solo "
        f"**{union['unique_images']} imágenes distintas**: {duplicates} están repetidas. "
        "Unir sin quitar los repetidos hace que esas imágenes pesen el doble que las demás "
        "sin que nadie lo haya decidido.", "",
    ]
    if union["missing_classes"]:
        absent = union["missing_classes"]
        digits = ", ".join(str(digit) for digit in absent)
        subject = f"el dígito {digits}" if len(absent) == 1 else f"los dígitos {digits}"
        lines += [f"Incluso después de unir sigue faltando {subject}.", ""]
    else:
        lines += [
            f"Una vez unidos están los diez dígitos, así que el techo de accuracy sube a "
            f"**{_comma(100 * union['accuracy_ceiling'], 1)} %**. Ese salto es lo que "
            f"explica la mayor parte de la mejora entre los ejercicios 2 y 3, y no las "
            f"técnicas que se apliquen encima.", "",
        ]

    counts = np.array(union["class_counts"])
    rarest = int(np.argmin(np.where(counts > 0, counts, counts.max() + 1)))
    most_common = int(np.argmax(counts))
    lines += [
        "## Desbalance que queda", "",
        f"El reparto sigue siendo desparejo: el dígito {most_common} aparece "
        f"{counts[most_common]} veces y el {rarest} solo {counts[rarest]}, una razón de "
        f"**{_comma(union['imbalance_ratio'], 1)} a 1**. El test, en cambio, está parejo "
        f"(alrededor del 10 % por dígito), así que los dígitos escasos pesan en la nota "
        f"final mucho más de lo que pesaron durante el entrenamiento. Conviene compensarlo "
        f"con pesos por clase o remuestreo.", "",
        "## Qué hacer con esto", "",
        f"1. Unir los archivos de entrenamiento quitando las {duplicates} imágenes repetidas.",
        "2. Compensar el desbalance que queda en los dígitos escasos.",
        f"3. Reservar una parte para validación y dejar `{test_name}` para una única medición final.",
        "4. Al comparar técnicas, mirar el error de validación (generalización) y no el de entrenamiento.",
        "",
        "## Reproducir", "", "```bash", "python -m experiments.digits.dataset_audit", "```", "",
    ]
    return lines


def write_report(result: dict[str, Any], output: str | Path) -> Path:
    """Write the figures, the per-class table, the raw summary and the report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    names = list(result["datasets"])
    table = pd.DataFrame(
        {name: result["datasets"][name]["class_counts"] for name in names},
        index=pd.Index(range(N_DIGITS), name="digit"),
    )
    table["union_of_training_files"] = result["union"]["class_counts"]
    table.to_csv(output / "class_counts.csv")

    serializable = dict(result)
    serializable["overlap"] = {f"{first} ∩ {second}": count for (first, second), count in result["overlap"].items()}
    (output / "summary.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_class_distribution(result, output)
    _plot_class_share(result, output)
    (output / "report.md").write_text("\n".join(_report_lines(result)) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/digits_datasets")
    parser.add_argument("--train", default=DEFAULT_TRAIN, help="file whose class coverage sets the ceiling")
    parser.add_argument("--test", default=DEFAULT_TEST)
    args = parser.parse_args()

    result = audit_datasets(DEFAULT_DATASETS, train_name=args.train, test_name=args.test)
    write_report(result, args.output)

    ceiling = result["accuracy_ceiling"]
    print(f"Audit: {args.output}")
    for name, info in result["datasets"].items():
        missing = info["missing_classes"] or "ninguna"
        print(f"  {name}: {info['samples']} filas, clases ausentes: {missing}")
    print(f"  techo de accuracy con {ceiling['train']}: {100 * ceiling['value']:.1f}%")
    print(f"  techo uniendo {' + '.join(result['union']['names'])}: {100 * result['union']['accuracy_ceiling']:.1f}%")


if __name__ == "__main__":
    main()
