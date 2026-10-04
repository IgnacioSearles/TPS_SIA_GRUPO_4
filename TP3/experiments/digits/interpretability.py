"""Contrastive pixel attribution for the selected E3 digit network.

Uses input-gradient saliency and patch occlusion on a fixed validation split.
The script includes both correct classifications and confident errors; it never
loads digits_test.csv and never trains the model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nn  # noqa: F401 - register model components
from experiments.digits.baseline import digit_metrics
from experiments.digits.data import merge_digit_files, split_train_validation
from nn.network import build_model, load_weights


def _margin_gradient(net, image, predicted, reference):
    """Absolute gradient of score(predicted) - score(reference) by input pixel."""
    scores = net.forward(image[None, :])
    grad_output = np.zeros_like(scores)
    grad_output[0, predicted] = 1.0
    grad_output[0, reference] -= 1.0
    return np.abs(net.backward(grad_output)[0])


def _occlusion_margin(net, image, predicted, reference, patch_size):
    """Change in class-score margin after replacing each patch with background."""
    original_scores = net.forward(image[None, :])[0]
    original_margin = original_scores[predicted] - original_scores[reference]
    image_2d = image.reshape(28, 28)
    heatmap = np.zeros((28, 28), dtype=float)
    for top in range(0, 28, patch_size):
        for left in range(0, 28, patch_size):
            bottom, right = min(top + patch_size, 28), min(left + patch_size, 28)
            occluded = image_2d.copy()
            occluded[top:bottom, left:right] = 0.0
            scores = net.forward(occluded.reshape(1, -1))[0]
            margin = scores[predicted] - scores[reference]
            # Positive: removing this patch reduces the predicted-vs-reference margin.
            heatmap[top:bottom, left:right] = original_margin - margin
    return heatmap


def _attribute(net, X, row, true_label, scores, patch_size):
    predicted = int(scores[row].argmax())
    if predicted == true_label:
        reference = int(np.argsort(scores[row])[-2])  # strongest alternative class
        comparison = "predicted_vs_runner_up"
    else:
        reference = int(true_label)
        comparison = "predicted_vs_true"
    image = X[row]
    return {"validation_row": int(row), "true_label": int(true_label),
            "predicted_label": predicted, "reference_label": reference,
            "comparison": comparison, "predicted_score": float(scores[row, predicted]),
            "reference_score": float(scores[row, reference]),
            "margin": float(scores[row, predicted] - scores[row, reference]),
            "gradient": _margin_gradient(net, image, predicted, reference),
            "occlusion": _occlusion_margin(net, image, predicted, reference, patch_size),
            "image": image}


def run_study(config):
    study_dir = Path(config["architecture_study"])
    study = json.loads((study_dir / "study_config.json").read_text())
    summary = json.loads((study_dir / "study_summary.json").read_text())
    architecture = [int(value) for value in summary["winner"]["architecture"].split("-")]
    model_cfg = dict(study["model"])
    model_cfg["layers"] = architecture
    net = build_model(model_cfg, np.random.default_rng(0))
    load_weights(net, study_dir / "winner_weights.npz")

    data = merge_digit_files(study["files"])
    _, validation_rows = split_train_validation(
        data.y, study["validation_ratio"], study["split_seed"])
    X, y = data.X[validation_rows], data.y[validation_rows]
    scores = net.forward(X)
    predictions = scores.argmax(axis=1)
    metrics = digit_metrics(y, scores)

    # Correct examples: strongest class score for each true digit.
    correct_rows = []
    error_rows = []
    per_class_errors = int(config["errors_per_true_class"])
    for digit in range(10):
        correct = np.flatnonzero((y == digit) & (predictions == digit))
        correct_order = correct[np.argsort(scores[correct, digit])[::-1]]
        correct_rows.extend((int(row), "correct") for row in correct_order[:config["correct_examples_per_class"]])

        errors = np.flatnonzero((y == digit) & (predictions != digit))
        # Rank errors by how confidently the model prefers its wrong answer to truth.
        errors = sorted(errors, key=lambda row: scores[row, predictions[row]] - scores[row, digit], reverse=True)
        chosen, seen_predictions = [], set()
        for row in errors:
            if predictions[row] not in seen_predictions:
                chosen.append(int(row))
                seen_predictions.add(int(predictions[row]))
            if len(chosen) == per_class_errors:
                break
        error_rows.extend((row, "error") for row in chosen)

    examples = [_attribute(net, X, row, int(y[row]), scores,
                           config["occlusion_patch_size"])
                for row, _kind in correct_rows + error_rows]
    kind_by_row = {row: kind for row, kind in correct_rows + error_rows}
    for item in examples:
        item["example_type"] = kind_by_row[item["validation_row"]]

    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for index, item in enumerate(examples):
        table.append({key: item[key] for key in (
            "validation_row", "example_type", "true_label", "predicted_label",
            "reference_label", "comparison", "predicted_score", "reference_score", "margin")}
            | {"example": index,
               "gradient_top_10pct_mean": float(np.mean(np.sort(item["gradient"])[-79:])),
               "occlusion_largest_margin_drop": float(np.max(item["occlusion"]))})
    pd.DataFrame(table).to_csv(output / "attribution_examples.csv", index=False)

    class_metrics = pd.DataFrame(metrics["per_class"])
    class_metrics.to_csv(output / "validation_per_class.csv", index=False)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=int)
    pd.DataFrame(confusion, index=range(10), columns=range(10)).to_csv(
        output / "validation_confusion_matrix.csv", index_label="true\\predicted")
    _plot(examples, output, config["occlusion_patch_size"])

    error_pairs = [{"true": true, "predicted": predicted, "count": int(confusion[true, predicted])}
                   for true in range(10) for predicted in range(10)
                   if true != predicted and confusion[true, predicted] > 0]
    error_pairs.sort(key=lambda pair: pair["count"], reverse=True)
    report = {
        "architecture": architecture,
        "model_selection": "existing E3 validation study",
        "source": "fixed E3 validation split; correct cases and confident misclassifications",
        "validation_samples": int(len(validation_rows)),
        "validation_accuracy": float(metrics["accuracy"]),
        "validation_macro_f1": float(metrics["macro_f1"]),
        "correct_examples_analyzed": len(correct_rows),
        "error_examples_analyzed": len(error_rows),
        "methods": ["absolute input gradient of predicted-class score margin",
                    f"change in predicted-class score margin after zeroing {config['occlusion_patch_size']}x{config['occlusion_patch_size']} pixel patches"],
        "correct_case_reference": "runner-up class",
        "error_case_reference": "true class",
        "largest_validation_confusions": error_pairs[:10],
        "uses_digits_test": False,
        "limitations": [
            "Attributions describe this trained model, not causal evidence about handwriting.",
            "Input gradients are local and can be noisy; occlusion depends on the zero-pixel baseline and patch size.",
            "The examples are selected from validation, which was also used for model selection; they illustrate model behavior, not an independent performance estimate."
        ]
    }
    (output / "interpretability_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def _plot(examples, output, patch_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correct = [item for item in examples if item["example_type"] == "correct"]
    errors = [item for item in examples if item["example_type"] == "error"]
    for name, selected, title in (("correct", correct, "Clasificaciones correctas"),
                                  ("errors", errors, "Errores de clasificación")):
        if not selected:
            continue
        fig, axes = plt.subplots(len(selected), 4, figsize=(13, 2.2 * len(selected)), squeeze=False)
        for row, item in enumerate(selected):
            row_title = (f"real {item['true_label']} → pred {item['predicted_label']} "
                         f"(ref {item['reference_label']}, margen {item['margin']:.2f})")
            panels = [(item["image"].reshape(28, 28), "gray", False),
                      (item["gradient"].reshape(28, 28), "magma", False),
                      (item["occlusion"], "coolwarm", True)]
            for col, (values, cmap, diverging) in enumerate(panels):
                vmax = float(np.max(np.abs(values))) or 1.0 if col > 0 else None
                axes[row, col].imshow(values, cmap=cmap,
                                      vmin=-vmax if diverging else None, vmax=vmax)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(("Imagen", "Gradiente: sensibilidad local",
                                              f"Oclusión {patch_size}×{patch_size}: cambio del margen")[col])
            axes[row, 3].axis("off")
            axes[row, 3].text(0.0, 1.0, row_title + "\n\n"
                              "Rojo: el parche favorece la clase predicha.\n"
                              "Azul: favorece la clase de referencia.\n\n"
                              f"Score predicho: {item['predicted_score']:.3f}\n"
                              f"Score referencia: {item['reference_score']:.3f}",
                              va="top", fontsize=9)
        fig.suptitle(title + ": predicción frente a alternativa", y=1.002)
        fig.tight_layout()
        fig.savefig(output / f"attribution_{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="experiments/digits/configs/e3_interpretability.json")
    args = parser.parse_args()
    run_study(json.loads(Path(args.config).read_text()))
