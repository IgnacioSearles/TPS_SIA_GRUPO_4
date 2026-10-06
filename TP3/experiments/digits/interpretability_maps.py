"""Readable interpretability views of the final exercise 3 network (validation only).

1. Class occlusion map: for every correctly classified validation image of a digit,
   slide a small gray-out window over the image and measure how much the network's
   score for that digit drops. Averaged per digit and drawn over the digit's mean
   image, it shows which regions the network relies on to recognize each digit.
2. First-layer weights: each hidden neuron of the first layer has one weight per
   pixel, so its weights can be drawn as a 28x28 image: red pixels push the neuron
   up, blue pixels push it down. It shows which strokes each neuron responds to.

The score is the output pre-activation (before the final sigmoid), because the
sigmoid saturates near 1 for confident predictions and would hide the differences.
The test file is never loaded.

Run from TP3:
    python -m experiments.digits.interpretability_maps [config.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import nn  # noqa: F401 - register model components
from experiments.config import Config, deep_merge, read_json
from experiments.digits.data import merge_digit_files, split_train_validation
from nn.network import Sequential, build_model, load_weights

DEFAULTS: Config = {
    "study": "reports/digits_e3_final/entrenamiento",
    "run": "architecture_00/seed_7",
    "occlusion": {"window": 4, "stride": 2},
    "first_layer_neurons": 24,
    "output_directory": "reports/digits_e3_interpretability_maps",
}
IMAGE_SIDE = 28


def load_final_model(study_directory: Path, run: str) -> tuple[Sequential, dict]:
    study = json.loads((study_directory / "study_config.json").read_text(encoding="utf-8"))
    architecture = study["study"]["architectures"][0]
    net = build_model({**study["model"], "layers": architecture}, np.random.default_rng(0))
    load_weights(net, study_directory / run / "best_weights.npz")
    return net, study


def validation_images(study: dict) -> tuple[np.ndarray, np.ndarray]:
    data = merge_digit_files(study["files"])
    _, validation_rows = split_train_validation(data.y, study["validation_ratio"], study["split_seed"])
    return data.X[validation_rows], data.y[validation_rows]


def class_scores(net: Sequential, X: np.ndarray) -> np.ndarray:
    """Output pre-activations: every layer except the final sigmoid."""
    for layer in net.layers[:-1]:
        X = layer.forward(X)
    return X


def occlusion_map(net: Sequential, images: np.ndarray, digit: int, window: int, stride: int) -> np.ndarray:
    """Mean drop of the digit's score when each window is grayed out, spread over the window's pixels."""
    baseline = class_scores(net, images)[:, digit]
    importance = np.zeros((IMAGE_SIDE, IMAGE_SIDE))
    coverage = np.zeros((IMAGE_SIDE, IMAGE_SIDE))
    grid = images.reshape(-1, IMAGE_SIDE, IMAGE_SIDE)
    for top in range(0, IMAGE_SIDE - window + 1, stride):
        for left in range(0, IMAGE_SIDE - window + 1, stride):
            occluded = grid.copy()
            occluded[:, top:top + window, left:left + window] = 0.0
            drop = baseline - class_scores(net, occluded.reshape(len(images), -1))[:, digit]
            importance[top:top + window, left:left + window] += drop.mean()
            coverage[top:top + window, left:left + window] += 1
    return importance / np.maximum(coverage, 1)


def most_active_first_layer_neurons(net: Sequential, X: np.ndarray, count: int) -> np.ndarray:
    """First-layer neurons whose activation varies most across validation images."""
    activations = net.layers[1].forward(net.layers[0].forward(X))
    return np.argsort(activations.std(axis=0))[::-1][:count]


def plot_class_occlusion(mean_images: np.ndarray, maps: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = np.percentile(np.clip(maps, 0, None), 99)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4))
    for digit, ax in enumerate(axes.flat):
        ax.imshow(mean_images[digit], cmap="gray_r", vmin=0, vmax=1)
        positive = np.clip(maps[digit], 0, None)
        ax.imshow(np.ma.masked_where(positive < 0.15 * vmax, positive), cmap="Reds", vmin=0, vmax=vmax, alpha=0.75)
        ax.set_title(str(digit), fontsize=16)
        ax.axis("off")
    fig.suptitle("Dónde mira la red para reconocer cada dígito (rojo: taparlo baja el puntaje del dígito correcto)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_first_layer(weights: np.ndarray, neurons: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 8
    rows = int(np.ceil(len(neurons) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(1.6 * columns, 1.6 * rows + 0.5))
    for ax, neuron in zip(axes.flat, neurons):
        image = weights[:, neuron].reshape(IMAGE_SIDE, IMAGE_SIDE)
        limit = np.abs(image).max()
        ax.imshow(image, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.axis("off")
    for ax in axes.flat[len(neurons):]:
        ax.remove()
    fig.suptitle("Pesos de la primera capa (rojo: el píxel activa la neurona; azul: la inhibe)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Class occlusion maps and first-layer weights for the E3 model")
    parser.add_argument("config", nargs="?", help="optional JSON config; omitted keys use defaults")
    args = parser.parse_args()
    config = deep_merge(DEFAULTS, read_json(args.config)) if args.config else DEFAULTS
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)

    net, study = load_final_model(Path(config["study"]), config["run"])
    X, y = validation_images(study)
    predicted = net.forward(X).argmax(axis=1)
    correct = predicted == y
    window, stride = config["occlusion"]["window"], config["occlusion"]["stride"]

    mean_images = np.stack([X[y == digit].mean(axis=0).reshape(IMAGE_SIDE, IMAGE_SIDE) for digit in range(10)])
    maps = np.stack([occlusion_map(net, X[correct & (y == digit)], digit, window, stride) for digit in range(10)])
    neurons = most_active_first_layer_neurons(net, X, config["first_layer_neurons"])
    np.savez(output / "class_occlusion.npz", maps=maps, mean_images=mean_images, neurons=neurons)
    plot_class_occlusion(mean_images, maps, output / "class_occlusion.png")
    plot_first_layer(net.layers[0].W.value, neurons, output / "first_layer_weights.png")

    summary = {"model": f"{config['study']}/{config['run']}", "validation_samples": int(len(y)),
               "validation_accuracy": float(correct.mean()),
               "images_per_digit": [int((correct & (y == digit)).sum()) for digit in range(10)],
               "occlusion": config["occlusion"], "score": "output pre-activation (before sigmoid)",
               "first_layer_neurons": neurons.tolist(), "external_test_evaluated": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
