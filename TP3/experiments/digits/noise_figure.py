"""One figure for the noise optional: what each noise level looks like, aligned with its accuracy.

Top rows: example digits from digits.csv (never the test file) with the same Gaussian
noise the robustness study used. Bottom: the saved test accuracy per σ from
`experiments.digits.noise_sweep` (mean ± std over repetitions), one column per σ, so
each accuracy sits directly under the images it corresponds to. Nothing is retrained
or re-evaluated.

Run from TP3:
    python -m experiments.digits.noise_figure [config.json]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.augmentation import add_gaussian_noise
from datasets.digit_dataset_loader import load_digit_arrays
from experiments.config import Config, deep_merge, read_json

DEFAULTS: Config = {
    "noise_summary": "reports/digits_e3_noise/noise_summary.csv",
    "example_file": "datasets/digits.csv",
    # The digits whose recall falls the most under strong noise.
    "example_digits": [1, 7, 9],
    "seed": 0,
    "output": "reports/digits_e3_noise/noise_examples_and_accuracy.png",
}
TARGET_ACCURACY = 98.0  # percent, from the exercise 3 statement


def example_images(path: str, digits: list[int], rng: np.random.Generator) -> np.ndarray:
    """One random training image per requested digit."""
    X, y = load_digit_arrays(path)
    return np.stack([X[rng.choice(np.flatnonzero(y == digit))] for digit in digits])


def plot(summary: pd.DataFrame, examples: np.ndarray, rng: np.random.Generator, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = summary["sigma"].to_numpy()
    rows = len(examples)
    # Wide and short so it fills a 16:9 slide; each image column sits above its accuracy point.
    fig = plt.figure(figsize=(13, 5.8))
    grid = fig.add_gridspec(rows + 1, len(sigmas), height_ratios=[1] * rows + [2.2], hspace=0.25)
    for row, image in enumerate(examples):
        for column, sigma in enumerate(sigmas):
            ax = fig.add_subplot(grid[row, column])
            noisy = add_gaussian_noise(image[None, :], rng, sigma=sigma)[0]
            ax.imshow(noisy.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"σ = {sigma:g}".replace(".", ","))

    ax = fig.add_subplot(grid[rows, :])
    positions = np.arange(len(sigmas))
    accuracy = 100 * summary["accuracy_mean"].to_numpy()
    spread = 100 * summary["accuracy_std"].fillna(0).to_numpy()
    ax.errorbar(positions, accuracy, yerr=spread, marker="o", capsize=4)
    for position, value in zip(positions, accuracy):
        # Labels of points below the target go underneath so they don't cross the target line.
        below_target = value < TARGET_ACCURACY
        ax.annotate(f"{value:.1f} %".replace(".", ","), (position, value), textcoords="offset points",
                    xytext=(10, -14) if below_target else (0, 8), ha="left" if below_target else "center")
    ax.axhline(TARGET_ACCURACY, color="C3", linestyle=":", label="objetivo 98 %")
    ax.set(xlim=(-0.5, len(sigmas) - 0.5), ylim=(0, 115), ylabel="Accuracy en test (%)")
    ax.set_xticks(positions, [f"σ = {sigma:g}".replace(".", ",") for sigma in sigmas])
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise examples aligned with test accuracy")
    parser.add_argument("config", nargs="?", help="optional JSON config; omitted keys use defaults")
    args = parser.parse_args()
    config = deep_merge(DEFAULTS, read_json(args.config)) if args.config else DEFAULTS
    rng = np.random.default_rng(config["seed"])
    summary = pd.read_csv(config["noise_summary"]).sort_values("sigma")
    examples = example_images(config["example_file"], config["example_digits"], rng)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    plot(summary, examples, rng, output)
    print(f"Figure written to {output}")


if __name__ == "__main__":
    main()
