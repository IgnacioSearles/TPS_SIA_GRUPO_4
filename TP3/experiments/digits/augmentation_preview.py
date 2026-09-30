"""Eyeball the augmentation before trusting it: python -m experiments.digits.augmentation_preview

Writes a before/after grid and a noise ladder. If a nudged 7 still reads as a 7 the
settings are fine; if strokes are leaving the frame or dissolving, they are too strong.
No model is involved.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.augmentation import IMAGE_SIZE, add_gaussian_noise, random_affine
from experiments.digits.data import DEFAULT_TRAINING_FILES, merge_digit_files

NOISE_LEVELS = (0.0, 0.1, 0.2, 0.4, 0.6)


def _show(ax, image: np.ndarray) -> None:
    ax.imshow(image.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def write_preview(X: np.ndarray, y: np.ndarray, output: Path, seed: int = 0, variants: int = 7) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    digits = sorted(set(y.tolist()))

    fig, axes = plt.subplots(len(digits), variants + 1, figsize=(1.1 * (variants + 1), 1.1 * len(digits)))
    for row, digit in enumerate(digits):
        original = X[np.flatnonzero(y == digit)[0]][None]
        _show(axes[row, 0], original[0])
        axes[row, 0].set_ylabel(str(digit), rotation=0, labelpad=10, va="center")
        for column in range(1, variants + 1):
            _show(axes[row, column], random_affine(original, rng)[0])
    axes[0, 0].set_title("original", fontsize=9)
    axes[0, (variants + 2) // 2].set_title("versiones desplazadas, rotadas y escaladas", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "examples.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(digits), len(NOISE_LEVELS), figsize=(1.1 * len(NOISE_LEVELS), 1.1 * len(digits)))
    for row, digit in enumerate(digits):
        original = X[np.flatnonzero(y == digit)[0]][None]
        for column, sigma in enumerate(NOISE_LEVELS):
            _show(axes[row, column], add_gaussian_noise(original, rng, sigma=sigma)[0])
            if row == 0:
                axes[row, column].set_title(f"σ = {sigma:.1f}".replace(".", ","), fontsize=9)
        axes[row, 0].set_ylabel(str(digit), rotation=0, labelpad=10, va="center")
    fig.tight_layout()
    fig.savefig(output / "noise_levels.png", dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/digits_augmentation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = merge_digit_files(DEFAULT_TRAINING_FILES)
    write_preview(data.X, data.y, Path(args.output), seed=args.seed)
    print(f"Preview: {args.output}")


if __name__ == "__main__":
    main()
