"""Loading, merging and splitting the handwritten digit files for ejercicios 2 and 3.

`digits.csv` and `more_digits.csv` share 3689 images, so concatenating them would
weight those twice without anyone deciding to. Everything here works on distinct
images: merge first, then split, then (only afterwards) augment the training side.
Splitting before augmenting is what keeps a rotated copy of a training image out
of validation.

`digits_test.csv` is never loaded here. It stands in for production and is read
once, at the end, by whatever reports the final number.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from data.splits import stratified_holdout
from datasets.digit_dataset_loader import load_digit_arrays
from experiments.digits.dataset_audit import N_DIGITS, image_keys

DEFAULT_TRAINING_FILES: list[str] = ["datasets/digits.csv", "datasets/more_digits.csv"]
TEST_FILE = "datasets/digits_test.csv"


@dataclass(frozen=True)
class DigitData:
    """Distinct images with their labels, and the file each one first appeared in."""

    X: np.ndarray
    y: np.ndarray
    sources: np.ndarray


@dataclass(frozen=True)
class TrainingSplit:
    """Training and validation halves, labels one-hot encoded for a 10-output network."""

    X_train: np.ndarray
    Y_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    Y_validation: np.ndarray
    y_validation: np.ndarray
    class_weights: np.ndarray


def merge_digit_files(paths: Sequence[str | Path]) -> DigitData:
    """Concatenate the files, keeping each distinct image once.

    An image kept from an earlier file wins, and its `source` records where it came
    from. The same image carrying different labels in two files is a data problem
    rather than something to resolve silently, so it raises.
    """
    if not paths:
        raise ValueError("Give at least one dataset file to merge")

    images: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[str] = []
    label_of_image: dict[str, tuple[int, str]] = {}

    for path in paths:
        X, y = load_digit_arrays(path)
        for image, label, key in zip(X, y, image_keys(X)):
            seen = label_of_image.get(key)
            if seen is not None:
                if seen[0] != label:
                    raise ValueError(
                        f"One image carries conflicting labels: {seen[0]} in {seen[1]} "
                        f"and {label} in {path}"
                    )
                continue
            label_of_image[key] = (int(label), str(path))
            images.append(image)
            labels.append(int(label))
            sources.append(str(path))

    return DigitData(
        X=np.stack(images),
        y=np.array(labels, dtype=np.int64),
        sources=np.array(sources),
    )


def one_hot(y: np.ndarray, n_classes: int = N_DIGITS) -> np.ndarray:
    """Labels as one row per sample with a single 1, which is what a 10-output network needs."""
    y = np.asarray(y, dtype=int).reshape(-1)
    if y.size and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError(f"Labels must be in [0, {n_classes}), got range {y.min()}-{y.max()}")
    encoded = np.zeros((len(y), n_classes), dtype=float)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def class_weights(y: np.ndarray, n_classes: int = N_DIGITS) -> np.ndarray:
    """Per-class weights that give every present class the same total say.

    A class with a tenth of the samples gets ten times the weight. Weights average
    to 1 across the samples, so switching them on does not silently rescale the
    learning rate. Absent classes get 0: nothing can be learned about them.
    """
    counts = np.bincount(np.asarray(y, dtype=int).reshape(-1), minlength=n_classes).astype(float)
    present = counts > 0
    if not present.any():
        raise ValueError("No labels given, cannot compute class weights")
    weights = np.zeros(n_classes)
    weights[present] = len(y) / (present.sum() * counts[present])
    return weights


def balanced_indices(y: np.ndarray, rng: np.random.Generator, n_classes: int = N_DIGITS) -> np.ndarray:
    """Row indices that repeat the scarce classes until every present class is equally common.

    Nothing is invented — rare rows are simply drawn more than once. This is the
    data-side alternative to weighting the loss, and it needs no change to the
    training code.
    """
    y = np.asarray(y, dtype=int).reshape(-1)
    counts = np.bincount(y, minlength=n_classes)
    target = int(counts.max())
    if target == 0:
        raise ValueError("No labels given, cannot balance")

    chosen = []
    for digit in np.flatnonzero(counts):
        rows = np.flatnonzero(y == digit)
        # Keep every original row, then draw the shortfall from the same rows.
        extra = rng.choice(rows, size=target - len(rows), replace=True) if len(rows) < target else np.array([], dtype=int)
        chosen.append(np.concatenate([rows, extra]))
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices


def split_train_validation(
    y: np.ndarray,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into (train, validation), keeping each digit's share in both.

    The labels are the strata, so a digit that is 3% of the data is 3% of both
    halves. Reproducible from `seed`.
    """
    train, validation = stratified_holdout(np.asarray(y).reshape(-1), validation_ratio, seed)
    return train, validation


def load_training_data(
    paths: Sequence[str | Path] = tuple(DEFAULT_TRAINING_FILES),
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> TrainingSplit:
    """Merge the training files, split off validation, and one-hot encode the labels.

    Class weights are computed from the training half only — validation must not
    influence anything the model sees.
    """
    data = merge_digit_files(list(paths))
    train_rows, validation_rows = split_train_validation(data.y, validation_ratio, seed)
    return TrainingSplit(
        X_train=data.X[train_rows],
        Y_train=one_hot(data.y[train_rows]),
        y_train=data.y[train_rows],
        X_validation=data.X[validation_rows],
        Y_validation=one_hot(data.y[validation_rows]),
        y_validation=data.y[validation_rows],
        class_weights=class_weights(data.y[train_rows]),
    )
