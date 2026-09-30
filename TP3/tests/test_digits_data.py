import numpy as np
import pandas as pd
import pytest

from experiments.digits.data import (
    N_DIGITS,
    balanced_indices,
    class_weights,
    load_training_data,
    merge_digit_files,
    one_hot,
    split_train_validation,
)
from experiments.digits.dataset_audit import class_counts, image_keys


def digit_csv(path, labels, images):
    rows = [", ".join(f"{value:.6f}" for value in image) for image in images]
    pd.DataFrame({"label": labels, "image": [f"[{row}]" for row in rows]}).to_csv(path, index=False)
    return path


def flat(value):
    return np.full((1, 784), value)


def two_files(tmp_path, conflicting=False):
    """Two files sharing exactly the images 0.01 and 0.02, with 3 and 4 unique rows of their own."""
    shared = [flat(0.01), flat(0.02)]
    first = digit_csv(tmp_path / "a.csv", [1, 2, 3, 4, 5],
                      np.vstack(shared + [flat(0.11), flat(0.12), flat(0.13)]))
    second_labels = [9, 2, 6, 7, 8, 0] if conflicting else [1, 2, 6, 7, 8, 0]
    second = digit_csv(tmp_path / "b.csv", second_labels,
                       np.vstack(shared + [flat(0.21), flat(0.22), flat(0.23), flat(0.24)]))
    return first, second


def test_merge_keeps_every_distinct_image_exactly_once(tmp_path):
    first, second = two_files(tmp_path)

    data = merge_digit_files([first, second])

    # 5 + 6 rows, minus the 2 images present in both.
    assert len(data.y) == 9
    assert len(set(image_keys(data.X))) == 9


def test_merge_records_which_file_each_row_came_from(tmp_path):
    first, second = two_files(tmp_path)

    data = merge_digit_files([first, second])

    assert data.sources[0] == str(first)
    # The two shared images are credited to the file that introduced them.
    assert list(data.sources).count(str(first)) == 5
    assert list(data.sources).count(str(second)) == 4


def test_merge_rejects_the_same_image_carrying_two_labels(tmp_path):
    first, second = two_files(tmp_path, conflicting=True)

    with pytest.raises(ValueError, match="conflicting labels"):
        merge_digit_files([first, second])


def test_merge_of_one_file_changes_nothing(tmp_path):
    first, _ = two_files(tmp_path)

    data = merge_digit_files([first])

    assert len(data.y) == 5


def test_one_hot_has_one_column_per_digit(tmp_path):
    encoded = one_hot(np.array([0, 3, 9]))

    assert encoded.shape == (3, N_DIGITS)
    assert encoded[1, 3] == 1.0
    assert encoded.sum(axis=1).tolist() == [1.0, 1.0, 1.0]


def test_class_weights_make_every_present_class_count_the_same():
    labels = np.array([0] * 90 + [1] * 10)

    weights = class_weights(labels)

    assert weights[1] == pytest.approx(9 * weights[0])
    # Rare class contributes as much total weight as the common one.
    assert 90 * weights[0] == pytest.approx(10 * weights[1])
    # Absent classes cannot be learned, so they carry no weight.
    assert weights[5] == 0.0


def test_class_weights_average_to_one_over_the_samples():
    labels = np.array([0] * 90 + [1] * 10)

    assert float(np.mean(class_weights(labels)[labels])) == pytest.approx(1.0)


def test_balanced_indices_equalize_the_classes_without_inventing_rows():
    labels = np.array([0] * 9 + [1] * 3)

    indices = balanced_indices(labels, np.random.default_rng(0))

    counts = class_counts(labels[indices])
    assert counts[0] == counts[1] == 9
    assert set(indices.tolist()) <= set(range(len(labels)))


def test_balanced_indices_are_reproducible_from_the_seed():
    labels = np.array([0] * 9 + [1] * 3)

    first = balanced_indices(labels, np.random.default_rng(7))
    second = balanced_indices(labels, np.random.default_rng(7))

    assert first.tolist() == second.tolist()


def test_split_is_disjoint_covers_everything_and_keeps_the_class_mix():
    labels = np.repeat(np.arange(N_DIGITS), 50)

    train, validation = split_train_validation(labels, validation_ratio=0.2, seed=0)

    assert set(train.tolist()).isdisjoint(validation.tolist())
    assert len(train) + len(validation) == len(labels)
    assert class_counts(labels[validation]).tolist() == [10] * N_DIGITS


def test_split_is_reproducible_from_the_seed():
    labels = np.repeat(np.arange(N_DIGITS), 50)

    assert split_train_validation(labels, 0.2, seed=3)[1].tolist() == split_train_validation(labels, 0.2, seed=3)[1].tolist()
    assert split_train_validation(labels, 0.2, seed=3)[1].tolist() != split_train_validation(labels, 0.2, seed=4)[1].tolist()


def test_load_training_data_returns_disjoint_one_hot_splits(tmp_path):
    images = np.vstack([flat(0.01 * (index + 1)) for index in range(30)])
    path = digit_csv(tmp_path / "all.csv", list(np.repeat(np.arange(N_DIGITS), 3)), images)

    split = load_training_data([path], validation_ratio=1 / 3, seed=0)

    assert split.X_train.shape[1] == 784
    assert split.Y_train.shape == (20, N_DIGITS)
    assert split.Y_validation.shape == (10, N_DIGITS)
    assert len(split.class_weights) == N_DIGITS
    # No image may sit on both sides of the split.
    assert set(image_keys(split.X_train)).isdisjoint(image_keys(split.X_validation))
