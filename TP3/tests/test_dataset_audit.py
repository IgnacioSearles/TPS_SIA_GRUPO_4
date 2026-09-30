import numpy as np
import pandas as pd

from experiments.digits.dataset_audit import (
    accuracy_ceiling,
    audit_datasets,
    class_counts,
    duplicate_images,
    image_keys,
    imbalance_ratio,
    missing_classes,
    shared_images,
    write_report,
)

EXPECTED_OUTPUTS = {"class_distribution.png", "class_share.png", "class_counts.csv", "summary.json", "report.md"}


def digit_csv(path, labels, images=None):
    """Write a CSV in the loader's format: one row per image, pixels as '[v, v, ...]'."""
    rng = np.random.default_rng(0)
    if images is None:
        images = rng.uniform(0, 1, size=(len(labels), 784))
    rows = [", ".join(f"{value:.6f}" for value in image) for image in images]
    pd.DataFrame({"label": labels, "image": [f"[{row}]" for row in rows]}).to_csv(path, index=False)
    return path


def test_class_counts_has_one_entry_per_digit_including_zeros():
    counts = class_counts(np.array([0, 0, 3, 9]))

    assert len(counts) == 10
    assert counts[0] == 2 and counts[3] == 1 and counts[9] == 1
    assert counts[8] == 0


def test_missing_classes_lists_absent_digits():
    counts = class_counts(np.array([0, 1, 2, 3, 4, 5, 6, 7, 9]))

    assert missing_classes(counts) == [8]


def test_accuracy_ceiling_drops_test_rows_of_classes_absent_from_training():
    train = class_counts(np.array([0, 0, 1, 1]))
    test = class_counts(np.array([0, 1, 8, 8]))

    # Half the test rows are 8s and training has none of them.
    assert accuracy_ceiling(train, test) == 0.5


def test_accuracy_ceiling_is_one_when_every_test_class_is_represented():
    train = class_counts(np.array([0, 1, 8]))
    test = class_counts(np.array([0, 8]))

    assert accuracy_ceiling(train, test) == 1.0


def test_imbalance_ratio_ignores_absent_classes():
    counts = class_counts(np.array([0] * 30 + [1] * 10))

    assert imbalance_ratio(counts) == 3.0


def test_image_keys_match_for_identical_images_only():
    images = np.array([[0.5] * 784, [0.5] * 784, [0.25] * 784])

    keys = image_keys(images)

    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


def test_shared_and_duplicate_images_are_counted():
    a = np.array([[0.1] * 784, [0.2] * 784, [0.2] * 784])
    b = np.array([[0.2] * 784, [0.9] * 784])

    assert duplicate_images(image_keys(a)) == 1
    assert shared_images(image_keys(a), image_keys(b)) == 1


def flat(value):
    return np.full((1, 784), value)


def test_audit_reports_the_missing_class_and_the_ceiling(tmp_path):
    shared = [flat(0.01), flat(0.02)]
    train = digit_csv(
        tmp_path / "train.csv",
        [0, 1, 2, 3, 4, 5, 6, 7, 9],
        np.vstack(shared + [flat(0.03 + 0.01 * i) for i in range(7)]),
    )
    extra = digit_csv(
        tmp_path / "extra.csv",
        list(range(10)),
        np.vstack(shared + [flat(0.11 + 0.01 * i) for i in range(8)]),
    )
    test = digit_csv(tmp_path / "test.csv", [0, 8], np.vstack([flat(0.21), flat(0.22)]))

    result = audit_datasets({"train": train, "extra": extra, "test": test}, train_name="train", test_name="test")

    assert result["datasets"]["train"]["missing_classes"] == [8]
    assert result["datasets"]["extra"]["missing_classes"] == []
    assert result["datasets"]["train"]["duplicate_images"] == 0
    assert result["overlap"][("train", "extra")] == 2
    assert result["overlap"][("train", "test")] == 0
    # Half the test rows are 8s, which training never shows.
    assert result["accuracy_ceiling"]["value"] == 0.5
    # Merging train with extra covers every digit, so nothing is unlearnable.
    assert result["union"]["missing_classes"] == []
    assert result["union"]["accuracy_ceiling"] == 1.0
    assert result["union"]["unique_images"] == 17  # 9 + 10 rows, minus the 2 shared images


def test_write_report_produces_every_figure_and_file(tmp_path):
    train = digit_csv(tmp_path / "train.csv", [0, 1, 2, 2])
    test = digit_csv(tmp_path / "test.csv", [0, 1, 2])
    result = audit_datasets({"train": train, "test": test}, train_name="train", test_name="test")

    write_report(result, tmp_path / "out")

    assert {path.name for path in (tmp_path / "out").iterdir()} == EXPECTED_OUTPUTS
