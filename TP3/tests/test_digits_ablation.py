import numpy as np
import pandas as pd

from experiments.digits.ablation import (
    STEPS,
    build_variants,
    resolve_steps,
    run_ablation,
    training_rows_for,
)
from experiments.digits.data import merge_digit_files, split_train_validation


def digit_csv(path, labels, values):
    images = np.vstack([np.full((1, 784), value) for value in values])
    rows = [", ".join(f"{pixel:.6f}" for pixel in image) for image in images]
    pd.DataFrame({"label": labels, "image": [f"[{row}]" for row in rows]}).to_csv(path, index=False)
    return str(path)


def two_files(tmp_path, per_digit=4):
    """First file has no 8s; the second adds them, and the two share every other image."""
    labels_a, values_a, labels_b, values_b = [], [], [], []
    value = 0.001
    for digit in range(10):
        for _ in range(per_digit):
            value += 0.001
            if digit != 8:
                labels_a.append(digit)
                values_a.append(value)
            labels_b.append(digit)
            values_b.append(value)
    first = digit_csv(tmp_path / "a.csv", labels_a, values_a)
    second = digit_csv(tmp_path / "b.csv", labels_b, values_b)
    return first, second


def small_config(first, second, **overrides):
    config = {
        "files": [first, second],
        "baseline_files": [first],
        "validation_ratio": 0.25,
        "split_seed": 0,
        "seeds": [0],
        "model": {"layers": [784, 8, 10], "activation": "tanh",
                  "output_activation": "sigmoid", "initializer": "xavier"},
        "optimizer": {"name": "momentum", "lr": 0.1, "momentum": 0.9},
        "training": {"epochs": 2, "batch_size": 16},
    }
    config.update(overrides)
    return config


def test_training_rows_are_filtered_by_the_file_they_came_from(tmp_path):
    first, second = two_files(tmp_path)
    data = merge_digit_files([first, second])
    train_rows, _ = split_train_validation(data.y, 0.25, seed=0)

    only_first = training_rows_for(data, train_rows, [first])

    assert set(only_first.tolist()) <= set(train_rows.tolist())
    assert set(data.sources[only_first].tolist()) == {first}
    # The first file has no 8s, so no row credited to it can be an 8.
    assert 8 not in data.y[only_first].tolist()


def test_every_variant_is_scored_on_the_very_same_validation_set(tmp_path):
    first, second = two_files(tmp_path)
    data = merge_digit_files([first, second])
    train_rows, validation_rows = split_train_validation(data.y, 0.25, seed=0)

    steps = resolve_steps([first, second], [first])
    variants = build_variants(data, train_rows, validation_rows, steps, np.random.default_rng(0))

    references = {tuple(variant.validation_rows.tolist()) for variant in variants}
    assert len(references) == 1, "variants must share one validation set to be comparable"


def test_validation_contains_the_digit_the_first_file_lacks(tmp_path):
    first, second = two_files(tmp_path)
    data = merge_digit_files([first, second])
    _, validation_rows = split_train_validation(data.y, 0.25, seed=0)

    # Otherwise the table could not show the baseline failing on 8s.
    assert 8 in data.y[validation_rows].tolist()


def test_balancing_equalizes_the_classes_only_in_training(tmp_path):
    first, second = two_files(tmp_path)
    data = merge_digit_files([first, second])
    train_rows, validation_rows = split_train_validation(data.y, 0.25, seed=0)

    steps = resolve_steps([first, second], [first])
    variants = build_variants(data, train_rows, validation_rows, steps, np.random.default_rng(0))
    balanced = next(variant for variant in variants if variant.balance)
    unbalanced = next(variant for variant in variants if not variant.balance and len(variant.files) == 2)

    counts = np.bincount(balanced.y_train, minlength=10)
    assert len(set(counts[counts > 0].tolist())) == 1
    assert len(balanced.y_train) >= len(unbalanced.y_train)


def test_no_training_image_appears_in_validation(tmp_path):
    from experiments.digits.dataset_audit import image_keys

    first, second = two_files(tmp_path)
    data = merge_digit_files([first, second])
    train_rows, validation_rows = split_train_validation(data.y, 0.25, seed=0)

    steps = resolve_steps([first, second], [first])
    variants = build_variants(data, train_rows, validation_rows, steps, np.random.default_rng(0))

    validation_keys = set(image_keys(data.X[validation_rows]))
    for variant in variants:
        assert validation_keys.isdisjoint(image_keys(variant.X_train)), variant.name


def test_run_ablation_scores_every_step_in_order(tmp_path):
    first, second = two_files(tmp_path, per_digit=8)

    result = run_ablation(small_config(first, second))

    assert [row["step"] for row in result["rows"]] == [step["name"] for step in STEPS]
    for row in result["rows"]:
        assert 0.0 <= row["validation_accuracy_mean"] <= 1.0
        assert row["recall_8_mean"] is not None
    # The step that trains without a single 8 cannot recognise one.
    assert result["rows"][0]["recall_8_mean"] == 0.0


def test_run_ablation_is_reproducible(tmp_path):
    first, second = two_files(tmp_path, per_digit=8)
    config = small_config(first, second, optimizer={"name": "sgd", "lr": 0.1})

    first_run = [row["validation_accuracy_mean"] for row in run_ablation(config)["rows"]]
    second_run = [row["validation_accuracy_mean"] for row in run_ablation(config)["rows"]]

    assert first_run == second_run
