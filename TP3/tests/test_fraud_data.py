import numpy as np
import pandas as pd
import pytest

from experiments.fraud_data import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    StandardScaler,
    TEACHER_TARGET,
    load_fraud_dataset,
    prepare_fold,
    kfold,
    split_features_and_targets,
)


def valid_data() -> pd.DataFrame:
    data = {column: [1.0, 2.0] for column in REQUIRED_COLUMNS}
    data[TEACHER_TARGET] = [0.1, 0.9]
    data["flagged_fraud"] = [0, 1]
    return pd.DataFrame(data)[REQUIRED_COLUMNS + ["flagged_fraud"]]


def test_split_excludes_probability_and_flagged_fraud():
    X, teacher = split_features_and_targets(valid_data())

    assert X.shape == (2, len(FEATURE_COLUMNS))
    np.testing.assert_array_equal(teacher[:, 0], [0.1, 0.9])


def test_loader_discards_extra_columns(tmp_path):
    path = tmp_path / "fraud.csv"
    valid_data().to_csv(path, index=False)

    loaded = load_fraud_dataset(path)

    assert loaded.columns.tolist() == REQUIRED_COLUMNS


def test_scaler_uses_only_its_fit_data():
    scaler = StandardScaler().fit(np.array([[1.0, 5.0], [3.0, 5.0]]))

    np.testing.assert_allclose(scaler.transform(np.array([[1.0, 5.0], [3.0, 8.0]])), [[-1.0, 0.0], [1.0, 3.0]])


def test_loader_rejects_invalid_probability(tmp_path):
    data = valid_data()
    data.loc[0, TEACHER_TARGET] = 1.2
    path = tmp_path / "fraud.csv"
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_fraud_dataset(path)


def test_kfold_validation_sets_are_disjoint_and_exhaustive():
    data = pd.concat([valid_data()] * 10, ignore_index=True)
    folds = kfold(data, n_splits=5, seed=3)

    all_rows = np.concatenate([fold.validation for fold in folds])
    assert len(all_rows) == len(data)
    assert len(np.unique(all_rows)) == len(data)


def test_kfold_scaler_uses_train_only():
    data = pd.concat([valid_data()] * 100, ignore_index=True)
    folds = kfold(data, n_splits=5, seed=10)
    prepared = prepare_fold(data, folds[0])

    np.testing.assert_allclose(prepared.X_train.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(prepared.X_train.std(axis=0), 1.0)


def test_kfold_does_not_require_flagged_fraud():
    data = valid_data()
    data = data.drop(columns="flagged_fraud")
    assert len(kfold(data, n_splits=2)) == 2


def test_kfold_requires_at_least_one_sample_per_fold():
    with pytest.raises(ValueError, match="cannot exceed"):
        kfold(valid_data(), n_splits=3)
