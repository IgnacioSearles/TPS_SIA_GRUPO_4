import numpy as np
import pandas as pd
import pytest

from experiments.fraud.eda import write_eda
from experiments.fraud.data import (
    FRAUD_LABEL,
    REQUIRED_COLUMNS,
    TEACHER_TARGET,
    load_labeled_fraud_dataset,
)

EXPECTED_FIGURES = {
    "target_distribution.png",
    "correlation_matrices.png",
    "target_correlations.png",
    "feature_vs_target.png",
    "model_features_vs_target.png",
    "feature_distributions_by_class.png",
    "feature_scales.png",
}


def synthetic_fraud_data(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = pd.DataFrame({column: rng.uniform(1, 100, n_rows) for column in REQUIRED_COLUMNS})
    data["quantity_purchased"] = rng.integers(1, 25, n_rows)
    data[TEACHER_TARGET] = rng.uniform(0, 1, n_rows)
    data[FRAUD_LABEL] = (data[TEACHER_TARGET] > 0.85).astype(int)
    return data


def test_labeled_loader_keeps_fraud_label(tmp_path):
    path = tmp_path / "fraud.csv"
    synthetic_fraud_data().to_csv(path, index=False)

    loaded = load_labeled_fraud_dataset(path)

    assert loaded.columns.tolist() == REQUIRED_COLUMNS + [FRAUD_LABEL]
    assert set(loaded[FRAUD_LABEL].unique()) <= {0, 1}


def test_labeled_loader_rejects_non_binary_label(tmp_path):
    data = synthetic_fraud_data()
    data.loc[0, FRAUD_LABEL] = 2
    path = tmp_path / "fraud.csv"
    data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="only contain 0 and 1"):
        load_labeled_fraud_dataset(path)


def test_write_eda_produces_every_figure_and_summary(tmp_path):
    write_eda(synthetic_fraud_data(), tmp_path)

    written = {path.name for path in tmp_path.iterdir()}
    assert EXPECTED_FIGURES | {"data_summary.csv"} == written
