"""Loading, inspection and leakage-safe preprocessing for the fraud exercise."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.preprocessing import StandardScaler
from data.splits import Fold, quantile_strata, stratified_holdout, stratified_kfold

FEATURE_COLUMNS = [
    "amount_usd",
    "quantity_purchased",
    "session_duration_seconds",
    "days_since_last_purchase",
    "account_age_days",
    "items_viewed_before_purchase",
]
EXCLUDED_COLUMNS = ["timestamp", "device_screen_resolution", "time_since_last_login_s"]
TEACHER_TARGET = "big_model_fraud_probability"
# Binary label shipped with the dataset. Not a training target (we distill BigModel's
# probability), but useful in the EDA to see how that probability separates the classes.
FRAUD_LABEL = "flagged_fraud"
REQUIRED_COLUMNS = FEATURE_COLUMNS + EXCLUDED_COLUMNS + [TEACHER_TARGET]
DEFAULT_STRATA_BINS = 10


@dataclass(frozen=True)
class DevelopmentSplit:
    """Original row indices of the development set (training, CV, tuning) and the held-out test set."""

    development_rows: np.ndarray
    test_rows: np.ndarray


@dataclass
class PreparedFold:
    """A fold whose features use standardization learned from its train rows only."""

    X_train: np.ndarray
    X_validation: np.ndarray
    teacher_train: np.ndarray
    teacher_validation: np.ndarray
    scaler: StandardScaler


def load_fraud_dataset(path: str | Path) -> pd.DataFrame:
    """Load only the documented input columns and probability target."""
    data = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data[REQUIRED_COLUMNS].isna().any().any():
        columns = data[REQUIRED_COLUMNS].columns[data[REQUIRED_COLUMNS].isna().any()].tolist()
        raise ValueError(f"Dataset contains missing values in: {columns}")
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in REQUIRED_COLUMNS):
        raise ValueError("All input features and the probability target must be numeric")
    if not np.isfinite(data[REQUIRED_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite values")
    if not data[TEACHER_TARGET].between(0, 1).all():
        raise ValueError(f"{TEACHER_TARGET} must be between 0 and 1")
    return data[REQUIRED_COLUMNS].copy()


def load_labeled_fraud_dataset(path: str | Path) -> pd.DataFrame:
    """Like `load_fraud_dataset`, but also keeps the binary `flagged_fraud` label (for the EDA only)."""
    labels = pd.read_csv(path, usecols=lambda column: column == FRAUD_LABEL)
    if FRAUD_LABEL not in labels.columns:
        raise ValueError(f"Dataset is missing the '{FRAUD_LABEL}' column")
    if not labels[FRAUD_LABEL].isin([0, 1]).all():
        raise ValueError(f"{FRAUD_LABEL} must only contain 0 and 1")
    data = load_fraud_dataset(path)
    data[FRAUD_LABEL] = labels[FRAUD_LABEL].astype(int).to_numpy()
    return data


def split_features_and_targets(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return features and BigModel probabilities without target leakage.

    `big_model_fraud_probability` is the regression target and is excluded
    from TinyModel's inputs.
    """
    X = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    teacher_probability = data[[TEACHER_TARGET]].to_numpy(dtype=float)
    return X, teacher_probability


def target_strata(data: pd.DataFrame, n_bins: int = DEFAULT_STRATA_BINS) -> np.ndarray:
    """Stratum per row from quantile bins of BigModel's probability.

    Stratifying on the continuous target keeps its distribution (including the
    high-probability fraud tail) equal across splits without using the label.
    """
    return quantile_strata(data[TEACHER_TARGET].to_numpy(), n_bins)


def split_development_and_test(
    data: pd.DataFrame,
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
    strata_bins: int = DEFAULT_STRATA_BINS,
) -> DevelopmentSplit:
    """Hold out a stratified test set that no training or tuning decision may use."""
    development_rows, test_rows = stratified_holdout(target_strata(data, strata_bins), test_ratio, seed)
    return DevelopmentSplit(development_rows=development_rows, test_rows=test_rows)


def kfold(
    data: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    strata_bins: int = DEFAULT_STRATA_BINS,
) -> list[Fold]:
    """Reproducible K-Fold partitions stratified on the target probability.

    Every row is used once for validation and ``n_splits - 1`` times for training.
    """
    return stratified_kfold(target_strata(data, strata_bins), n_splits, seed)


def prepare_fold(data: pd.DataFrame, fold: Fold) -> PreparedFold:
    """Separate targets and scale a K-Fold partition using train statistics only."""
    X_train, teacher_train = split_features_and_targets(data.iloc[fold.train])
    X_validation, teacher_validation = split_features_and_targets(data.iloc[fold.validation])
    scaler = StandardScaler().fit(X_train)
    return PreparedFold(
        X_train=scaler.transform(X_train),
        X_validation=scaler.transform(X_validation),
        teacher_train=teacher_train,
        teacher_validation=teacher_validation,
        scaler=scaler,
    )