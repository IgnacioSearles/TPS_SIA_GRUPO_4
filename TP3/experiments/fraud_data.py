"""Loading, inspection and leakage-safe preprocessing for the fraud exercise."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


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
REQUIRED_COLUMNS = FEATURE_COLUMNS + EXCLUDED_COLUMNS + [TEACHER_TARGET]


@dataclass
class StandardScaler:
    """Column-wise standardization fitted exclusively on training data."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        if X.ndim != 2 or len(X) == 0:
            raise ValueError(f"Expected a non-empty 2-D feature matrix, got shape {X.shape}")
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler.transform called before fit")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


@dataclass(frozen=True)
class Fold:
    """Indices for one training/validation partition in cross-validation."""

    train: np.ndarray
    validation: np.ndarray


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


def split_features_and_targets(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return features and BigModel probabilities without target leakage.

    `big_model_fraud_probability` is the regression target and is excluded
    from TinyModel's inputs.
    """
    X = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    teacher_probability = data[[TEACHER_TARGET]].to_numpy(dtype=float)
    return X, teacher_probability


def kfold(
    data: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[Fold]:
    """Return reproducible shuffled K-Fold partitions for probability regression.

    Every row is used once for validation and ``n_splits - 1`` times for
    training.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if n_splits > len(data):
        raise ValueError("n_splits cannot exceed the number of samples")
    shuffled = np.random.default_rng(seed).permutation(len(data))
    all_indices = np.arange(len(data))
    folds = []
    for validation in np.array_split(shuffled, n_splits):
        train = np.setdiff1d(all_indices, validation, assume_unique=True)
        folds.append(Fold(train=train, validation=validation))
    return folds


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


def dataset_profile(data: pd.DataFrame) -> dict[str, object]:
    """Produce JSON/Markdown-friendly descriptive statistics for the EDA report."""
    summary = data.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicates": int(data.duplicated().sum()),
        "missing_values": {name: int(value) for name, value in data.isna().sum().items()},
        "unique_values": {name: int(value) for name, value in data.nunique().items()},
        "summary": summary,
        "correlation_with_teacher": data.corr(numeric_only=True)[TEACHER_TARGET].sort_values(),
    }
