"""Feature scalers. Fit them on training rows only, then reuse them to transform validation and test rows."""

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    """Column-wise standardization (mean 0, standard deviation 1)."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        if X.ndim != 2 or len(X) == 0:
            raise ValueError(f"Expected a non-empty 2-D feature matrix, got shape {X.shape}")
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # A constant column would divide by zero; leave it centered but unscaled.
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler.transform called before fit")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
