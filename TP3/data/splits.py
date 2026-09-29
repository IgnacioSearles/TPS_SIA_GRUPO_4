"""Reproducible, stratified dataset splits that return row-index arrays.

Both splitters use systematic sampling: rows are shuffled within each stratum,
lined up stratum by stratum, and then dealt out in order (every k-th row goes to
the same fold). This keeps every stratum's share equal across the parts, even
when some strata have only a handful of rows.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    """Indices for one training/validation partition in cross-validation."""

    train: np.ndarray
    validation: np.ndarray


def quantile_strata(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Stratum id per row from quantile bins of a continuous variable (e.g. a regression target)."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right")


def _stratified_order(strata: np.ndarray, seed: int) -> np.ndarray:
    """Row indices grouped by stratum, in random order inside each stratum."""
    random_keys = np.random.default_rng(seed).permutation(len(strata))
    return np.lexsort((random_keys, strata))


def stratified_holdout(strata: np.ndarray, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into (development, test) index arrays with the same strata mix in both."""
    strata = np.asarray(strata).reshape(-1)
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    order = _stratified_order(strata, seed)
    positions = np.arange(len(order))
    # A row goes to test whenever the running count of test rows (position * ratio) crosses an integer.
    is_test = np.floor((positions + 1) * test_ratio) > np.floor(positions * test_ratio)
    if is_test.all() or not is_test.any():
        raise ValueError(f"test_ratio {test_ratio} leaves an empty split for {len(strata)} rows")
    return np.sort(order[~is_test]), np.sort(order[is_test])


def stratified_kfold(strata: np.ndarray, n_splits: int, seed: int) -> list[Fold]:
    """K folds whose validation parts share the strata mix of the whole set."""
    strata = np.asarray(strata).reshape(-1)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if n_splits > len(strata):
        raise ValueError(f"n_splits ({n_splits}) cannot exceed the number of rows ({len(strata)})")
    order = _stratified_order(strata, seed)
    fold_of_row = np.empty(len(strata), dtype=int)
    fold_of_row[order] = np.arange(len(order)) % n_splits
    return [
        Fold(train=np.flatnonzero(fold_of_row != fold), validation=np.flatnonzero(fold_of_row == fold))
        for fold in range(n_splits)
    ]
