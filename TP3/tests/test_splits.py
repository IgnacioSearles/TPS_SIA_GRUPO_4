import numpy as np
import pytest

from data.splits import quantile_strata, stratified_holdout, stratified_kfold


def imbalanced_strata() -> np.ndarray:
    """1000 rows, 10% in stratum 1 (like a rare fraud class)."""
    return np.r_[np.zeros(900, dtype=int), np.ones(100, dtype=int)]


def test_holdout_is_a_disjoint_partition_with_the_requested_size():
    development, test = stratified_holdout(imbalanced_strata(), test_ratio=0.2, seed=0)

    assert len(test) == 200
    assert np.intersect1d(development, test).size == 0
    np.testing.assert_array_equal(np.sort(np.r_[development, test]), np.arange(1000))


def test_holdout_keeps_the_strata_mix():
    strata = imbalanced_strata()
    _, test = stratified_holdout(strata, test_ratio=0.2, seed=0)

    assert strata[test].mean() == pytest.approx(0.1)


def test_holdout_is_reproducible_and_depends_on_seed():
    first = stratified_holdout(imbalanced_strata(), 0.2, seed=1)[1]
    again = stratified_holdout(imbalanced_strata(), 0.2, seed=1)[1]
    other = stratified_holdout(imbalanced_strata(), 0.2, seed=2)[1]

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_holdout_rejects_invalid_ratio():
    with pytest.raises(ValueError, match="between 0 and 1"):
        stratified_holdout(imbalanced_strata(), test_ratio=1.0, seed=0)


def test_kfold_validation_parts_cover_every_row_once_with_equal_strata_mix():
    strata = imbalanced_strata()
    folds = stratified_kfold(strata, n_splits=5, seed=0)

    validation_rows = np.concatenate([fold.validation for fold in folds])
    np.testing.assert_array_equal(np.sort(validation_rows), np.arange(1000))
    for fold in folds:
        assert np.intersect1d(fold.train, fold.validation).size == 0
        assert strata[fold.validation].mean() == pytest.approx(0.1)


def test_quantile_strata_splits_values_into_equal_sized_bins():
    strata = quantile_strata(np.arange(100.0), n_bins=4)

    np.testing.assert_array_equal(np.bincount(strata), [25, 25, 25, 25])
