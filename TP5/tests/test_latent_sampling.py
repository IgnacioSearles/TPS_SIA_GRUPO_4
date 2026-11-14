import numpy as np
import pytest

from src.analysis.latent_sampling import fit_gaussian, sample_fitted


def test_gaussian_preserves_full_covariance_including_singular_case():
    codes = np.array([[1, 3], [2, 5], [4, 9]], dtype=float)
    mean, covariance, root = fit_gaussian(codes)
    np.testing.assert_allclose(mean, codes.mean(axis=0))
    np.testing.assert_allclose(covariance, np.cov(codes, rowvar=False))
    np.testing.assert_allclose(root @ root.T, covariance, atol=1e-14)
    _, covariance, root = fit_gaussian(np.ones((4, 1)))
    np.testing.assert_array_equal(root @ root.T, covariance)


def test_sampling_only_encodes_supplied_training_and_is_reproducible():
    class Model:
        precision = np.float32

        def encode(self, samples):
            np.testing.assert_array_equal(samples, training)
            return samples[:, :2]

        def decode(self, coordinates):
            return coordinates * 2

    training = np.array([[1, 2, 99], [2, 4, 88], [3, 3, 77]], dtype=np.float32)
    epsilon = np.random.default_rng(1234).standard_normal((64, 2))
    first = sample_fitted(Model(), training, epsilon)
    second = sample_fitted(Model(), training, epsilon)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    samples, coordinates, mean, covariance = first
    np.testing.assert_array_equal(samples, coordinates * 2)
    assert coordinates.dtype == np.float32
    np.testing.assert_allclose(mean, training[:, :2].mean(axis=0))
    np.testing.assert_allclose(covariance, np.cov(training[:, :2], rowvar=False))


@pytest.mark.parametrize("codes", [np.ones((1, 2)), [[np.nan], [1]], [1, 2]])
def test_invalid_codes_fail(codes):
    with pytest.raises(ValueError):
        fit_gaussian(codes)
