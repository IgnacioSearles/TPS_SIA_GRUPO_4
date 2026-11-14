import numpy as np
import pytest

from src.analysis.art_sampling_control import aggregate_moments,sampling_coordinates
from src.options import ConfigurationError


def test_total_covariance_includes_within_component_uncertainty():
    means=np.array([[0.,0.],[2.,0.]])
    logs=np.log([[1.,4.],[1.,4.]])
    center,covariance,root=aggregate_moments(means,logs)
    np.testing.assert_allclose(center,[1.,0.])
    np.testing.assert_allclose(covariance,[[2.,0.],[0.,4.]])
    np.testing.assert_allclose(root@root.T,covariance,atol=1e-14)


def test_zero_noise_uses_only_selected_training_means_and_preserves_inputs():
    means=np.array([[0.,1.],[7.,8.]])
    logs=np.zeros_like(means)
    before=means.copy()
    coordinates,_=sampling_coordinates(means,logs,np.zeros((3,2)),np.array([1,0,1]))
    np.testing.assert_array_equal(coordinates['posterior_mixture'],means[[1,0,1]])
    np.testing.assert_array_equal(coordinates['mean_replay'],means[[1,0,1]])
    np.testing.assert_array_equal(coordinates['prior'],np.zeros((3,2)))
    np.testing.assert_array_equal(means,before)


def test_selected_posterior_uses_its_own_diagonal_variance():
    coordinates,_=sampling_coordinates(np.array([[1.,2.],[3.,4.]]),
        np.log([[4.,9.],[16.,25.]]),np.ones((1,2)),np.array([1]))
    np.testing.assert_allclose(coordinates['posterior_mixture'],[[7.,9.]])


@pytest.mark.parametrize('means,logs',[(np.empty((0,2)),np.empty((0,2))),
    (np.ones((2,2)),np.ones((2,3))),(np.array([[np.nan,0]]),np.zeros((1,2)))])
def test_invalid_posterior_statistics_are_rejected(means,logs):
    with pytest.raises(ConfigurationError):aggregate_moments(means,logs)


def test_out_of_training_indices_are_rejected():
    with pytest.raises(ConfigurationError):
        sampling_coordinates(np.ones((2,2)),np.zeros((2,2)),np.ones((1,2)),np.array([2]))
