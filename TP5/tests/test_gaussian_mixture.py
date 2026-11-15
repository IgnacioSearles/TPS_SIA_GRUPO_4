import numpy as np
import pytest

from src.analysis.gaussian_mixture import fit_mixture,sample_mixture


def test_two_separated_populations_are_recovered_without_labels():
    rng=np.random.default_rng(3)
    samples=np.vstack([rng.normal([-4,0],[.4,.7],size=(400,2)),
                       rng.normal([4,0],[.4,.7],size=(400,2))])
    fitted=fit_mixture(samples,2,seed=7)
    order=np.argsort(fitted['means'][:,0])
    np.testing.assert_allclose(fitted['means'][order],[[-4,0],[4,0]],atol=.15)
    np.testing.assert_allclose(fitted['weights'],[.5,.5],atol=.03)
    assert np.linalg.eigvalsh(fitted['covariances']).min()>0


def test_one_component_matches_empirical_moments_and_is_reproducible():
    samples=np.random.default_rng(4).normal(size=(100,3))
    fitted=fit_mixture(samples,1,seed=7)
    repeat=fit_mixture(samples,1,seed=7)
    np.testing.assert_allclose(fitted['means'][0],samples.mean(axis=0))
    expected=np.cov(samples,rowvar=False,bias=True)+1e-5*np.eye(3)
    np.testing.assert_allclose(fitted['covariances'][0],expected)
    np.testing.assert_array_equal(fitted['covariances'],repeat['covariances'])


def test_zero_noise_selects_expected_component_centers():
    fitted={'weights':np.array([.25,.75]),'means':np.array([[1.,2.],[3.,4.]]),
            'covariances':np.array([np.eye(2),np.eye(2)])}
    samples,indices=sample_mixture(fitted,np.zeros((3,2)),np.array([.1,.3,.9]))
    np.testing.assert_array_equal(indices,[0,1,1])
    np.testing.assert_array_equal(samples,[[1,2],[3,4],[3,4]])


@pytest.mark.parametrize('data,k',[(np.zeros((1,2)),1),(np.ones((3,2)),4),
                                  (np.array([[0,np.nan],[1,2]]),1)])
def test_invalid_fits_are_rejected(data,k):
    with pytest.raises(ValueError):fit_mixture(data,k)
