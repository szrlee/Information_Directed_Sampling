import numpy as np
from utils import sample_noise


def test_sample_noise_sphere():
    M = 10
    dim = 3
    noise_type = "Sphere"
    noise = sample_noise(noise_type, M, dim)
    assert noise.shape == (dim, M)
    assert np.allclose(np.linalg.norm(noise, axis=1), np.ones(dim))


def test_sample_noise_gaussian():
    M = 10
    dim = 3
    noise_type = "Gaussian"
    noise = sample_noise(noise_type, M, dim)
    assert noise.shape == (dim, M)


def test_sample_noise_pmcoord():
    M = 10
    dim = 3
    noise_type = "PMCoord"
    noise = sample_noise(noise_type, M, dim)
    assert noise.shape == (dim, M)
    assert np.allclose(np.linalg.norm(noise, axis=1), np.ones(dim))


def test_sample_noise_sparse():
    M = 10
    dim = 3
    noise_type = "Sparse"
    noise = sample_noise(noise_type, M, dim)
    assert noise.shape == (dim, M)
    assert np.allclose(np.linalg.norm(noise, axis=1), np.ones(dim))


def test_sample_noise_sparse_consistent():
    M = 10
    dim = 3
    noise_type = "SparseConsistent"
    noise = sample_noise(noise_type, M, dim)
    assert noise.shape == (dim, M)
    assert np.allclose(np.linalg.norm(noise, axis=1), np.ones(dim))


def test_sample_noise_unifcube():
    M = 10
    dim = 3
    noise_type = "UnifCube"
    noise = sample_noise(noise_type, M, dim)
    assert noise.shape == (dim, M)
    assert np.allclose(np.linalg.norm(noise, axis=1), np.ones(dim))


from utils import sample_noise

from utils import index_sampling


def test_index_sampling_sphere():
    A = np.array([[1, 0.5], [0.5, 1]])
    mu = np.array([1, 2])
    index = "Sphere"
    n_samples = 1000
    samples = index_sampling(A, mu, index, n_samples)
    assert samples.shape == (2, n_samples)
    assert np.allclose(np.mean(samples, axis=1), mu, atol=0.1)
    assert np.allclose(np.cov(samples), A @ A.T, atol=0.1)
    samples = index_sampling(A, mu, index, 1)
    assert samples.shape == (2, 1)


def test_index_sampling_gaussian():
    A = np.array([[1, 0.5], [0.5, 1]])
    mu = np.array([1, 2])
    index = "Gaussian"
    n_samples = 1000
    samples = index_sampling(A, mu, index, n_samples)
    assert samples.shape == (2, n_samples)
    assert np.allclose(np.mean(samples, axis=1), mu, atol=0.1)
    assert np.allclose(np.cov(samples), A @ A.T, atol=0.1)
    samples = index_sampling(A, mu, index, 1)
    assert samples.shape == (2, 1)


def test_index_sampling_pmcoord():
    A = np.array([[1, 0.5], [0.5, 1]])
    mu = np.array([1, 2])
    index = "PMCoord"
    n_samples = 1000
    samples = index_sampling(A, mu, index, n_samples)
    assert samples.shape == (2, n_samples)
    assert np.allclose(np.mean(samples, axis=1), mu, atol=0.1)
    assert np.allclose(np.cov(samples), A @ A.T, atol=0.1)
    samples = index_sampling(A, mu, index, 1)
    assert samples.shape == (2, 1)


def test_index_sampling_sparse():
    A = np.array([[1, 0.5], [0.5, 1]])
    mu = np.array([1, 2])
    index = "Sparse"
    n_samples = 1000
    samples = index_sampling(A, mu, index, n_samples)
    assert samples.shape == (2, n_samples)
    assert np.allclose(np.mean(samples, axis=1), mu, atol=0.1)
    assert np.allclose(np.cov(samples), A @ A.T, atol=0.1)
    samples = index_sampling(A, mu, index, 1)
    assert samples.shape == (2, 1)


def test_index_sampling_sparse_consistent():
    A = np.diag([1, 2, 3, 4, 3, 2, 1])
    mu = np.array([1, 2, 2, 1, 0, 0, 1])
    index = "SparseConsistent"
    n_samples = 100000
    samples = index_sampling(A, mu, index, n_samples)
    assert samples.shape == (7, n_samples)
    assert np.allclose(np.mean(samples, axis=1), mu, atol=0.1)
    assert np.allclose(np.cov(samples), A @ A.T, rtol=1e-1, atol=1e-1)
    samples = index_sampling(A, mu, index, 1)
    assert samples.shape == (7, 1)


def test_index_sampling_unifcube():
    A = np.array([[1, 0.5], [0.5, 1]])
    mu = np.array([1, 2])
    index = "UnifCube"
    n_samples = 1000
    samples = index_sampling(A, mu, index, n_samples)
    assert samples.shape == (2, n_samples)
    assert np.allclose(np.mean(samples, axis=1), mu, atol=0.1)
    assert np.allclose(np.cov(samples), A @ A.T, atol=0.1)
    samples = index_sampling(A, mu, index, 1)
    assert samples.shape == (2, 1)


from numpy.testing import assert_allclose
from utils import posterior_sampling


def test_posterior_sampling_mean():
    # Define input parameters
    mu = np.array([1, 2])
    sigma = np.array([[1, 0.5], [0.5, 1]])
    n_samples = 1000

    # Call function to generate samples
    samples = posterior_sampling(mu, sigma, n_samples)

    # Check output mean
    assert_allclose(np.mean(samples, axis=1), mu, rtol=1e-1)


def test_posterior_sampling_covariance():
    # Define input parameters
    mu = np.array([1, 2])
    sigma = np.array([[1, 0.5], [0.5, 1]])
    n_samples = 1000

    # Call function to generate samples
    samples = posterior_sampling(mu, sigma, n_samples)

    # Check output covariance
    assert_allclose(np.cov(samples), sigma, rtol=1e-1)


def test_posterior_sampling_shape():
    # Define input parameters
    mu = np.array([1, 2])
    sigma = np.array([[1, 0.5], [0.5, 1]])
    n_samples = 1000

    # Call function to generate samples
    samples = posterior_sampling(mu, sigma, n_samples)

    # Check output shape
    assert samples.shape == (2, n_samples)


def test_posterior_sampling_singular():
    # Define input parameters
    mu = np.array([1, 2])
    sigma = np.array([[1, 1], [1, 1]])
    n_samples = 1000

    # Call function to generate samples
    samples = posterior_sampling(mu, sigma, n_samples)

    # Check output mean
    assert_allclose(np.mean(samples, axis=1), mu, rtol=1e-1)

    # Check output covariance
    assert_allclose(np.cov(samples), sigma, rtol=1e-1)


def test_posterior_sampling_non_positive_definite():
    # Define input parameters
    mu = np.array([1, 2])
    sigma = np.array([[1, -1], [-1, 1]])
    n_samples = 1000

    # Call function to generate samples
    samples = posterior_sampling(mu, sigma, n_samples)

    # Check output mean
    assert_allclose(np.mean(samples, axis=1), mu, rtol=1e-1)

    # Check output covariance
    assert_allclose(np.cov(samples), sigma, rtol=1e-1)


def test_posterior_sampling_large():
    # Define input parameters
    mu = np.array([1, 2, 3, 4, 5])
    sigma = np.diag([1, 2, 3, 4, 5])
    n_samples = 10000

    # Call function to generate samples
    samples = posterior_sampling(mu, sigma, n_samples)

    # Check output mean
    assert_allclose(np.mean(samples, axis=1), mu, rtol=1e-1)

    # Check output covariance
    assert_allclose(np.cov(samples), sigma, rtol=1e-1, atol=1e-1)


def test_posterior_sampling_zero_covariance():
    # Define input parameters
    mu = np.array([1, 2])
    sigma = np.array([[1, 0], [0, 0]])
    n_samples = 1000

    # Call function to generate samples
    samples = posterior_sampling(mu, sigma, n_samples)

    # Check output mean
    assert_allclose(np.mean(samples, axis=1), mu, rtol=1e-1)

    # Check output covariance
    assert_allclose(np.cov(samples), sigma, rtol=1e-1)
