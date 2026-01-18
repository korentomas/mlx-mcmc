"""Tests for probability distributions."""

import pytest
import mlx.core as mx
import numpy as np
from mlx_mcmc.distributions import Normal, HalfNormal


class TestNormal:
    """Tests for Normal distribution."""

    def test_init(self):
        """Test initialization."""
        dist = Normal(0, 1)
        assert float(dist.loc) == 0.0
        assert float(dist.scale) == 1.0

    def test_log_prob_at_mean(self):
        """Test log probability at mean."""
        dist = Normal(0, 1)
        log_p = dist.log_prob(mx.array(0.0))

        # For standard normal at mean: log(1/sqrt(2π)) = -0.5 * log(2π)
        expected = -0.5 * np.log(2 * np.pi)
        assert np.isclose(float(log_p), expected, rtol=1e-5)

    def test_log_prob_symmetry(self):
        """Test symmetry around mean."""
        dist = Normal(0, 1)
        log_p_plus = dist.log_prob(mx.array(1.0))
        log_p_minus = dist.log_prob(mx.array(-1.0))
        assert np.isclose(float(log_p_plus), float(log_p_minus), rtol=1e-5)

    def test_sample_shape(self):
        """Test sampling produces correct shape."""
        dist = Normal(0, 1)
        samples = dist.sample(mx.random.key(0), shape=(100,))
        assert samples.shape == (100,)

    def test_sample_statistics(self):
        """Test sample statistics match distribution parameters."""
        dist = Normal(5.0, 2.0)
        samples = dist.sample(mx.random.key(42), shape=(10000,))

        # Check mean and std (should be close)
        sample_mean = float(mx.mean(samples))
        sample_std = float(mx.std(samples))

        assert np.isclose(sample_mean, 5.0, atol=0.1)
        assert np.isclose(sample_std, 2.0, atol=0.1)


class TestHalfNormal:
    """Tests for HalfNormal distribution."""

    def test_init(self):
        """Test initialization."""
        dist = HalfNormal(5.0)
        assert float(dist.scale) == 5.0

    def test_log_prob_positive(self):
        """Test log probability for positive values."""
        dist = HalfNormal(1.0)
        log_p = dist.log_prob(mx.array(0.5))
        assert float(log_p) < 0  # Log probability should be negative

    def test_log_prob_negative(self):
        """Test log probability for negative values is -inf."""
        dist = HalfNormal(1.0)
        log_p = dist.log_prob(mx.array(-1.0))
        assert float(log_p) == -np.inf

    def test_log_prob_zero(self):
        """Test log probability at zero."""
        dist = HalfNormal(1.0)
        log_p = dist.log_prob(mx.array(0.0))
        # At zero: log(2/sqrt(2π))
        expected = np.log(2.0) - 0.5 * np.log(2 * np.pi)
        assert np.isclose(float(log_p), expected, rtol=1e-5)

    def test_sample_all_positive(self):
        """Test all samples are positive."""
        dist = HalfNormal(5.0)
        samples = dist.sample(mx.random.key(0), shape=(1000,))
        assert mx.all(samples >= 0)

    def test_sample_statistics(self):
        """Test sample statistics match expected values."""
        dist = HalfNormal(2.0)
        samples = dist.sample(mx.random.key(42), shape=(10000,))

        # For half-normal with scale σ:
        # Mean = σ * sqrt(2/π)
        # Std = σ * sqrt(1 - 2/π)
        expected_mean = 2.0 * np.sqrt(2 / np.pi)
        expected_std = 2.0 * np.sqrt(1 - 2 / np.pi)

        sample_mean = float(mx.mean(samples))
        sample_std = float(mx.std(samples))

        assert np.isclose(sample_mean, expected_mean, rtol=0.1)
        assert np.isclose(sample_std, expected_std, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
