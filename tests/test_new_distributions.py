"""Tests for new probability distributions (Beta, Gamma, Exponential, Categorical)."""

import pytest
import mlx.core as mx
import numpy as np
from mlx_mcmc.distributions import Beta, Gamma, Exponential, Categorical


class TestBeta:
    """Tests for Beta distribution."""

    def test_init(self):
        """Test initialization."""
        dist = Beta(2, 5)
        assert float(dist.alpha) == 2.0
        assert float(dist.beta) == 5.0

    def test_log_prob_inside_support(self):
        """Test log probability for values in (0, 1)."""
        dist = Beta(2, 2)
        log_p = dist.log_prob(mx.array(0.5))
        # Log probability should be finite (PDF can be > 1, so log can be positive)
        assert not np.isinf(float(log_p))
        assert not np.isnan(float(log_p))

    def test_log_prob_outside_support(self):
        """Test log probability returns -inf outside (0, 1)."""
        dist = Beta(2, 2)
        assert float(dist.log_prob(mx.array(-0.1))) == -np.inf
        assert float(dist.log_prob(mx.array(1.5))) == -np.inf

    def test_log_prob_boundaries(self):
        """Test log probability at boundaries."""
        dist = Beta(2, 2)
        # At exact 0 and 1, log should be -inf due to log(0) or log(1-1)
        assert float(dist.log_prob(mx.array(0.0))) == -np.inf
        assert float(dist.log_prob(mx.array(1.0))) == -np.inf

    def test_sample_in_support(self):
        """Test all samples are in (0, 1)."""
        dist = Beta(2, 5)
        samples = dist.sample(mx.random.key(0), shape=(1000,))
        assert mx.all(samples > 0)
        assert mx.all(samples < 1)

    def test_sample_statistics(self):
        """Test sample statistics match theoretical values."""
        dist = Beta(5, 2)
        samples = dist.sample(mx.random.key(42), shape=(10000,))

        # Mean = alpha / (alpha + beta)
        expected_mean = 5.0 / (5.0 + 2.0)
        sample_mean = float(mx.mean(samples))
        assert np.isclose(sample_mean, expected_mean, atol=0.05)

        # Variance = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        expected_var = (5.0 * 2.0) / ((7.0 ** 2) * 8.0)
        sample_var = float(mx.var(samples))
        assert np.isclose(sample_var, expected_var, atol=0.01)

    def test_mean(self):
        """Test mean calculation."""
        dist = Beta(3, 7)
        expected = 3.0 / 10.0
        assert np.isclose(float(dist.mean()), expected)

    def test_uniform_case(self):
        """Test Beta(1, 1) is uniform on [0, 1]."""
        dist = Beta(1, 1)
        samples = dist.sample(mx.random.key(123), shape=(10000,))
        # Should be approximately uniform
        assert 0.4 < float(mx.mean(samples)) < 0.6


class TestGamma:
    """Tests for Gamma distribution."""

    def test_init(self):
        """Test initialization."""
        dist = Gamma(2, 3)
        assert float(dist.alpha) == 2.0
        assert float(dist.beta) == 3.0

    def test_init_default_beta(self):
        """Test initialization with default beta=1."""
        dist = Gamma(2)
        assert float(dist.beta) == 1.0

    def test_log_prob_positive(self):
        """Test log probability for positive values."""
        dist = Gamma(2, 1)
        log_p = dist.log_prob(mx.array(1.5))
        assert float(log_p) < 0
        assert not np.isinf(float(log_p))

    def test_log_prob_negative(self):
        """Test log probability returns -inf for negative values."""
        dist = Gamma(2, 1)
        assert float(dist.log_prob(mx.array(-1.0))) == -np.inf

    def test_sample_all_positive(self):
        """Test all samples are positive."""
        dist = Gamma(3, 2)
        samples = dist.sample(mx.random.key(0), shape=(1000,))
        assert mx.all(samples > 0)

    def test_sample_statistics(self):
        """Test sample statistics match theoretical values."""
        dist = Gamma(4, 2)
        samples = dist.sample(mx.random.key(42), shape=(10000,))

        # Mean = alpha / beta
        expected_mean = 4.0 / 2.0
        sample_mean = float(mx.mean(samples))
        assert np.isclose(sample_mean, expected_mean, rtol=0.1)

        # Variance = alpha / beta^2
        expected_var = 4.0 / (2.0 ** 2)
        sample_var = float(mx.var(samples))
        assert np.isclose(sample_var, expected_var, rtol=0.15)

    def test_mean(self):
        """Test mean calculation."""
        dist = Gamma(6, 3)
        expected = 6.0 / 3.0
        assert np.isclose(float(dist.mean()), expected)


class TestExponential:
    """Tests for Exponential distribution."""

    def test_init(self):
        """Test initialization."""
        dist = Exponential(2.0)
        assert float(dist.rate) == 2.0

    def test_log_prob_positive(self):
        """Test log probability for positive values."""
        dist = Exponential(1.0)
        log_p = dist.log_prob(mx.array(0.5))
        assert float(log_p) < 0
        assert not np.isinf(float(log_p))

    def test_log_prob_negative(self):
        """Test log probability returns -inf for negative values."""
        dist = Exponential(1.0)
        assert float(dist.log_prob(mx.array(-1.0))) == -np.inf

    def test_log_prob_zero(self):
        """Test log probability at zero."""
        dist = Exponential(2.0)
        log_p = dist.log_prob(mx.array(0.0))
        # Should be log(rate) = log(2)
        assert np.isclose(float(log_p), np.log(2.0))

    def test_sample_all_positive(self):
        """Test all samples are positive."""
        dist = Exponential(1.5)
        samples = dist.sample(mx.random.key(0), shape=(1000,))
        assert mx.all(samples >= 0)

    def test_sample_statistics(self):
        """Test sample statistics match theoretical values."""
        dist = Exponential(2.0)
        samples = dist.sample(mx.random.key(42), shape=(10000,))

        # Mean = 1 / rate
        expected_mean = 1.0 / 2.0
        sample_mean = float(mx.mean(samples))
        assert np.isclose(sample_mean, expected_mean, rtol=0.1)

        # Variance = 1 / rate^2
        expected_var = 1.0 / (2.0 ** 2)
        sample_var = float(mx.var(samples))
        assert np.isclose(sample_var, expected_var, rtol=0.15)

    def test_mean(self):
        """Test mean calculation."""
        dist = Exponential(3.0)
        expected = 1.0 / 3.0
        assert np.isclose(float(dist.mean()), expected)


class TestCategorical:
    """Tests for Categorical distribution."""

    def test_init_with_probs(self):
        """Test initialization with probabilities."""
        probs = [0.2, 0.3, 0.5]
        dist = Categorical(probs=probs)
        assert dist.num_categories == 3
        # Should normalize to sum to 1
        assert np.isclose(float(mx.sum(dist.probs)), 1.0)

    def test_init_with_logits(self):
        """Test initialization with logits."""
        logits = [1.0, 2.0, 3.0]
        dist = Categorical(logits=logits)
        assert dist.num_categories == 3
        # Probs should sum to 1 after softmax
        assert np.isclose(float(mx.sum(dist.probs)), 1.0)

    def test_init_error(self):
        """Test error when neither probs nor logits specified."""
        with pytest.raises(ValueError):
            Categorical()

    def test_init_error_both(self):
        """Test error when both probs and logits specified."""
        with pytest.raises(ValueError):
            Categorical(probs=[0.5, 0.5], logits=[1.0, 1.0])

    def test_log_prob_valid(self):
        """Test log probability for valid category indices."""
        dist = Categorical(probs=[0.2, 0.5, 0.3])
        log_p_0 = dist.log_prob(mx.array(0))
        log_p_1 = dist.log_prob(mx.array(1))

        assert np.isclose(float(log_p_0), np.log(0.2), rtol=0.01)
        assert np.isclose(float(log_p_1), np.log(0.5), rtol=0.01)

    def test_log_prob_invalid(self):
        """Test log probability returns -inf for invalid indices."""
        dist = Categorical(probs=[0.2, 0.5, 0.3])
        assert float(dist.log_prob(mx.array(-1))) == -np.inf
        assert float(dist.log_prob(mx.array(3))) == -np.inf

    def test_sample_valid_categories(self):
        """Test samples are valid category indices."""
        dist = Categorical(probs=[0.2, 0.5, 0.3])
        samples = dist.sample(mx.random.key(0), shape=(1000,))
        assert mx.all(samples >= 0)
        assert mx.all(samples < 3)

    def test_sample_distribution(self):
        """Test sample distribution matches probabilities."""
        dist = Categorical(probs=[0.1, 0.6, 0.3])
        samples = dist.sample(mx.random.key(42), shape=(10000,))

        # Count occurrences
        counts = [float(mx.sum(samples == i)) for i in range(3)]
        empirical_probs = [c / 10000 for c in counts]

        assert np.isclose(empirical_probs[0], 0.1, atol=0.02)
        assert np.isclose(empirical_probs[1], 0.6, atol=0.02)
        assert np.isclose(empirical_probs[2], 0.3, atol=0.02)

    def test_uniform_categorical(self):
        """Test uniform categorical distribution."""
        dist = Categorical(probs=[1/3, 1/3, 1/3])
        samples = dist.sample(mx.random.key(123), shape=(9000,))

        counts = [float(mx.sum(samples == i)) for i in range(3)]
        # Each category should appear ~3000 times
        for count in counts:
            assert 2700 < count < 3300

    def test_mode(self):
        """Test mode returns category with highest probability."""
        dist = Categorical(probs=[0.1, 0.7, 0.2])
        assert int(dist.mode()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
