"""Tests for Hamiltonian Monte Carlo (HMC) sampler."""

import pytest
import mlx.core as mx
import numpy as np
from mlx_mcmc.kernels.hmc import hmc
from mlx_mcmc.distributions import Normal, HalfNormal


class TestHMC:
    """Tests for HMC sampler."""

    def test_simple_normal(self):
        """Test HMC on standard normal distribution."""

        def log_prob(params):
            return Normal(0, 1).log_prob(params['x'])

        initial = {'x': 0.5}
        samples, accept_rate = hmc(
            log_prob,
            initial,
            num_samples=1000,
            num_warmup=500,
            step_size=0.1,
            num_leapfrog_steps=10,
            key=mx.random.key(42),
        )

        # Check shape
        assert samples['x'].shape == (1000,)

        # Check acceptance rate is reasonable
        assert 0.5 <= accept_rate <= 1.0

        # Check sample statistics (should be close to N(0,1))
        sample_mean = float(mx.mean(samples['x']))
        sample_std = float(mx.std(samples['x']))

        assert np.isclose(sample_mean, 0.0, atol=0.15)
        assert np.isclose(sample_std, 1.0, atol=0.15)

    def test_multivariate_normal(self):
        """Test HMC on multivariate normal model."""

        def log_prob(params):
            lp = Normal(0, 1).log_prob(params['x'])
            lp += Normal(2, 0.5).log_prob(params['y'])
            return lp

        initial = {'x': 0.0, 'y': 2.0}
        samples, accept_rate = hmc(
            log_prob,
            initial,
            num_samples=2000,
            num_warmup=1000,
            step_size=0.1,
            num_leapfrog_steps=10,
            key=mx.random.key(123),
        )

        # Check shapes
        assert samples['x'].shape == (2000,)
        assert samples['y'].shape == (2000,)

        # Check acceptance rate
        assert accept_rate > 0.5

        # Check x ~ N(0, 1)
        x_mean = float(mx.mean(samples['x']))
        x_std = float(mx.std(samples['x']))
        assert np.isclose(x_mean, 0.0, atol=0.15)
        assert np.isclose(x_std, 1.0, atol=0.15)

        # Check y ~ N(2, 0.5)
        y_mean = float(mx.mean(samples['y']))
        y_std = float(mx.std(samples['y']))
        assert np.isclose(y_mean, 2.0, atol=0.15)
        assert np.isclose(y_std, 0.5, atol=0.1)

    def test_step_size_adaptation(self):
        """Test that step size adaptation works."""

        def log_prob(params):
            return Normal(0, 1).log_prob(params['x'])

        initial = {'x': 0.0}

        # With adaptation
        samples_adapt, accept_adapt = hmc(
            log_prob,
            initial,
            num_samples=500,
            num_warmup=500,
            step_size=0.01,  # Start very small
            num_leapfrog_steps=10,
            adapt_step_size=True,
            target_accept=0.7,
            key=mx.random.key(42),
        )

        # Without adaptation (same initial step size)
        samples_no_adapt, accept_no_adapt = hmc(
            log_prob,
            initial,
            num_samples=500,
            num_warmup=500,
            step_size=0.01,
            num_leapfrog_steps=10,
            adapt_step_size=False,
            key=mx.random.key(42),
        )

        # With adaptation, acceptance should be closer to target
        # Without adaptation, small step size -> very high acceptance
        assert accept_no_adapt > accept_adapt

    def test_constrained_parameter(self):
        """Test HMC with constrained parameter (half-normal).

        Note: HMC can struggle with hard constraints since it may propose
        moves that violate constraints. This test mainly checks that samples
        remain valid (positive) even if the distribution isn't perfectly recovered.
        """

        def log_prob(params):
            # sigma must be positive
            return HalfNormal(2.0).log_prob(params['sigma'])

        initial = {'sigma': 1.0}
        samples, accept_rate = hmc(
            log_prob,
            initial,
            num_samples=1000,
            num_warmup=500,
            step_size=0.05,
            num_leapfrog_steps=10,
            key=mx.random.key(999),
        )

        # All samples should be positive (main constraint check)
        assert mx.all(samples['sigma'] > 0)

        # Check that we're sampling in a reasonable range
        sample_mean = float(mx.mean(samples['sigma']))
        assert 0.5 < sample_mean < 3.0  # Reasonable range for HalfNormal(2.0)

    def test_reproducibility(self):
        """Test that same seed gives same results."""

        def log_prob(params):
            return Normal(0, 1).log_prob(params['x'])

        initial = {'x': 0.0}

        samples1, _ = hmc(
            log_prob,
            initial,
            num_samples=100,
            num_warmup=50,
            step_size=0.1,
            num_leapfrog_steps=5,
            key=mx.random.key(12345),
        )

        samples2, _ = hmc(
            log_prob,
            initial,
            num_samples=100,
            num_warmup=50,
            step_size=0.1,
            num_leapfrog_steps=5,
            key=mx.random.key(12345),
        )

        # Same seed should give same samples
        assert mx.allclose(samples1['x'], samples2['x'])

    def test_posterior_inference(self):
        """Test HMC on realistic inference problem."""
        # Generate data
        np.random.seed(42)
        true_mu = 3.0
        true_sigma = 1.5
        data = np.random.normal(true_mu, true_sigma, 50)

        def log_prob(params):
            mu = params['mu']
            sigma = params['sigma']

            # Priors
            lp = Normal(0, 10).log_prob(mu)
            lp += HalfNormal(5).log_prob(sigma)

            # Likelihood
            lp += mx.sum(Normal(mu, sigma).log_prob(mx.array(data)))

            return lp

        initial = {'mu': 0.0, 'sigma': 1.0}
        samples, accept_rate = hmc(
            log_prob,
            initial,
            num_samples=2000,
            num_warmup=1000,
            step_size=0.1,
            num_leapfrog_steps=10,
            adapt_step_size=True,
            key=mx.random.key(42),
        )

        # Check we recovered approximately correct parameters
        mu_mean = float(mx.mean(samples['mu']))
        sigma_mean = float(mx.mean(samples['sigma']))

        assert np.isclose(mu_mean, true_mu, atol=0.4)
        assert np.isclose(sigma_mean, true_sigma, atol=0.4)

        # Check acceptance rate is reasonable
        assert 0.5 <= accept_rate <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
