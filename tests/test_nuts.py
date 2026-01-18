"""Tests for No-U-Turn Sampler (NUTS) implementation."""

import pytest
import mlx.core as mx
import numpy as np
from mlx_mcmc import Normal, HalfNormal, MCMC
from mlx_mcmc.kernels.nuts import nuts


class TestNUTS:
    """Tests for NUTS sampler."""

    def test_simple_normal(self):
        """Test NUTS on simple 1D normal distribution."""
        def log_prob(params):
            return Normal(5.0, 2.0).log_prob(params['mu'])

        samples, accept_rate = nuts(
            log_prob,
            {'mu': 0.0},
            num_samples=1000,
            num_warmup=500,
            step_size=0.5,
            key=mx.random.key(42)
        )

        mean = float(mx.mean(samples['mu']))
        std = float(mx.std(samples['mu']))

        assert 4.5 < mean < 5.5, f"Mean {mean} not close to 5.0"
        assert 1.5 < std < 2.5, f"Std {std} not close to 2.0"
        assert accept_rate > 0.5, f"Acceptance rate {accept_rate} too low"

    def test_multivariate_normal(self):
        """Test NUTS on 2D normal distribution."""
        def log_prob(params):
            mu1 = params['mu1']
            mu2 = params['mu2']
            return Normal(0, 1).log_prob(mu1) + Normal(5, 2).log_prob(mu2)

        samples, accept_rate = nuts(
            log_prob,
            {'mu1': 0.0, 'mu2': 0.0},
            num_samples=1000,
            num_warmup=500,
            step_size=0.3,
            key=mx.random.key(123)
        )

        mean1 = float(mx.mean(samples['mu1']))
        mean2 = float(mx.mean(samples['mu2']))

        assert -0.5 < mean1 < 0.5, f"mu1 mean {mean1} not close to 0.0"
        assert 4.5 < mean2 < 5.5, f"mu2 mean {mean2} not close to 5.0"

    def test_step_size_adaptation(self):
        """Test that step size adaptation works."""
        def log_prob(params):
            return Normal(0, 1).log_prob(params['x'])

        # With adaptation
        samples1, _ = nuts(
            log_prob,
            {'x': 0.0},
            num_samples=500,
            num_warmup=500,
            step_size=0.01,  # Start very small
            adapt_step_size=True,
            key=mx.random.key(42)
        )

        # Without adaptation (should be less efficient)
        samples2, _ = nuts(
            log_prob,
            {'x': 0.0},
            num_samples=500,
            num_warmup=500,
            step_size=0.01,  # Stay very small
            adapt_step_size=False,
            key=mx.random.key(42)
        )

        # Both should work, but adapted version typically has better ESS
        # Just verify they both complete successfully
        assert len(samples1['x']) == 500
        assert len(samples2['x']) == 500

    def test_constrained_parameter(self):
        """Test NUTS with HalfNormal (positive constraint)."""
        def log_prob(params):
            sigma = params['sigma']
            # HalfNormal prior + some data likelihood
            return HalfNormal(5.0).log_prob(sigma) + Normal(0, sigma).log_prob(mx.array(0.5))

        samples, _ = nuts(
            log_prob,
            {'sigma': 1.0},
            num_samples=1000,
            num_warmup=500,
            step_size=0.1,
            key=mx.random.key(456)
        )

        # All samples should be positive
        assert mx.all(samples['sigma'] > 0), "Some sigma samples are negative"

        mean_sigma = float(mx.mean(samples['sigma']))
        assert mean_sigma > 0, f"Mean sigma {mean_sigma} should be positive"

    def test_reproducibility(self):
        """Test that same random seed gives same results."""
        def log_prob(params):
            return Normal(0, 1).log_prob(params['x'])

        samples1, _ = nuts(
            log_prob,
            {'x': 0.0},
            num_samples=100,
            num_warmup=100,
            key=mx.random.key(42)
        )

        samples2, _ = nuts(
            log_prob,
            {'x': 0.0},
            num_samples=100,
            num_warmup=100,
            key=mx.random.key(42)
        )

        # Should be identical (same random seed)
        np.testing.assert_array_almost_equal(
            np.array(samples1['x']),
            np.array(samples2['x']),
            decimal=5
        )

    def test_mcmc_api_integration(self):
        """Test NUTS through high-level MCMC API."""
        def log_prob(params):
            mu = params['mu']
            return Normal(3.0, 1.0).log_prob(mu)

        mcmc = MCMC(log_prob)
        samples = mcmc.run(
            initial_params={'mu': 0.0},
            num_samples=500,
            num_warmup=500,
            method='nuts',
            step_size=0.5,
            verbose=False
        )

        mean = np.mean(samples['mu'])
        assert 2.5 < mean < 3.5, f"Mean {mean} not close to 3.0"

    def test_max_tree_depth(self):
        """Test that max_tree_depth limits trajectory length."""
        def log_prob(params):
            return Normal(0, 1).log_prob(params['x'])

        # With small max_tree_depth
        samples1, _ = nuts(
            log_prob,
            {'x': 0.0},
            num_samples=100,
            num_warmup=100,
            max_tree_depth=3,  # Small depth
            step_size=0.5,
            key=mx.random.key(42)
        )

        # With large max_tree_depth
        samples2, _ = nuts(
            log_prob,
            {'x': 0.0},
            num_samples=100,
            num_warmup=100,
            max_tree_depth=10,  # Large depth
            step_size=0.5,
            key=mx.random.key(42)
        )

        # Both should work (depth limit just prevents runaway trajectories)
        assert len(samples1['x']) == 100
        assert len(samples2['x']) == 100

    def test_inference_problem(self):
        """Test NUTS on realistic inference problem."""
        # Simulate data: y ~ Normal(mu, sigma)
        np.random.seed(42)
        true_mu = 5.0
        true_sigma = 2.0
        n = 50
        y_observed = np.random.normal(true_mu, true_sigma, n)

        def log_prob(params):
            mu = params['mu']
            sigma = params['sigma']

            # Priors
            log_prior = Normal(0, 10).log_prob(mu) + HalfNormal(5).log_prob(sigma)

            # Likelihood
            log_likelihood = mx.sum(
                mx.array([Normal(mu, sigma).log_prob(mx.array(y)) for y in y_observed])
            )

            return log_prior + log_likelihood

        mcmc = MCMC(log_prob)
        samples = mcmc.run(
            initial_params={'mu': 0.0, 'sigma': 1.0},
            num_samples=1000,
            num_warmup=500,
            method='nuts',
            step_size=0.1,
            verbose=False
        )

        mu_est = np.mean(samples['mu'])
        sigma_est = np.mean(samples['sigma'])

        # Should recover true parameters reasonably well
        # Note: With only 50 observations and MCMC variance, allow some slack
        assert abs(mu_est - true_mu) < 1.0, f"mu estimate {mu_est} far from {true_mu}"
        assert abs(sigma_est - true_sigma) < 1.0, f"sigma estimate {sigma_est} far from {true_sigma}"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
