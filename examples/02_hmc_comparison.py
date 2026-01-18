#!/usr/bin/env python3
"""
Example 2: Comparing HMC vs Metropolis-Hastings

This example demonstrates the efficiency gains of Hamiltonian Monte Carlo (HMC)
compared to Metropolis-Hastings on a simple normal model.

HMC uses gradient information to efficiently explore the posterior, resulting in:
- Higher acceptance rates
- Better exploration (higher effective sample size)
- Faster convergence

Key differences:
- Metropolis: Random walk, ~30% acceptance
- HMC: Gradient-guided, ~65-80% acceptance
"""

import numpy as np
import mlx.core as mx
from mlx_mcmc import Normal, HalfNormal, MCMC

# Generate synthetic data
np.random.seed(42)
true_mu = 5.0
true_sigma = 2.0
n_obs = 100

data = np.random.normal(true_mu, true_sigma, n_obs)

print("="*70)
print("Example 2: HMC vs Metropolis Comparison")
print("="*70)
print("\nGenerating synthetic data...")
print(f"  True μ: {true_mu}")
print(f"  True σ: {true_sigma}")
print(f"  Sample size: {n_obs}")


# Define the model
def log_prob(params):
    """Log probability for normal model with unknown mean and std."""
    mu = params['mu']
    sigma = params['sigma']

    # Priors
    log_prior_mu = Normal(0, 10).log_prob(mu)
    log_prior_sigma = HalfNormal(5).log_prob(sigma)

    # Likelihood: data ~ Normal(mu, sigma)
    log_likelihood = mx.sum(Normal(mu, sigma).log_prob(mx.array(data)))

    return log_prior_mu + log_prior_sigma + log_likelihood


# Initial parameters
initial_params = {'mu': 0.0, 'sigma': 1.0}

# ============================================================
# Run Metropolis-Hastings
# ============================================================
print("\n" + "="*70)
print("1. METROPOLIS-HASTINGS SAMPLING")
print("="*70)

mcmc_mh = MCMC(log_prob)
samples_mh = mcmc_mh.run(
    initial_params,
    num_samples=5000,
    num_warmup=1000,
    method='metropolis',
    proposal_scale=0.1,
    random_seed=42
)

print("\nMetropolis-Hastings Results:")
mcmc_mh.print_summary()
print(f"Acceptance rate: {mcmc_mh.acceptance_rate:.2%}")

# ============================================================
# Run HMC
# ============================================================
print("\n" + "="*70)
print("2. HAMILTONIAN MONTE CARLO (HMC) SAMPLING")
print("="*70)

mcmc_hmc = MCMC(log_prob)
samples_hmc = mcmc_hmc.run(
    initial_params,
    num_samples=5000,
    num_warmup=1000,
    method='hmc',
    step_size=0.1,
    num_leapfrog_steps=10,
    adapt_step_size=True,
    target_accept=0.8,
    random_seed=42
)

print("\nHMC Results:")
mcmc_hmc.print_summary()
print(f"Acceptance rate: {mcmc_hmc.acceptance_rate:.2%}")

# ============================================================
# Comparison
# ============================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

# Compute effective sample sizes (ESS)
def compute_ess(samples):
    """Compute effective sample size using autocorrelation."""
    n = len(samples)
    mean = np.mean(samples)
    c0 = np.mean((samples - mean) ** 2)

    # Compute autocorrelations
    acf = []
    for lag in range(1, min(n // 2, 100)):
        c_lag = np.mean((samples[:-lag] - mean) * (samples[lag:] - mean))
        acf.append(c_lag / c0)
        # Stop when autocorrelation becomes insignificant
        if len(acf) > 1 and acf[-1] < 0.05:
            break

    # ESS formula
    ess = n / (1 + 2 * np.sum(acf))
    return ess

ess_mh_mu = compute_ess(samples_mh['mu'])
ess_mh_sigma = compute_ess(samples_mh['sigma'])
ess_hmc_mu = compute_ess(samples_hmc['mu'])
ess_hmc_sigma = compute_ess(samples_hmc['sigma'])

print("\nEffective Sample Size (ESS):")
print(f"  Metropolis μ:    {ess_mh_mu:,.0f} / 5000 ({ess_mh_mu/5000:.1%})")
print(f"  Metropolis σ:    {ess_mh_sigma:,.0f} / 5000 ({ess_mh_sigma/5000:.1%})")
print(f"  HMC μ:           {ess_hmc_mu:,.0f} / 5000 ({ess_hmc_mu/5000:.1%})")
print(f"  HMC σ:           {ess_hmc_sigma:,.0f} / 5000 ({ess_hmc_sigma/5000:.1%})")

print("\nEfficiency gain:")
print(f"  μ: {ess_hmc_mu / ess_mh_mu:.2f}x")
print(f"  σ: {ess_hmc_sigma / ess_mh_sigma:.2f}x")

print("\nParameter estimation comparison:")
mh_mu_mean = np.mean(samples_mh['mu'])
mh_sigma_mean = np.mean(samples_mh['sigma'])
hmc_mu_mean = np.mean(samples_hmc['mu'])
hmc_sigma_mean = np.mean(samples_hmc['sigma'])

print(f"  True μ:          {true_mu:.3f}")
print(f"  Metropolis μ:    {mh_mu_mean:.3f} (error: {abs(mh_mu_mean - true_mu):.3f})")
print(f"  HMC μ:           {hmc_mu_mean:.3f} (error: {abs(hmc_mu_mean - true_mu):.3f})")
print()
print(f"  True σ:          {true_sigma:.3f}")
print(f"  Metropolis σ:    {mh_sigma_mean:.3f} (error: {abs(mh_sigma_mean - true_sigma):.3f})")
print(f"  HMC σ:           {hmc_sigma_mean:.3f} (error: {abs(hmc_sigma_mean - true_sigma):.3f})")

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. Acceptance Rate:
   - Metropolis: ~30% (random walk)
   - HMC: ~65-80% (gradient-guided proposals)

2. Effective Sample Size:
   - HMC typically achieves 2-5x higher ESS
   - Means fewer samples needed for same precision

3. Convergence:
   - Both methods recover true parameters accurately
   - HMC converges faster due to better exploration

4. When to use:
   - Metropolis: Simple models, discrete parameters
   - HMC: Complex continuous models, high dimensions
""")

print("="*70)
print("✅ Comparison completed successfully!")
print("="*70)
