"""
Example 1: Simple Normal Model

Estimate the mean and standard deviation of a normal distribution
from observed data using Metropolis-Hastings sampling.

Model:
    μ ~ Normal(0, 10)     # Prior on mean
    σ ~ HalfNormal(5)     # Prior on std
    y ~ Normal(μ, σ)      # Likelihood
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from mlx_mcmc import Normal, HalfNormal, MCMC


def main():
    print("\n" + "="*70)
    print("Example 1: Simple Normal Model")
    print("="*70 + "\n")

    # Generate synthetic data
    print("Generating synthetic data...")
    np.random.seed(42)
    n = 100
    true_mu = 5.0
    true_sigma = 2.0
    y_observed = np.random.normal(true_mu, true_sigma, n)

    print(f"  True μ: {true_mu}")
    print(f"  True σ: {true_sigma}")
    print(f"  Sample size: {n}\n")

    # Define log probability function
    def log_prob(params):
        mu = params['mu']
        sigma = params['sigma']

        # Priors
        log_prior_mu = Normal(0, 10).log_prob(mu)
        log_prior_sigma = HalfNormal(5).log_prob(sigma)

        # Likelihood
        log_likelihood = mx.array(0.0)
        for y in y_observed:
            log_likelihood = log_likelihood + Normal(mu, sigma).log_prob(mx.array(y))

        return log_prior_mu + log_prior_sigma + log_likelihood

    # Create MCMC sampler
    mcmc = MCMC(log_prob)

    # Run sampling
    samples = mcmc.run(
        initial_params={'mu': 0.0, 'sigma': 1.0},
        num_samples=5000,
        num_warmup=1000,
        proposal_scale=0.3
    )

    # Print summary
    mcmc.print_summary()

    # Compare to true values
    print("\nComparison to true values:")
    summary = mcmc.summary()
    print(f"  μ: true={true_mu:.3f}, estimated={summary['mu']['mean']:.3f}, "
          f"error={abs(true_mu - summary['mu']['mean']):.3f}")
    print(f"  σ: true={true_sigma:.3f}, estimated={summary['sigma']['mean']:.3f}, "
          f"error={abs(true_sigma - summary['sigma']['mean']):.3f}")

    # Visualize results
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    warmup = 0  # Already discarded in run()

    # Trace plot for μ
    axes[0, 0].plot(samples['mu'], alpha=0.7, linewidth=0.5)
    axes[0, 0].axhline(true_mu, color='red', linestyle='--',
                      linewidth=2, label='True value')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('μ')
    axes[0, 0].set_title('Trace Plot: μ')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Trace plot for σ
    axes[0, 1].plot(samples['sigma'], alpha=0.7, linewidth=0.5)
    axes[0, 1].axhline(true_sigma, color='red', linestyle='--',
                      linewidth=2, label='True value')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('σ')
    axes[0, 1].set_title('Trace Plot: σ')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Posterior histogram for μ
    axes[1, 0].hist(samples['mu'], bins=50, alpha=0.7, density=True,
                   edgecolor='black', linewidth=0.5)
    axes[1, 0].axvline(true_mu, color='red', linestyle='--',
                      linewidth=2, label='True value')
    axes[1, 0].axvline(np.mean(samples['mu']), color='blue', linestyle='-',
                      linewidth=2, label='Posterior mean')
    axes[1, 0].set_xlabel('μ')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Posterior Distribution: μ')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Posterior histogram for σ
    axes[1, 1].hist(samples['sigma'], bins=50, alpha=0.7, density=True,
                   edgecolor='black', linewidth=0.5)
    axes[1, 1].axvline(true_sigma, color='red', linestyle='--',
                      linewidth=2, label='True value')
    axes[1, 1].axvline(np.mean(samples['sigma']), color='blue', linestyle='-',
                      linewidth=2, label='Posterior mean')
    axes[1, 1].set_xlabel('σ')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Posterior Distribution: σ')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle('MLX-MCMC: Simple Normal Model', fontsize=16, y=0.995)
    plt.tight_layout()

    output_file = '01_simple_normal_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")

    print("\n" + "="*70)
    print("✅ Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
