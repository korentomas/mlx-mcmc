"""
Example 3: Bayesian A/B Testing

Compare conversion rates between two versions (A and B) using Beta-Binomial model
with Metropolis-Hastings sampling. Demonstrates Beta and Categorical distributions.

Model:
    p_A ~ Beta(1, 1)           # Prior on conversion rate A (uniform)
    p_B ~ Beta(1, 1)           # Prior on conversion rate B (uniform)
    conversions_A ~ Binomial(n_A, p_A)
    conversions_B ~ Binomial(n_B, p_B)

We estimate P(p_B > p_A) - the probability that B is better than A.
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from mlx_mcmc import Beta, Categorical, MCMC


def main():
    print("\n" + "="*70)
    print("Example 3: Bayesian A/B Testing")
    print("="*70 + "\n")

    # Simulate A/B test data
    print("Simulating A/B test data...")
    np.random.seed(42)

    # Version A: 1000 visitors, 12% conversion rate
    n_A = 1000
    true_p_A = 0.12
    conversions_A = np.random.binomial(n_A, true_p_A)

    # Version B: 1000 visitors, 15% conversion rate
    n_B = 1000
    true_p_B = 0.15
    conversions_B = np.random.binomial(n_B, true_p_B)

    print(f"  Version A: {conversions_A}/{n_A} conversions ({100*conversions_A/n_A:.2f}%)")
    print(f"  Version B: {conversions_B}/{n_B} conversions ({100*conversions_B/n_B:.2f}%)")
    print(f"  True rates: A={100*true_p_A:.1f}%, B={100*true_p_B:.1f}%\n")

    # Define log probability function
    def log_prob(params):
        p_A = params['p_A']
        p_B = params['p_B']

        # Priors: Beta(1,1) = Uniform(0,1)
        log_prior_A = Beta(1, 1).log_prob(p_A)
        log_prior_B = Beta(1, 1).log_prob(p_B)

        # Likelihood: Binomial(n, p) ∝ p^k * (1-p)^(n-k)
        # This is proportional to Beta(k+1, n-k+1)
        log_likelihood_A = Beta(conversions_A + 1, n_A - conversions_A + 1).log_prob(p_A)
        log_likelihood_B = Beta(conversions_B + 1, n_B - conversions_B + 1).log_prob(p_B)

        return log_prior_A + log_prior_B + log_likelihood_A + log_likelihood_B

    # Create MCMC sampler
    mcmc = MCMC(log_prob)

    # Run sampling
    print("Running MCMC sampling...")
    samples = mcmc.run(
        initial_params={'p_A': 0.1, 'p_B': 0.1},
        num_samples=5000,
        num_warmup=1000,
        proposal_scale=0.02  # Small scale for bounded [0,1] parameters
    )

    # Print summary
    mcmc.print_summary()

    # Compute probability that B > A
    p_B_better = np.mean(samples['p_B'] > samples['p_A'])
    print(f"\nP(p_B > p_A) = {p_B_better:.3f}")

    if p_B_better > 0.95:
        print("✅ Strong evidence that B is better than A")
    elif p_B_better > 0.90:
        print("⚠️  Moderate evidence that B is better than A")
    else:
        print("❌ Insufficient evidence to conclude B is better")

    # Expected lift
    expected_lift = np.mean((samples['p_B'] - samples['p_A']) / samples['p_A']) * 100
    print(f"Expected relative lift: {expected_lift:.1f}%")

    # Visualize results
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Posterior distributions
    axes[0, 0].hist(samples['p_A'], bins=50, alpha=0.6, label='Version A',
                    density=True, color='blue', edgecolor='black', linewidth=0.5)
    axes[0, 0].hist(samples['p_B'], bins=50, alpha=0.6, label='Version B',
                    density=True, color='orange', edgecolor='black', linewidth=0.5)
    axes[0, 0].axvline(true_p_A, color='blue', linestyle='--', linewidth=2, label='True A')
    axes[0, 0].axvline(true_p_B, color='orange', linestyle='--', linewidth=2, label='True B')
    axes[0, 0].set_xlabel('Conversion Rate')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Posterior Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Trace plots
    axes[0, 1].plot(samples['p_A'], alpha=0.5, linewidth=0.5, label='Version A')
    axes[0, 1].plot(samples['p_B'], alpha=0.5, linewidth=0.5, label='Version B')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Conversion Rate')
    axes[0, 1].set_title('Trace Plots')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Difference distribution
    difference = samples['p_B'] - samples['p_A']
    axes[1, 0].hist(difference, bins=50, alpha=0.7, density=True,
                   color='green', edgecolor='black', linewidth=0.5)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[1, 0].axvline(np.mean(difference), color='blue', linestyle='-',
                      linewidth=2, label='Mean difference')
    axes[1, 0].set_xlabel('Difference (p_B - p_A)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title(f'Difference Distribution\nP(p_B > p_A) = {p_B_better:.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Relative lift distribution
    relative_lift = (samples['p_B'] - samples['p_A']) / samples['p_A'] * 100
    axes[1, 1].hist(relative_lift, bins=50, alpha=0.7, density=True,
                   color='purple', edgecolor='black', linewidth=0.5)
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='No lift')
    axes[1, 1].axvline(expected_lift, color='blue', linestyle='-',
                      linewidth=2, label='Expected lift')
    axes[1, 1].set_xlabel('Relative Lift (%)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title(f'Relative Lift Distribution\nExpected: {expected_lift:.1f}%')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle('MLX-MCMC: Bayesian A/B Testing', fontsize=16, y=0.995)
    plt.tight_layout()

    output_file = '03_ab_testing_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")

    print("\n" + "="*70)
    print("✅ Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
