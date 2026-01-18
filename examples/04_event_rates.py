"""
Example 4: Modeling Event Rates

Estimate the rate parameter of events (e.g., customer arrivals, system failures)
using Gamma-Exponential model with Metropolis-Hastings sampling.
Demonstrates Gamma and Exponential distributions.

Model:
    λ ~ Gamma(2, 1)           # Prior on rate parameter (mean=2)
    waiting_times ~ Exponential(λ)

The Exponential distribution models waiting times between events,
while Gamma is the conjugate prior for the rate parameter.
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from mlx_mcmc import Gamma, Exponential, MCMC


def main():
    print("\n" + "="*70)
    print("Example 4: Modeling Event Rates")
    print("="*70 + "\n")

    # Simulate event data (waiting times)
    print("Simulating event data...")
    np.random.seed(42)

    # True rate: 3 events per unit time
    # Mean waiting time: 1/3 time units
    true_rate = 3.0
    n_events = 50
    waiting_times = np.random.exponential(scale=1/true_rate, size=n_events)

    print(f"  True event rate: {true_rate:.2f} events/time")
    print(f"  True mean waiting time: {1/true_rate:.3f} time units")
    print(f"  Number of observed events: {n_events}")
    print(f"  Observed mean waiting time: {np.mean(waiting_times):.3f} time units\n")

    # Define log probability function
    def log_prob(params):
        rate = params['rate']

        # Prior: Gamma(2, 1) with mean=2
        # This is a weakly informative prior
        log_prior = Gamma(alpha=2, beta=1).log_prob(rate)

        # Likelihood: Product of Exponential densities
        log_likelihood = mx.array(0.0)
        for t in waiting_times:
            log_likelihood = log_likelihood + Exponential(rate).log_prob(mx.array(t))

        return log_prior + log_likelihood

    # Create MCMC sampler
    mcmc = MCMC(log_prob)

    # Run sampling
    print("Running MCMC sampling...")
    samples = mcmc.run(
        initial_params={'rate': 2.0},
        num_samples=5000,
        num_warmup=1000,
        proposal_scale=0.3
    )

    # Print summary
    mcmc.print_summary()

    # Compute derived quantities
    mean_waiting_time = 1.0 / samples['rate']
    summary = mcmc.summary()

    print("\nDerived Quantities:")
    print(f"  Mean waiting time:")
    print(f"    True: {1/true_rate:.3f}")
    print(f"    Estimated: {np.mean(mean_waiting_time):.3f}")
    print(f"    95% CI: [{np.percentile(mean_waiting_time, 2.5):.3f}, "
          f"{np.percentile(mean_waiting_time, 97.5):.3f}]")

    # Probability of high activity
    p_high_rate = np.mean(samples['rate'] > 4.0)
    print(f"\n  P(rate > 4.0) = {p_high_rate:.3f}")

    # Visualize results
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Trace plot
    axes[0, 0].plot(samples['rate'], alpha=0.7, linewidth=0.5)
    axes[0, 0].axhline(true_rate, color='red', linestyle='--',
                      linewidth=2, label='True rate')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Rate (λ)')
    axes[0, 0].set_title('Trace Plot: Event Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Posterior distribution for rate
    axes[0, 1].hist(samples['rate'], bins=50, alpha=0.7, density=True,
                   color='blue', edgecolor='black', linewidth=0.5)
    axes[0, 1].axvline(true_rate, color='red', linestyle='--',
                      linewidth=2, label='True rate')
    axes[0, 1].axvline(np.mean(samples['rate']), color='green', linestyle='-',
                      linewidth=2, label='Posterior mean')

    # Add prior for comparison
    x_range = np.linspace(0, axes[0, 1].get_xlim()[1], 200)
    prior_density = np.array([float(Gamma(2, 1).log_prob(mx.array(x))) for x in x_range])
    prior_density = np.exp(prior_density)
    axes[0, 1].plot(x_range, prior_density, 'k--', linewidth=2, label='Prior', alpha=0.5)

    axes[0, 1].set_xlabel('Rate (λ)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Posterior Distribution: Event Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Posterior distribution for mean waiting time
    axes[1, 0].hist(mean_waiting_time, bins=50, alpha=0.7, density=True,
                   color='orange', edgecolor='black', linewidth=0.5)
    axes[1, 0].axvline(1/true_rate, color='red', linestyle='--',
                      linewidth=2, label='True mean')
    axes[1, 0].axvline(np.mean(mean_waiting_time), color='green', linestyle='-',
                      linewidth=2, label='Posterior mean')
    axes[1, 0].set_xlabel('Mean Waiting Time (1/λ)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Posterior Distribution: Mean Waiting Time')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Posterior predictive: next waiting time
    axes[1, 1].hist(waiting_times, bins=30, alpha=0.5, density=True,
                   color='gray', edgecolor='black', linewidth=0.5, label='Observed data')

    # Sample from posterior predictive
    np.random.seed(123)
    n_pred = 10000
    pred_samples = []
    for _ in range(n_pred):
        rate_sample = samples['rate'][np.random.randint(len(samples['rate']))]
        pred_samples.append(np.random.exponential(1/float(rate_sample)))

    axes[1, 1].hist(pred_samples, bins=50, alpha=0.7, density=True,
                   color='green', edgecolor='black', linewidth=0.5,
                   label='Posterior predictive')

    # True distribution
    x_pred = np.linspace(0, max(pred_samples), 200)
    true_density = true_rate * np.exp(-true_rate * x_pred)
    axes[1, 1].plot(x_pred, true_density, 'r--', linewidth=2, label='True distribution')

    axes[1, 1].set_xlabel('Waiting Time')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Posterior Predictive: Next Waiting Time')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim(0, 2)

    plt.suptitle('MLX-MCMC: Event Rate Modeling (Gamma-Exponential)', fontsize=16, y=0.995)
    plt.tight_layout()

    output_file = '04_event_rates_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")

    print("\n" + "="*70)
    print("✅ Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
