"""
Example 5: Categorical Outcomes with Dirichlet Prior

Estimate probabilities for categorical outcomes (e.g., customer preferences,
product choices) using Dirichlet-Categorical model with Metropolis-Hastings.
Demonstrates Categorical and Beta distributions.

Model:
    p ~ Dirichlet(α)           # Prior on category probabilities
    outcomes ~ Categorical(p)   # Observed category choices

For K=3 categories, we use independent Beta priors for simplicity:
    p1 ~ Beta(2, 2)
    p2 ~ Beta(2, 2)
    Then p3 = 1 - p1 - p2 (constrained to simplex)
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from mlx_mcmc import Beta, Categorical, MCMC


def main():
    print("\n" + "="*70)
    print("Example 5: Categorical Outcomes (Product Choice)")
    print("="*70 + "\n")

    # Simulate categorical data
    print("Simulating customer product choices...")
    np.random.seed(42)

    # True probabilities for 3 products
    true_probs = np.array([0.5, 0.3, 0.2])
    n_customers = 300

    # Generate choices (0, 1, or 2)
    choices = np.random.choice(3, size=n_customers, p=true_probs)

    # Count observations
    counts = [np.sum(choices == i) for i in range(3)]
    observed_probs = np.array(counts) / n_customers

    print(f"  Product choices observed:")
    for i in range(3):
        print(f"    Product {i}: {counts[i]}/{n_customers} ({100*observed_probs[i]:.1f}%)")
    print(f"\n  True probabilities: {true_probs}")
    print(f"  Observed frequencies: {observed_probs}\n")

    # Define log probability function
    # We parameterize with p1, p2 and compute p3 = 1 - p1 - p2
    def log_prob(params):
        p1 = params['p1']
        p2 = params['p2']

        # Check simplex constraint
        p3 = 1.0 - p1 - p2
        if float(p3) <= 0 or float(p1) <= 0 or float(p2) <= 0:
            return mx.array(-mx.inf)

        # Priors: Beta(2,2) for each (weakly informative)
        log_prior_p1 = Beta(2, 2).log_prob(p1)
        log_prior_p2 = Beta(2, 2).log_prob(p2)

        # Additional prior to keep p1 + p2 < 1
        # This is implicit in the hard constraint above

        # Likelihood: Categorical for each observation
        probs = mx.array([float(p1), float(p2), float(p3)])
        log_likelihood = mx.array(0.0)
        for choice in choices:
            log_likelihood = log_likelihood + Categorical(probs=probs).log_prob(mx.array(choice))

        return log_prior_p1 + log_prior_p2 + log_likelihood

    # Create MCMC sampler
    mcmc = MCMC(log_prob)

    # Run sampling
    print("Running MCMC sampling...")
    samples = mcmc.run(
        initial_params={'p1': 0.33, 'p2': 0.33},
        num_samples=5000,
        num_warmup=1000,
        proposal_scale=0.02  # Small scale for bounded parameters
    )

    # Compute p3 from samples
    samples['p3'] = 1.0 - samples['p1'] - samples['p2']

    # Print summary
    print("\nPosterior Summaries:")
    summary = mcmc.summary()
    for i, param in enumerate(['p1', 'p2', 'p3']):
        if param in summary:
            mean_val = summary[param]['mean']
            std_val = summary[param]['std']
            ci_lower = summary[param]['ci_lower']
            ci_upper = summary[param]['ci_upper']
        else:
            mean_val = np.mean(samples[param])
            std_val = np.std(samples[param])
            ci_lower = np.percentile(samples[param], 2.5)
            ci_upper = np.percentile(samples[param], 97.5)

        print(f"  Product {i}:")
        print(f"    True: {true_probs[i]:.3f}")
        print(f"    Posterior: {mean_val:.3f} ± {std_val:.3f}")
        print(f"    95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    # Compute derived quantities
    print("\nDerived Quantities:")
    p_prod0_most_popular = np.mean(
        (samples['p1'] > samples['p2']) & (samples['p1'] > samples['p3'])
    )
    print(f"  P(Product 0 is most popular) = {p_prod0_most_popular:.3f}")

    # Visualize results
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Trace plots
    for i, param in enumerate(['p1', 'p2', 'p3']):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(samples[param], alpha=0.5, linewidth=0.5)
        ax.axhline(true_probs[i], color='red', linestyle='--',
                  linewidth=2, label='True value')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'p{i+1}')
        ax.set_title(f'Trace Plot: Product {i}')
        ax.legend()
        ax.grid(alpha=0.3)

    # Posterior distributions
    for i, param in enumerate(['p1', 'p2', 'p3']):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(samples[param], bins=50, alpha=0.7, density=True,
               color=f'C{i}', edgecolor='black', linewidth=0.5)
        ax.axvline(true_probs[i], color='red', linestyle='--',
                  linewidth=2, label='True value')
        ax.axvline(np.mean(samples[param]), color='blue', linestyle='-',
                  linewidth=2, label='Posterior mean')
        ax.set_xlabel(f'p{i+1}')
        ax.set_ylabel('Density')
        ax.set_title(f'Posterior: Product {i}')
        ax.legend()
        ax.grid(alpha=0.3)

    # Joint distribution: p1 vs p2
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(samples['p1'], samples['p2'], alpha=0.1, s=1)
    ax.scatter([true_probs[0]], [true_probs[1]], color='red',
              s=100, marker='*', label='True values', zorder=5)
    # Add constraint line
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, 1 - x_line, 'r--', linewidth=2, alpha=0.5, label='p1 + p2 = 1')
    ax.set_xlabel('p1 (Product 0)')
    ax.set_ylabel('p2 (Product 1)')
    ax.set_title('Joint Posterior: p1 vs p2')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Bar plot: True vs Estimated
    ax = fig.add_subplot(gs[2, 1])
    x_pos = np.arange(3)
    width = 0.35
    ax.bar(x_pos - width/2, true_probs, width, label='True', alpha=0.7)
    estimated_probs = [np.mean(samples[f'p{i+1}']) for i in range(3)]
    ax.bar(x_pos + width/2, estimated_probs, width, label='Estimated', alpha=0.7)
    ax.set_xlabel('Product')
    ax.set_ylabel('Probability')
    ax.set_title('True vs Estimated Probabilities')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Product {i}' for i in range(3)])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Posterior predictive check
    ax = fig.add_subplot(gs[2, 2])

    # Observed counts
    ax.bar(x_pos - width/2, counts, width, label='Observed', alpha=0.7, color='gray')

    # Expected counts from posterior
    n_pred_samples = 1000
    predicted_counts = np.zeros((n_pred_samples, 3))
    for j in range(n_pred_samples):
        idx = np.random.randint(len(samples['p1']))
        pred_probs = [samples['p1'][idx], samples['p2'][idx], samples['p3'][idx]]
        pred_choices = np.random.choice(3, size=n_customers, p=pred_probs)
        for i in range(3):
            predicted_counts[j, i] = np.sum(pred_choices == i)

    mean_pred = predicted_counts.mean(axis=0)
    ax.bar(x_pos + width/2, mean_pred, width, label='Predicted', alpha=0.7, color='green')

    ax.set_xlabel('Product')
    ax.set_ylabel('Count')
    ax.set_title('Posterior Predictive Check')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Product {i}' for i in range(3)])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle('MLX-MCMC: Categorical Model (Product Choice)', fontsize=16)

    output_file = '05_categorical_model_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")

    print("\n" + "="*70)
    print("✅ Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
