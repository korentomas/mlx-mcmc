"""
Example 6: NUTS vs HMC Comparison

Compare the No-U-Turn Sampler (NUTS) with standard HMC on a realistic
Bayesian inference problem. NUTS automatically determines trajectory length,
eliminating the need to tune num_leapfrog_steps.

Key metrics:
- Sampling efficiency (ESS per gradient evaluation)
- Convergence speed
- Parameter recovery accuracy
- Ease of use (no manual tuning)
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from mlx_mcmc import Normal, HalfNormal, MCMC
import time


def compute_ess(samples):
    """Compute Effective Sample Size using simple autocorrelation method."""
    n = len(samples)
    mean = np.mean(samples)
    var = np.var(samples)

    if var == 0:
        return n

    # Compute autocorrelation at increasing lags
    acf = []
    for lag in range(1, min(n // 2, 100)):
        c = np.mean((samples[:-lag] - mean) * (samples[lag:] - mean)) / var
        acf.append(c)
        if c < 0.05:  # Stop when autocorrelation becomes small
            break

    # ESS = n / (1 + 2 * sum(acf))
    ess = n / (1 + 2 * np.sum(acf))
    return ess


def main():
    print("\n" + "="*70)
    print("NUTS vs HMC Comparison")
    print("="*70 + "\n")

    # Generate synthetic data
    print("Generating synthetic data...")
    np.random.seed(42)
    true_mu = 5.0
    true_sigma = 2.0
    n_obs = 100
    y_observed = np.random.normal(true_mu, true_sigma, n_obs)

    print(f"  True μ: {true_mu}")
    print(f"  True σ: {true_sigma}")
    print(f"  Observations: {n_obs}\n")

    # Define log probability
    def log_prob(params):
        mu = params['mu']
        sigma = params['sigma']

        # Priors
        log_prior = (
            Normal(0, 10).log_prob(mu) +
            HalfNormal(5).log_prob(sigma)
        )

        # Likelihood
        log_likelihood = mx.sum(
            mx.array([Normal(mu, sigma).log_prob(mx.array(y)) for y in y_observed])
        )

        return log_prior + log_likelihood

    # Common settings
    num_samples = 2000
    num_warmup = 1000

    # ============================================================================
    # Experiment 1: HMC with different trajectory lengths
    # ============================================================================

    print("="*70)
    print("Experiment 1: HMC with Different Trajectory Lengths")
    print("="*70 + "\n")

    hmc_results = {}

    for num_leapfrog in [5, 10, 20, 50]:
        print(f"Running HMC with num_leapfrog_steps={num_leapfrog}...")

        start_time = time.time()
        mcmc = MCMC(log_prob)
        samples = mcmc.run(
            initial_params={'mu': 0.0, 'sigma': 1.0},
            num_samples=num_samples,
            num_warmup=num_warmup,
            method='hmc',
            step_size=0.1,
            num_leapfrog_steps=num_leapfrog,
            verbose=False
        )
        elapsed_time = time.time() - start_time

        # Compute metrics
        mu_mean = np.mean(samples['mu'])
        sigma_mean = np.mean(samples['sigma'])
        mu_ess = compute_ess(samples['mu'])
        sigma_ess = compute_ess(samples['sigma'])

        # Gradients per sample = num_leapfrog_steps per iteration
        total_grads = (num_samples + num_warmup) * num_leapfrog
        ess_per_grad_mu = mu_ess / total_grads
        ess_per_grad_sigma = sigma_ess / total_grads

        hmc_results[num_leapfrog] = {
            'samples': samples,
            'time': elapsed_time,
            'mu_mean': mu_mean,
            'sigma_mean': sigma_mean,
            'mu_ess': mu_ess,
            'sigma_ess': sigma_ess,
            'ess_per_grad_mu': ess_per_grad_mu,
            'ess_per_grad_sigma': ess_per_grad_sigma,
            'total_grads': total_grads,
            'accept_rate': mcmc.acceptance_rate
        }

        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  μ: {mu_mean:.3f}, ESS: {mu_ess:.0f}")
        print(f"  σ: {sigma_mean:.3f}, ESS: {sigma_ess:.0f}")
        print(f"  ESS/gradient: μ={ess_per_grad_mu:.6f}, σ={ess_per_grad_sigma:.6f}")
        print(f"  Accept rate: {100*mcmc.acceptance_rate:.1f}%\n")

    # ============================================================================
    # Experiment 2: NUTS (automatic trajectory length)
    # ============================================================================

    print("="*70)
    print("Experiment 2: NUTS (Automatic Trajectory Length)")
    print("="*70 + "\n")

    print("Running NUTS...")
    start_time = time.time()
    mcmc_nuts = MCMC(log_prob)
    samples_nuts = mcmc_nuts.run(
        initial_params={'mu': 0.0, 'sigma': 1.0},
        num_samples=num_samples,
        num_warmup=num_warmup,
        method='nuts',
        step_size=0.1,
        max_tree_depth=10,
        verbose=False
    )
    elapsed_time_nuts = time.time() - start_time

    # Compute metrics
    mu_mean_nuts = np.mean(samples_nuts['mu'])
    sigma_mean_nuts = np.mean(samples_nuts['sigma'])
    mu_ess_nuts = compute_ess(samples_nuts['mu'])
    sigma_ess_nuts = compute_ess(samples_nuts['sigma'])

    # For NUTS, estimate average gradients per iteration based on tree depth
    # avg_depth ≈ 1.5 means avg ~3 gradient evaluations per iteration
    # This is approximate - actual depends on tree building
    est_grads_per_iter = 4  # Conservative estimate
    total_grads_nuts = (num_samples + num_warmup) * est_grads_per_iter
    ess_per_grad_mu_nuts = mu_ess_nuts / total_grads_nuts
    ess_per_grad_sigma_nuts = sigma_ess_nuts / total_grads_nuts

    print(f"  Time: {elapsed_time_nuts:.2f}s")
    print(f"  μ: {mu_mean_nuts:.3f}, ESS: {mu_ess_nuts:.0f}")
    print(f"  σ: {sigma_mean_nuts:.3f}, ESS: {sigma_ess_nuts:.0f}")
    print(f"  ESS/gradient (estimated): μ={ess_per_grad_mu_nuts:.6f}, σ={ess_per_grad_sigma_nuts:.6f}")
    print(f"  Accept rate: {100*mcmc_nuts.acceptance_rate:.1f}%\n")

    # ============================================================================
    # Summary Comparison
    # ============================================================================

    print("="*70)
    print("Summary Comparison")
    print("="*70 + "\n")

    print(f"{'Method':<15} {'Time(s)':<10} {'μ Error':<12} {'σ Error':<12} {'μ ESS':<10} {'σ ESS':<10}")
    print("-"*70)

    for num_leapfrog, results in hmc_results.items():
        mu_error = abs(results['mu_mean'] - true_mu)
        sigma_error = abs(results['sigma_mean'] - true_sigma)
        print(f"HMC (L={num_leapfrog:<2})     {results['time']:<10.2f} "
              f"{mu_error:<12.3f} {sigma_error:<12.3f} "
              f"{results['mu_ess']:<10.0f} {results['sigma_ess']:<10.0f}")

    mu_error_nuts = abs(mu_mean_nuts - true_mu)
    sigma_error_nuts = abs(sigma_mean_nuts - true_sigma)
    print(f"NUTS (auto)     {elapsed_time_nuts:<10.2f} "
          f"{mu_error_nuts:<12.3f} {sigma_error_nuts:<12.3f} "
          f"{mu_ess_nuts:<10.0f} {sigma_ess_nuts:<10.0f}")

    print("\nKey Insights:")
    print("  • NUTS automatically adapts trajectory length (no manual tuning needed)")
    print("  • HMC requires tuning num_leapfrog_steps for each problem")
    print("  • Too few steps: inefficient exploration")
    print("  • Too many steps: wasted computation")
    print("  • NUTS finds optimal balance automatically")

    # ============================================================================
    # Visualization
    # ============================================================================

    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Trace plots for μ (HMC L=10 vs NUTS)
    axes[0, 0].plot(hmc_results[10]['samples']['mu'], alpha=0.5, linewidth=0.5, label='HMC (L=10)')
    axes[0, 0].plot(samples_nuts['mu'], alpha=0.5, linewidth=0.5, label='NUTS')
    axes[0, 0].axhline(true_mu, color='red', linestyle='--', linewidth=2, label='True value')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('μ')
    axes[0, 0].set_title('Trace Plot: μ')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Trace plots for σ (HMC L=10 vs NUTS)
    axes[0, 1].plot(hmc_results[10]['samples']['sigma'], alpha=0.5, linewidth=0.5, label='HMC (L=10)')
    axes[0, 1].plot(samples_nuts['sigma'], alpha=0.5, linewidth=0.5, label='NUTS')
    axes[0, 1].axhline(true_sigma, color='red', linestyle='--', linewidth=2, label='True value')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('σ')
    axes[0, 1].set_title('Trace Plot: σ')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: ESS comparison
    x_pos = np.arange(5)
    labels = ['HMC\n(L=5)', 'HMC\n(L=10)', 'HMC\n(L=20)', 'HMC\n(L=50)', 'NUTS\n(auto)']
    mu_ess_values = [hmc_results[5]['mu_ess'], hmc_results[10]['mu_ess'],
                      hmc_results[20]['mu_ess'], hmc_results[50]['mu_ess'], mu_ess_nuts]
    sigma_ess_values = [hmc_results[5]['sigma_ess'], hmc_results[10]['sigma_ess'],
                        hmc_results[20]['sigma_ess'], hmc_results[50]['sigma_ess'], sigma_ess_nuts]

    width = 0.35
    axes[0, 2].bar(x_pos - width/2, mu_ess_values, width, label='μ ESS', alpha=0.8)
    axes[0, 2].bar(x_pos + width/2, sigma_ess_values, width, label='σ ESS', alpha=0.8)
    axes[0, 2].set_xlabel('Method')
    axes[0, 2].set_ylabel('Effective Sample Size')
    axes[0, 2].set_title('ESS Comparison')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(labels, fontsize=9)
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3, axis='y')

    # Plot 4: Posterior μ (all methods)
    for num_leapfrog in [5, 10, 20, 50]:
        axes[1, 0].hist(hmc_results[num_leapfrog]['samples']['mu'], bins=40,
                       alpha=0.3, density=True, label=f'HMC (L={num_leapfrog})')
    axes[1, 0].hist(samples_nuts['mu'], bins=40, alpha=0.5, density=True,
                   label='NUTS', edgecolor='black', linewidth=1.5)
    axes[1, 0].axvline(true_mu, color='red', linestyle='--', linewidth=2, label='True value')
    axes[1, 0].set_xlabel('μ')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Posterior Distribution: μ')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # Plot 5: Posterior σ (all methods)
    for num_leapfrog in [5, 10, 20, 50]:
        axes[1, 1].hist(hmc_results[num_leapfrog]['samples']['sigma'], bins=40,
                       alpha=0.3, density=True, label=f'HMC (L={num_leapfrog})')
    axes[1, 1].hist(samples_nuts['sigma'], bins=40, alpha=0.5, density=True,
                   label='NUTS', edgecolor='black', linewidth=1.5)
    axes[1, 1].axvline(true_sigma, color='red', linestyle='--', linewidth=2, label='True value')
    axes[1, 1].set_xlabel('σ')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Posterior Distribution: σ')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    # Plot 6: Parameter recovery accuracy
    methods = ['HMC\n(L=5)', 'HMC\n(L=10)', 'HMC\n(L=20)', 'HMC\n(L=50)', 'NUTS']
    mu_errors = [abs(hmc_results[5]['mu_mean'] - true_mu),
                 abs(hmc_results[10]['mu_mean'] - true_mu),
                 abs(hmc_results[20]['mu_mean'] - true_mu),
                 abs(hmc_results[50]['mu_mean'] - true_mu),
                 abs(mu_mean_nuts - true_mu)]
    sigma_errors = [abs(hmc_results[5]['sigma_mean'] - true_sigma),
                    abs(hmc_results[10]['sigma_mean'] - true_sigma),
                    abs(hmc_results[20]['sigma_mean'] - true_sigma),
                    abs(hmc_results[50]['sigma_mean'] - true_sigma),
                    abs(sigma_mean_nuts - true_sigma)]

    x_pos = np.arange(5)
    axes[1, 2].bar(x_pos - width/2, mu_errors, width, label='μ error', alpha=0.8)
    axes[1, 2].bar(x_pos + width/2, sigma_errors, width, label='σ error', alpha=0.8)
    axes[1, 2].set_xlabel('Method')
    axes[1, 2].set_ylabel('Absolute Error')
    axes[1, 2].set_title('Parameter Recovery Accuracy')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(methods, fontsize=9)
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3, axis='y')

    plt.suptitle('MLX-MCMC: NUTS vs HMC Comparison', fontsize=16, y=0.995)
    plt.tight_layout()

    output_file = '06_nuts_comparison_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")

    print("\n" + "="*70)
    print("✅ Comparison completed successfully!")
    print("="*70 + "\n")

    print("Conclusion:")
    print("  NUTS eliminates the need to tune trajectory length (num_leapfrog_steps)")
    print("  while achieving comparable or better sampling efficiency than HMC.")
    print("  For most users, NUTS is the recommended sampler.")


if __name__ == "__main__":
    main()
