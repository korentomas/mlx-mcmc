"""High-level MCMC inference interface."""

import numpy as np
import mlx.core as mx
from mlx_mcmc.kernels.metropolis import metropolis_hastings
from mlx_mcmc.kernels.hmc import hmc
from mlx_mcmc.kernels.nuts import nuts


class MCMC:
    """High-level MCMC inference interface.

    Provides a clean API for running MCMC sampling with various algorithms.

    Parameters
    ----------
    log_prob_fn : callable
        Function that computes log probability given parameters dict

    Examples
    --------
    >>> from mlx_mcmc import Normal, MCMC
    >>>
    >>> def log_prob(params):
    ...     mu = params['mu']
    ...     return Normal(0, 10).log_prob(mu)
    >>>
    >>> mcmc = MCMC(log_prob)
    >>> samples = mcmc.run({'mu': 0.0}, num_samples=1000)
    >>> print(f"Mean: {np.mean(samples['mu']):.3f}")
    """

    def __init__(self, log_prob_fn):
        self.log_prob_fn = log_prob_fn
        self.samples = None
        self.acceptance_rate = None

    def run(
        self,
        initial_params,
        num_samples=1000,
        num_warmup=1000,
        method='metropolis',
        proposal_scale=0.1,
        random_seed=0,
        verbose=True,
        **kwargs
    ):
        """
        Run MCMC sampling.

        Parameters
        ----------
        initial_params : dict
            Initial parameter values {name: value}
        num_samples : int, optional
            Number of samples to draw after warmup (default: 1000)
        num_warmup : int, optional
            Number of warmup samples to discard (default: 1000)
        method : str, optional
            Sampling method: 'metropolis', 'hmc', or 'nuts' (default: 'metropolis')
        proposal_scale : float, optional
            Scale of proposal distribution for Metropolis (default: 0.1)
        random_seed : int, optional
            Random seed for reproducibility (default: 0)
        verbose : bool, optional
            If True, print progress information (default: True)
        **kwargs
            Additional arguments passed to the sampler:
            - For HMC: step_size, num_leapfrog_steps, adapt_step_size, target_accept
            - For NUTS: step_size, max_tree_depth, adapt_step_size, target_accept

        Returns
        -------
        samples : dict
            Dictionary of parameter samples {name: np.array of values}

        Raises
        ------
        ValueError
            If unknown sampling method specified
        """
        if method == 'hmc':
            # HMC has built-in warmup and sampling
            if verbose:
                print(f"\n{'='*70}")
                print(f"MLX-MCMC: HMC Sampling")
                print(f"{'='*70}\n")

            samples, accept_rate = hmc(
                self.log_prob_fn,
                initial_params,
                num_samples=num_samples,
                num_warmup=num_warmup,
                key=mx.random.key(random_seed),
                **kwargs
            )

            self.samples = {k: np.array(v) for k, v in samples.items()}
            self.acceptance_rate = accept_rate

            if verbose:
                print(f"\n{'='*70}")
                print("Sampling complete!")
                print(f"{'='*70}\n")

            return self.samples

        elif method == 'nuts':
            # NUTS has built-in warmup and sampling
            if verbose:
                print(f"\n{'='*70}")
                print(f"MLX-MCMC: NUTS Sampling")
                print(f"{'='*70}\n")

            samples, accept_rate = nuts(
                self.log_prob_fn,
                initial_params,
                num_samples=num_samples,
                num_warmup=num_warmup,
                key=mx.random.key(random_seed),
                **kwargs
            )

            self.samples = {k: np.array(v) for k, v in samples.items()}
            self.acceptance_rate = accept_rate

            if verbose:
                print(f"\n{'='*70}")
                print("Sampling complete!")
                print(f"{'='*70}\n")

            return self.samples

        elif method == 'metropolis':
            sampler = metropolis_hastings
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        if verbose:
            print(f"\n{'='*70}")
            print(f"MLX-MCMC: {method.upper()} Sampling")
            print(f"{'='*70}\n")

        # Warmup phase
        if num_warmup > 0:
            if verbose:
                print(f"Warmup phase: {num_warmup} samples")
            warmup_samples, warmup_accept = sampler(
                self.log_prob_fn,
                initial_params,
                num_samples=num_warmup,
                proposal_scale=proposal_scale,
                random_seed=random_seed,
                verbose=verbose,
                **kwargs
            )
            if verbose:
                print(f"Warmup acceptance rate: {warmup_accept:.2%}\n")

            # Use last warmup sample as starting point
            final_warmup = {k: v[-1] for k, v in warmup_samples.items()}
        else:
            final_warmup = initial_params
            warmup_accept = None

        # Sampling phase
        if verbose:
            print(f"Sampling phase: {num_samples} samples")
        self.samples, self.acceptance_rate = sampler(
            self.log_prob_fn,
            final_warmup,
            num_samples=num_samples,
            proposal_scale=proposal_scale,
            random_seed=random_seed + 1 if num_warmup > 0 else random_seed,
            verbose=verbose,
            **kwargs
        )

        if verbose:
            print(f"Sampling acceptance rate: {self.acceptance_rate:.2%}")
            print(f"\n{'='*70}")
            print("Sampling complete!")
            print(f"{'='*70}\n")

        # Convert to numpy arrays
        self.samples = {k: np.array(v) for k, v in self.samples.items()}

        return self.samples

    def summary(self, credible_interval=0.95):
        """
        Compute summary statistics for samples.

        Parameters
        ----------
        credible_interval : float, optional
            Credible interval width (default: 0.95 for 95% CI)

        Returns
        -------
        summary : dict
            Dictionary with summary statistics for each parameter

        Raises
        ------
        ValueError
            If sampling hasn't been run yet
        """
        if self.samples is None:
            raise ValueError("Must run sampling first. Call run() method.")

        alpha = 1 - credible_interval
        lower_pct = 100 * alpha / 2
        upper_pct = 100 * (1 - alpha / 2)

        summary = {}
        for param_name, param_samples in self.samples.items():
            summary[param_name] = {
                'mean': float(np.mean(param_samples)),
                'std': float(np.std(param_samples)),
                'median': float(np.median(param_samples)),
                f'{lower_pct:.1f}%': float(np.percentile(param_samples, lower_pct)),
                f'{upper_pct:.1f}%': float(np.percentile(param_samples, upper_pct)),
            }

        return summary

    def print_summary(self, credible_interval=0.95):
        """Print summary statistics in a formatted table."""
        summary = self.summary(credible_interval)

        print("\nPosterior Summary:")
        print("="*80)
        print(f"{'Parameter':<15} {'Mean':<10} {'Std':<10} {'Median':<10} {f'{int(credible_interval*100)}% CI':<20}")
        print("-"*80)

        for param_name, stats in summary.items():
            ci_lower = list(stats.values())[3]  # Lower percentile
            ci_upper = list(stats.values())[4]  # Upper percentile
            ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"

            print(f"{param_name:<15} {stats['mean']:<10.3f} {stats['std']:<10.3f} "
                  f"{stats['median']:<10.3f} {ci_str:<20}")

        print("="*80)
