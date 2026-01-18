"""Hamiltonian Monte Carlo (HMC) kernel implementation using MLX."""

import mlx.core as mx
from typing import Callable, Tuple, Dict


def hmc(
    log_prob_fn: Callable,
    initial_params: Dict[str, mx.array],
    num_samples: int = 1000,
    num_warmup: int = 1000,
    step_size: float = 0.1,
    num_leapfrog_steps: int = 10,
    adapt_step_size: bool = True,
    target_accept: float = 0.8,
    key: mx.array = None,
) -> Tuple[Dict[str, mx.array], float]:
    """Hamiltonian Monte Carlo sampler using gradient information.

    HMC uses the gradient of the log probability to efficiently explore
    the posterior distribution. It simulates Hamiltonian dynamics where
    parameters are particle positions and introduces auxiliary momentum
    variables.

    Args:
        log_prob_fn: Function that computes log probability given parameters.
        initial_params: Dictionary of initial parameter values.
        num_samples: Number of samples to draw after warmup.
        num_warmup: Number of warmup samples for step size adaptation.
        step_size: Initial step size for leapfrog integrator.
        num_leapfrog_steps: Number of leapfrog steps per HMC iteration.
        adapt_step_size: Whether to adapt step size during warmup.
        target_accept: Target acceptance rate for step size adaptation.
        key: Random key for reproducibility.

    Returns:
        samples: Dictionary mapping parameter names to arrays of samples.
        acceptance_rate: Final acceptance rate.
    """
    if key is None:
        key = mx.random.key(0)

    # Initialize
    current_params = {k: mx.array(v) for k, v in initial_params.items()}
    param_names = list(current_params.keys())

    # Storage for samples
    all_samples = {name: [] for name in param_names}
    n_accept = 0
    n_total = 0

    # Define gradient function
    def grad_log_prob(params_dict):
        """Compute gradients of log probability with respect to all parameters."""
        # Create a function that takes a flat array and returns log_prob
        def log_prob_flat(*param_values):
            params = {name: val for name, val in zip(param_names, param_values)}
            return log_prob_fn(params)

        # Get gradients for each parameter
        param_list = [params_dict[name] for name in param_names]
        grads = mx.grad(log_prob_flat, argnums=list(range(len(param_names))))(*param_list)

        # Return as dictionary
        if not isinstance(grads, tuple):
            grads = (grads,)
        return {name: grad for name, grad in zip(param_names, grads)}

    def leapfrog_step(params, momentum, epsilon):
        """Single leapfrog integration step.

        Args:
            params: Current parameter values (position)
            momentum: Current momentum values
            epsilon: Step size

        Returns:
            Updated params and momentum
        """
        # Half step for momentum
        grads = grad_log_prob(params)
        momentum_half = {
            name: momentum[name] + 0.5 * epsilon * grads[name]
            for name in param_names
        }

        # Full step for position
        params_new = {
            name: params[name] + epsilon * momentum_half[name]
            for name in param_names
        }

        # Half step for momentum
        grads_new = grad_log_prob(params_new)
        momentum_new = {
            name: momentum_half[name] + 0.5 * epsilon * grads_new[name]
            for name in param_names
        }

        return params_new, momentum_new

    def hamiltonian(params, momentum):
        """Compute Hamiltonian (negative log joint probability).

        H(q, p) = -log p(q) + 0.5 * ||p||^2

        where q are parameters and p is momentum.
        """
        potential_energy = -log_prob_fn(params)
        kinetic_energy = 0.5 * sum(mx.sum(p ** 2) for p in momentum.values())
        return potential_energy + kinetic_energy

    def hmc_step(params, epsilon, key):
        """Single HMC step with Metropolis acceptance."""
        # Sample momentum from standard normal
        key, *subkeys = mx.random.split(key, len(param_names) + 1)
        momentum = {
            name: mx.random.normal(params[name].shape, key=subkey)
            for name, subkey in zip(param_names, subkeys)
        }

        # Store initial state
        params_init = {name: mx.array(val) for name, val in params.items()}
        momentum_init = {name: mx.array(val) for name, val in momentum.items()}

        # Compute initial Hamiltonian
        H_init = hamiltonian(params_init, momentum_init)

        # Leapfrog integration
        params_prop = params_init
        momentum_prop = momentum_init
        for _ in range(num_leapfrog_steps):
            params_prop, momentum_prop = leapfrog_step(params_prop, momentum_prop, epsilon)

        # Negate momentum for reversibility (optional, maintains detailed balance)
        momentum_prop = {name: -val for name, val in momentum_prop.items()}

        # Compute proposed Hamiltonian
        H_prop = hamiltonian(params_prop, momentum_prop)

        # Metropolis acceptance
        log_accept_ratio = -(H_prop - H_init)  # Note: negative because H = -log_prob

        # Sample uniform for acceptance
        key, subkey = mx.random.split(key)
        log_u = mx.log(mx.random.uniform(shape=(), key=subkey))

        # Accept or reject
        accept = log_u < log_accept_ratio
        if float(accept):
            return params_prop, True, key
        else:
            return params_init, False, key

    # Warmup phase
    print(f"Warmup phase: {num_warmup} samples")
    epsilon = step_size

    for i in range(num_warmup):
        current_params, accepted, key = hmc_step(current_params, epsilon, key)
        n_accept += int(accepted)
        n_total += 1

        # Adapt step size
        if adapt_step_size and i > 10:
            current_accept_rate = n_accept / n_total
            if current_accept_rate < target_accept:
                epsilon *= 0.95  # Decrease step size
            else:
                epsilon *= 1.05  # Increase step size

        if (i + 1) % 500 == 0:
            print(f"  Iteration {i+1}/{num_warmup} (accept rate: {100*n_accept/n_total:.2f}%, step_size: {epsilon:.4f})")

    warmup_accept_rate = n_accept / n_total
    print(f"Warmup acceptance rate: {100*warmup_accept_rate:.2f}% (final step_size: {epsilon:.4f})")

    # Reset counters for sampling phase
    n_accept = 0
    n_total = 0

    # Sampling phase
    print(f"\nSampling phase: {num_samples} samples")

    for i in range(num_samples):
        current_params, accepted, key = hmc_step(current_params, epsilon, key)
        n_accept += int(accepted)
        n_total += 1

        # Store sample
        for name in param_names:
            all_samples[name].append(float(current_params[name]))

        if (i + 1) % 500 == 0:
            print(f"  Iteration {i+1}/{num_samples} (accept rate: {100*n_accept/n_total:.2f}%)")

    sampling_accept_rate = n_accept / n_total
    print(f"Sampling acceptance rate: {100*sampling_accept_rate:.2f}%")

    # Convert to arrays
    samples = {
        name: mx.array(all_samples[name])
        for name in param_names
    }

    return samples, sampling_accept_rate
